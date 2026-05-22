import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1),
                 mid_channels=None, last_act=False):
        super(InceptionBlock, self).__init__()
        self.last_act = last_act
        if mid_channels is None:
            mid_channels = in_channels * 4

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=stride,
                      padding=(0, 0), bias=False),
            nn.SELU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.SELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride,
                      padding=(1, 1), bias=False),
            nn.SELU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.SELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(1, 3),
                      stride=stride, padding=(0, 1), bias=False),
            nn.SELU()
        )

        self.merge_conv = nn.Conv2d(mid_channels * 3, out_channels,
                                    kernel_size=1, stride=1, padding=0,
                                    bias=False)

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
            )
        else:
            self.shortcut = nn.Identity()

        self.act = nn.SELU()

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.merge_conv(out)
        out += self.shortcut(x)
        if self.last_act:
            out = self.act(out)
        return out


class ASPP_1D_Dilation(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[1, 2, 4, 8],
                 kernel_size=3):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            padding_h = (kernel_size - 1) * d // 2
            padding = (padding_h, kernel_size // 2)
            self.branches.append(
                nn.Sequential(
                    nn.ConstantPad2d((padding[1], padding[1],
                                      padding[0], padding[0]), 0),
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=(kernel_size, kernel_size),
                              dilation=(d, 1), bias=False),
                    nn.SELU()
                )
            )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * len(dilations), out_channels,
                      kernel_size=1, bias=False),
        )

    def forward(self, x):
        out = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.project(out)


class TinyTimeMixerMLP(nn.Module):
    def __init__(self, in_features, out_features, expansion_factor=4,
                 dropout_p=0.1, bn=False):
        super().__init__()
        num_hidden = in_features * expansion_factor
        self.fc1 = nn.Linear(in_features, num_hidden)
        self.dropout1 = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(num_hidden, out_features)
        self.dropout2 = nn.Dropout(dropout_p)
        if bn:
            self.tbn = TinyTimeMixerBatchNorm(in_features)
        else:
            self.tbn = nn.Identity()

    def forward(self, inputs):
        inputs = self.tbn(inputs)
        inputs = self.dropout1(F.gelu(self.fc1(inputs)))
        inputs = self.fc2(inputs)
        inputs = self.dropout2(inputs)
        return inputs


class TinyTimeMixerBatchNorm(nn.Module):
    """BatchNorm over the time dimension of [B, C, F, T] inputs."""

    def __init__(self, chl):
        super().__init__()
        self.bn = nn.BatchNorm1d(chl)

    def forward(self, inputs):
        B, C, F_, T = inputs.shape
        inputs = inputs.view(B, C * F_, T)
        inputs = inputs.transpose(1, 2)
        inputs = self.bn(inputs)
        inputs = inputs.transpose(1, 2)
        return inputs.view(B, C, F_, T)


class Tmixer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Tmixer, self).__init__()
        self.fc1 = TinyTimeMixerMLP(in_dim, out_dim)
        self.res = (in_dim == out_dim)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        if self.res:
            x = x + residual
        return x


class Fmixer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Fmixer, self).__init__()
        self.fc1 = TinyTimeMixerMLP(in_dim, out_dim)
        self.res = (in_dim == out_dim)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        if self.res:
            x = x + residual
        return x


class Cmixer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Cmixer, self).__init__()
        self.fc1 = TinyTimeMixerMLP(in_dim, out_dim)
        self.res = (in_dim == out_dim)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        if self.res:
            x = x + residual
        return x


class LoraMixer(nn.Module):
    def __init__(self, T_in, T_out, F_in_dim, F_out_dim, c_in, c_out):
        super(LoraMixer, self).__init__()
        self.tm = Tmixer(T_in, T_out)
        self.tm2 = Tmixer(T_in, T_out)
        assert T_in == T_out
        self.fm = Fmixer(F_in_dim, F_out_dim)
        self.cm = Cmixer(c_in, c_out)

    def forward(self, x):
        x = self.tm(x)
        x = x.transpose(3, 2).contiguous()
        x = self.fm(x)
        x = x.transpose(2, 3).contiguous()
        x = self.tm2(x)
        x = x.transpose(3, 1).contiguous()
        x = self.cm(x)
        x = x.transpose(1, 3).contiguous()
        return x


class SNREstimator(nn.Module):
    def __init__(self, num_classes=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        snr_embed = x
        x = self.classifier(x)
        return x, snr_embed


class ASPP_1D_Dilation_backbone(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[1, 2, 4, 8],
                 kernel_size=3):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            padding_h = (kernel_size - 1) * d // 2
            padding = (padding_h, kernel_size // 2)
            self.branches.append(
                nn.Sequential(
                    nn.ConstantPad2d((padding[1], padding[1],
                                      padding[0], padding[0]), 0),
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=(kernel_size, kernel_size),
                              dilation=(d, 1), bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.SELU()
                )
            )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * len(dilations), out_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SELU()
        )

    def forward(self, x):
        out = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.project(out) + x


class Backbone(nn.Module):
    def __init__(self, sf):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ZeroPad2d((3, 3, 0, 0)),
            nn.Conv2d(2, 64, kernel_size=(1, 7), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            nn.ZeroPad2d((0, 0, 3, 3)),
            nn.Conv2d(64, 64, kernel_size=(7, 1), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            nn.ZeroPad2d(2),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            nn.ZeroPad2d((2, 2, 4, 4)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(2, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            nn.ZeroPad2d((2, 2, 8, 8)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(4, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            nn.ZeroPad2d((2, 2, 16, 16)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(8, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            nn.ZeroPad2d((2, 2, 32, 32)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(16, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            nn.Conv2d(64, 8, kernel_size=(1, 1), dilation=(1, 1)),
            nn.BatchNorm2d(8), nn.ReLU(),
        )

    @staticmethod
    def _build_aspp_block(sf, in_channels, out_channels):
        if sf == 10:
            return nn.Sequential(
                ASPP_1D_Dilation_backbone(in_channels, out_channels,
                                          dilations=[8, 16, 32, 64]),
                ASPP_1D_Dilation_backbone(out_channels, out_channels,
                                          dilations=[2, 4, 8, 16]),
            )
        return nn.Identity()

    def forward(self, x):
        x = x.transpose(2, 3).contiguous()
        x = self.conv(x)
        return x


class SNRMask(nn.Module):
    def __init__(self, conv_dim_lstm, lstm_dim, fc1_dim, freq_size):
        super().__init__()
        self.lstm = nn.LSTM(conv_dim_lstm, lstm_dim,
                            batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(2 * lstm_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, freq_size * 2)

    def forward(self, x):
        out = x.transpose(1, 2).contiguous()
        out = out.view(out.size(0), out.size(1), -1)
        out, _ = self.lstm(out)
        out = F.relu(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = out.view(out.size(0), out.size(1), 2, -1)
        out = torch.sigmoid(out)
        out = out.transpose(1, 2).contiguous()
        mask = out.transpose(2, 3).contiguous()
        return mask


class SNRConditionalMoE(nn.Module):
    def __init__(self, base_model_fn, num_experts=4, snr_embed_dim=32, topk=3):
        super().__init__()
        self.experts = nn.ModuleList([base_model_fn()
                                      for _ in range(num_experts)])
        self.gating = nn.Sequential(
            nn.Linear(snr_embed_dim, num_experts),
        )
        self.num_experts = num_experts
        self.topk = topk
        self.residual_expert = base_model_fn()

    def route_topk(self, expert_outputs, gate_weights, k=3, return_info=False):
        topk_vals, topk_idx = torch.topk(gate_weights, k=k, dim=1)
        topk_weights = torch.softmax(topk_vals, dim=1)
        B, E, C, H, W = expert_outputs.shape
        topk_idx_exp = topk_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)\
                                .expand(-1, -1, C, H, W)
        topk_outputs = torch.gather(expert_outputs, dim=1, index=topk_idx_exp)
        topk_weights_exp = topk_weights.view(B, k, 1, 1, 1)
        topk_result = (topk_outputs * topk_weights_exp).sum(dim=1)
        if return_info:
            return topk_result, topk_vals, topk_idx, topk_weights
        return topk_result

    def forward(self, x, snr_embed, return_info=False):
        gate_weights = self.gating(snr_embed)

        outputs = []
        for expert in self.experts:
            out = expert(x)
            outputs.append(out.unsqueeze(1))
        expert_outputs = torch.cat(outputs, dim=1)

        residual_output = self.residual_expert(x)

        # Optional ablation: uniform-weight averaging over all experts.
        if getattr(self, "uniform_routing", False):
            B = x.shape[0]
            N = self.num_experts
            topk_result = expert_outputs.mean(dim=1)
            out = topk_result + residual_output
            if return_info:
                fake_gate = torch.zeros(B, N, device=x.device)
                fake_probs = torch.full((B, N), 1.0 / N, device=x.device)
                fake_idx = torch.arange(N, device=x.device)\
                                .unsqueeze(0).expand(B, -1)
                fake_w = torch.full((B, N), 1.0 / N, device=x.device)
                moe_info = {
                    "gate_weights":    fake_gate,
                    "gate_probs":      fake_probs,
                    "topk_vals":       fake_gate,
                    "topk_idx":        fake_idx,
                    "topk_weights":    fake_w,
                    "topk_result":     topk_result,
                    "residual_output": residual_output,
                }
                return out, moe_info
            return out

        if return_info:
            topk_result, topk_vals, topk_idx, topk_weights = self.route_topk(
                expert_outputs, gate_weights, self.topk, return_info=True
            )
        else:
            topk_result = self.route_topk(expert_outputs, gate_weights,
                                          self.topk)
        out = topk_result + residual_output

        if return_info:
            moe_info = {
                "gate_weights": gate_weights,
                "gate_probs": torch.softmax(gate_weights, dim=1),
                "topk_vals": topk_vals,
                "topk_idx": topk_idx,
                "topk_weights": topk_weights,
                "topk_result": topk_result,
                "residual_output": residual_output,
            }
            return out, moe_info
        return out


class ChirpMixer(nn.Module):
    """Lightweight chirp classifier head with multi-scale conv + axis-wise mixer."""

    def __init__(self, config_dict):
        super(ChirpMixer, self).__init__()
        model_cfg = config_dict["ChirpMixer"]
        num_classes = config_dict['num_classes']

        out_t = model_cfg['out_dim_t']
        out_f = num_classes // model_cfg['out_dim_f_downsample1']
        out_f_div = model_cfg['out_dim_f_downsample2']
        out_chl = model_cfg['out_dim_chl']
        mixer_chl = out_t * out_f * out_chl // out_f_div
        base_dilations = [1, 2, 4, 8]
        scaled_dilations = self._get_scaled_dilations(
            sf=config_dict['sf'], base_dilations=base_dilations
        )
        self.sf = config_dict['sf']

        self.conv1 = InceptionBlock(2, 16, stride=(2, 2),
                                    mid_channels=32, last_act=True)
        self.aspp1 = ASPP_1D_Dilation(16, 32, dilations=base_dilations)
        self.pool1 = InceptionBlock(32, 32, stride=(2, 1),
                                    mid_channels=64, last_act=True)
        self.aspp2 = ASPP_1D_Dilation(32, out_chl, dilations=base_dilations)

        if self.sf >= 9:
            self.aspp1_2 = ASPP_1D_Dilation(32, 32, dilations=scaled_dilations)
            self.aspp2_0 = ASPP_1D_Dilation(32, 32, dilations=scaled_dilations)

        self.LoraMixer1 = LoraMixer(out_t, out_t, out_f, out_f,
                                    out_chl, out_chl)
        self.LoraMixer2 = LoraMixer(out_t, out_t, out_f, out_f // out_f_div,
                                    out_chl, out_chl)
        self.mixer = nn.Linear(mixer_chl, num_classes)
        self.act = nn.SELU()

    def _get_scaled_dilations(self, sf, base_dilations=[1, 2, 4, 8], base_sf=8):
        scale = 2 ** (sf - base_sf)
        return [int(d * scale) for d in base_dilations]

    @staticmethod
    def _add_coords(x):
        B, _, H, W = x.shape
        v = torch.linspace(-1, 1, H, device=x.device)\
                 .view(1, 1, H, 1).expand(B, 1, H, W)
        u = torch.linspace(-1, 1, W, device=x.device)\
                 .view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([x, u, v], 1)

    def forward(self, x):
        if self.sf >= 9:
            out = self.conv1(x)
            out = self.aspp1(out)
            out = self.aspp1_2(out)
            out = self.pool1(out)
            out = self.aspp2_0(out)
            out = self.aspp2(out)
        else:
            out = self.conv1(x)
            out = self.aspp1(out)
            out = self.pool1(out)
            out = self.aspp2(out)

        out = self.LoraMixer1(out)
        out = self.LoraMixer2(out)

        out = out.view(out.size(0), -1)
        out = self.mixer(out)
        return out


class SNRNet(nn.Module):
    """SAM-Denoiser: SNR-aware MoE that produces a denoising mask."""

    def __init__(self, config):
        super(SNRNet, self).__init__()
        model_cfg = config["SNRNet"]

        self.conv_dim_lstm = config["conv_dim_lstm"]
        self.freq_size = config["freq_size"]
        self.lstm_dim = model_cfg["lstm_dim"]
        self.fc1_dim = model_cfg["fc1_dim"]
        self.num_experts = model_cfg["num_experts"]

        self.feature = Backbone(config['sf'])
        self.snr_est = SNREstimator(model_cfg["num_snr_classes"])
        self.moe = SNRConditionalMoE(
            base_model_fn=self._build_snr_mask,
            num_experts=self.num_experts,
            snr_embed_dim=model_cfg["num_snr_classes"],
            topk=model_cfg["topk"]
        )

    def _build_snr_mask(self):
        return SNRMask(self.conv_dim_lstm, self.lstm_dim,
                       self.fc1_dim, self.freq_size)

    def forward(self, x, return_moe_info=False):
        snr_logits, _ = self.snr_est(x)
        conv_feature = self.feature(x)
        if return_moe_info:
            mask, moe_info = self.moe(conv_feature, snr_logits, return_info=True)
        else:
            mask = self.moe(conv_feature, snr_logits)
        x = x * mask
        if return_moe_info:
            return x, snr_logits, moe_info
        return x, snr_logits


class SNRExpert(nn.Module):
    """Full SAM-Mixer: SAM-Denoiser followed by ChirpMixer."""

    def __init__(self, config_dict):
        super(SNRExpert, self).__init__()
        config_dict['conv_dim_lstm'] = config_dict['num_samples']
        config_dict['freq_size'] = config_dict['num_classes']

        self.denoiser = SNRNet(config_dict)
        self.classifier = ChirpMixer(config_dict)

    def forward(self, x, return_moe_info=False):
        """
        Args:
            x: STFT image of shape [B, 2, F, T] (real/imag channels)
        Returns:
            mask_Y:   denoised STFT [B, 2, F, T]
            outputs:  classification logits over 2^sf classes
            snr_est:  SNR-estimator logits
            (moe_info if return_moe_info=True)
        """
        if return_moe_info:
            mask_Y, snr_est, moe_info = self.denoiser(x, return_moe_info=True)
            outputs = self.classifier(mask_Y)
            return mask_Y, outputs, snr_est, moe_info

        mask_Y, snr_est = self.denoiser(x)
        outputs = self.classifier(mask_Y)
        return mask_Y, outputs, snr_est


class MultiLoss(nn.Module):
    """Kendall-style uncertainty-weighted multi-task loss combination."""

    def __init__(self, num_losses, config):
        super().__init__()
        self.log_vars = nn.Parameter(
            torch.tensor([config['MultiLoss_alph1'], 0.0, 0.0])
        )

    def forward(self, losses):
        assert len(losses) == len(self.log_vars)
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + self.log_vars[i]
            total_loss += 0.5 * weighted_loss
        return total_loss

    def get_weights(self):
        return torch.exp(-self.log_vars).detach().cpu().numpy()


def count_params_torchinfo(model, input_size, device="cuda"):
    """Optional helper: count total/trainable parameters via torchinfo."""
    from torchinfo import summary

    model = model.to(device)
    model.eval()
    info = summary(model, input_size=(1, *input_size),
                   device=device, verbose=0)
    total_params = info.total_params
    trainable_params = info.trainable_params
    print(f"[torchinfo] Total params: {total_params / 1e6:.2f} M")
    print(f"[torchinfo] Trainable params: {trainable_params / 1e6:.2f} M")
    return total_params, trainable_params
