#!/usr/bin/env python3
"""
Evaluate SAM-Mixer against the LoRaPHY baseline on NELoRa-bench-style data.

Example:
    python evaluate.py --config configs/sf7.yaml \
                       --weights weights/sf7.pth \
                       --data data/sf7

Expected data layout (NELoRa-bench style):
    data/sf7/<subfolder>/<truth_idx>_<...>.bin
where each .bin file holds `num_samples` complex64 IQ samples.
"""

import argparse
import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.fft import fft
from scipy.signal import chirp
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from snr_models import SNRExpert


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',   required=True, help='Path to model config yaml')
    p.add_argument('--weights',  required=True, help='Path to model state_dict (.pth)')
    p.add_argument('--data',     default=None, help='Directory of NELoRa-bench symbols (raw .bin files)')
    p.add_argument('--cache',    default=None, help='Pre-built pickle cache produced by load_data (overrides --data)')
    p.add_argument('--output',   default='./results', help='Output directory')
    p.add_argument('--snr-min',  type=float, default=-30.0)
    p.add_argument('--snr-max',  type=float, default=0.0)
    p.add_argument('--snr-step', type=float, default=2.0)
    p.add_argument('--device',   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed',     type=int, default=10)
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def make_downchirp(sf, bw, fs):
    num_classes = 2 ** sf
    num_samples = int(num_classes * fs / bw)
    t = np.linspace(0, num_samples / fs, num_samples + 1)[:-1]
    i = chirp(t, f0=bw / 2, f1=-bw / 2, t1=2 ** sf / bw, method='linear', phi=90)
    q = chirp(t, f0=bw / 2, f1=-bw / 2, t1=2 ** sf / bw, method='linear', phi=0)
    return i + 1j * q


def decode_loraphy(data_in, num_classes, downchirp, upsampling=100):
    """Conventional dechirp + FFT decoder. Returns predicted symbol index."""
    chirp_data = data_in * downchirp
    fft_raw = fft(chirp_data, len(chirp_data) * upsampling)
    target_nfft = num_classes * upsampling
    cut1 = np.asarray(fft_raw[:target_nfft])
    cut2 = np.asarray(fft_raw[-target_nfft:])
    return round(np.argmax(np.abs(cut1) + np.abs(cut2)) / upsampling) % num_classes


def perform_stft(data_in, num_samples, num_classes):
    """Compute STFT and reshape into the model input layout [B, 2, F, T]."""
    stft_full = torch.stft(input=data_in, n_fft=num_samples,
                           hop_length=num_classes // 4,
                           win_length=num_classes // 2,
                           pad_mode='constant',
                           return_complex=True)
    stft_img = torch.concat((stft_full[:, -num_classes // 2:, :],
                             stft_full[:, 0:num_classes // 2, :]), axis=1)
    return torch.stack((stft_img.real, stft_img.imag), 1)


def add_noise(dataY, snr_db, num_samples, normalize=True):
    """Add complex Gaussian noise at the target SNR (dB)."""
    amp = math.pow(0.1, snr_db / 20) * torch.mean(torch.abs(dataY))
    noise_r = amp / math.sqrt(2) * torch.randn(dataY.shape[0], num_samples)
    noise_i = amp / math.sqrt(2) * torch.randn(dataY.shape[0], num_samples)
    noise = (noise_r + 1j * noise_i).type(torch.cfloat)
    dataX = dataY + noise
    if normalize:
        dataX = dataX / torch.mean(torch.abs(dataX), dim=1, keepdim=True)
    return dataX


def load_data(data_dir, num_samples, num_classes, downchirp, cache_path=None):
    """Load NELoRa-bench symbols. Layout: <data_dir>/<subfolder>/<truth_idx>_*.bin"""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    files_by_class = [[] for _ in range(num_classes)]
    for subfolder in os.listdir(data_dir):
        sub_path = os.path.join(data_dir, subfolder)
        if not os.path.isdir(sub_path):
            continue
        for filename in os.listdir(sub_path):
            parts = filename.split('_')
            if len(parts) < 2:
                continue
            try:
                truth_idx = int(parts[1])
            except ValueError:
                continue
            if 0 <= truth_idx < num_classes:
                files_by_class[truth_idx].append(os.path.join(sub_path, filename))

    datax, datay = [], []
    for truth_idx, filelist in tqdm(enumerate(files_by_class),
                                    total=num_classes, desc='Loading symbols'):
        for filepath in filelist:
            with open(filepath, 'rb') as fid:
                chirp_raw = np.fromfile(fid, np.complex64, num_samples)
            if len(chirp_raw) != num_samples:
                continue
            # Integrity check: keep only symbols LoRaPHY decodes correctly at native SNR
            if decode_loraphy(chirp_raw, num_classes, downchirp) == truth_idx:
                datax.append(torch.tensor(chirp_raw, dtype=torch.cfloat))
                datay.append(truth_idx)

    if cache_path:
        with open(cache_path, 'wb') as f:
            pickle.dump((datax, datay), f)
    return datax, datay


def evaluate_at_snr(model, dataloader, snr_db, num_samples, num_classes,
                    downchirp, device):
    model.eval()
    sam_correct = 0
    lora_correct = 0
    total = 0
    pbar = tqdm(dataloader, desc=f'SNR={snr_db:>5.1f}dB', leave=False, ncols=80)
    with torch.no_grad():
        for dataY, truth in pbar:
            dataX = add_noise(dataY, snr_db, num_samples)
            input_img = perform_stft(dataX, num_samples, num_classes).to(device)
            _, logits, _ = model(input_img)
            pred = torch.argmax(logits, dim=1).cpu()
            sam_correct += (pred == truth).sum().item()

            dataX_np = dataX.numpy()
            truth_np = truth.numpy()
            for x_np, y in zip(dataX_np, truth_np):
                if decode_loraphy(x_np, num_classes, downchirp) == int(y):
                    lora_correct += 1
            total += truth.shape[0]
    return sam_correct / total, lora_correct / total


def snr_at_threshold(snrs, accs, threshold=0.9):
    """Linearly interpolate the SNR at which accuracy first crosses threshold."""
    snrs = np.asarray(snrs)
    accs = np.asarray(accs)
    above = accs >= threshold
    if not above.any():
        return None
    if above.all():
        return float(snrs[0])
    idx = int(np.argmax(above))
    if idx == 0:
        return float(snrs[0])
    x1, x2 = snrs[idx - 1], snrs[idx]
    y1, y2 = accs[idx - 1], accs[idx]
    if y2 == y1:
        return float(x2)
    return float(x1 + (threshold - y1) * (x2 - x1) / (y2 - y1))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_config(args.config)
    sf = cfg['sf']
    bw = cfg['bw']
    fs = cfg['fs']
    num_classes = 2 ** sf
    num_samples = int(num_classes * fs / bw)
    cfg['num_classes'] = num_classes
    cfg['num_samples'] = num_samples

    os.makedirs(args.output, exist_ok=True)

    print(f'[*] Building SAM-Mixer (SF={sf}) on {args.device}')
    model = SNRExpert(cfg)
    state = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(state, strict=True)
    model.to(args.device)
    model.eval()

    downchirp = make_downchirp(sf, bw, fs)

    if args.cache is not None:
        print(f'[*] Loading pre-built cache from {args.cache}')
        with open(args.cache, 'rb') as f:
            datax, datay = pickle.load(f)
    else:
        if args.data is None:
            raise SystemExit('Either --data (raw NELoRa-bench dir) or --cache (pre-built pkl) is required.')
        print(f'[*] Loading data from {args.data}')
        cache = os.path.join(args.output, f'cache_sf{sf}.pkl')
        datax, datay = load_data(args.data, num_samples, num_classes, downchirp,
                                 cache_path=cache)
    if len(datax) == 0:
        raise SystemExit(
            'No symbols loaded. If you intended to use the bundled cache, '
            'pass `--cache data/sf7_sample.pkl` instead of `--data`. '
            '`--data` expects the raw NELoRa-bench directory '
            '(<subfolder>/<truth_idx>_*.bin).'
        )
    print(f'[*] Loaded {len(datax)} symbols')

    dataset = TensorDataset(torch.stack(datax),
                            torch.tensor(datay, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    snrs = np.arange(args.snr_min,
                     args.snr_max + args.snr_step / 2,
                     args.snr_step).tolist()
    sam_accs, lora_accs = [], []

    print('\n   SNR (dB) | SAM-Mixer ACC | LoRaPHY ACC')
    print('   ---------+---------------+-------------')
    for snr_db in snrs:
        sam_acc, lora_acc = evaluate_at_snr(model, loader, snr_db,
                                            num_samples, num_classes,
                                            downchirp, args.device)
        sam_accs.append(sam_acc)
        lora_accs.append(lora_acc)
        print(f'   {snr_db:>7.1f}  |   {sam_acc:.4f}      |   {lora_acc:.4f}')

    sam_thr  = snr_at_threshold(snrs, sam_accs)
    lora_thr = snr_at_threshold(snrs, lora_accs)
    print('\n[*] SNR@10% (i.e. accuracy >= 90%)')
    print(f'    SAM-Mixer: {sam_thr}')
    print(f'    LoRaPHY:   {lora_thr}')
    if sam_thr is not None and lora_thr is not None:
        print(f'    Gain over LoRaPHY: {lora_thr - sam_thr:+.2f} dB')

    csv_path = os.path.join(args.output, f'results_sf{sf}.csv')
    with open(csv_path, 'w') as f:
        f.write('snr_db,sam_mixer_acc,sam_mixer_ser,loraphy_acc,loraphy_ser\n')
        for s, a_s, a_l in zip(snrs, sam_accs, lora_accs):
            f.write(f'{s},{a_s:.6f},{1 - a_s:.6f},{a_l:.6f},{1 - a_l:.6f}\n')
    print(f'[*] Wrote {csv_path}')

    plt.figure(figsize=(5, 4))
    plt.plot(snrs, [1 - a for a in sam_accs], 'o-',  label='SAM-Mixer', linewidth=2)
    plt.plot(snrs, [1 - a for a in lora_accs], 's--', label='LoRaPHY',   linewidth=2)
    plt.axhline(0.1, color='gray', linestyle=':', alpha=0.7, label='SER = 10%')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Symbol Error Rate (SER)')
    plt.title(f'SAM-Mixer vs LoRaPHY (SF={sf})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(args.output, f'results_sf{sf}.pdf')
    plt.savefig(plot_path)
    print(f'[*] Wrote {plot_path}')


if __name__ == '__main__':
    main()
