# SAM-Mixer: An SNR-aware Mixture-of-Experts Framework for Weak LoRa Signal Decoding

This repository contains the reference implementation and pre-trained checkpoint
for **SAM-Mixer**, accompanying our paper submission.

The framework combines:

- **SAM-Denoiser**: an SNR-aware Mixture-of-Experts denoising module guided by
  an auxiliary SNR estimator.
- **ChirpMixer**: a lightweight chirp classifier with a multi-scale feature
  extractor and an axis-wise Mixer head.

This release reproduces a subset of Fig. 7(a) in the paper — the
LoRaPHY-vs.-SAM-Mixer comparison at SF=7. The full Fig. 7(a) also includes
NELoRa as a learning-based baseline, which is not bundled here.

---

## Repository layout

```
.
├── evaluate.py            # SAM-Mixer vs. LoRaPHY evaluation script
├── snr_models.py          # Model definitions (SAM-Denoiser, ChirpMixer, etc.)
├── configs/
│   └── sf7.yaml           # Configuration for the SF=7 checkpoint
├── weights/
│   └── sf7.pt             # Trained weights, SF=7 (tracked via Git LFS)
├── data/
│   └── sf7_sample.pkl     # Cached NELoRa-bench symbols, SF=7 (Git LFS)
├── requirements.txt
└── LICENSE
```

## Requirements

- Python 3.9+
- PyTorch 2.0+ (CUDA optional)
- See `requirements.txt`

Install:

```bash
pip install -r requirements.txt
```

## Quick start

Make sure Git LFS is installed and the large files are pulled:

```bash
git lfs install
git lfs pull
```

Then run the SF=7 evaluation:

```bash
python evaluate.py \
    --config configs/sf7.yaml \
    --weights weights/sf7.pt \
    --cache   data/sf7_sample.pkl
```

The script will:

1. Load the SAM-Mixer architecture from the config and the pre-trained weights.
2. Iterate over a range of SNR values (default: -30 dB to 0 dB, step 2 dB).
3. For each SNR, evaluate both **SAM-Mixer** and the conventional
   **LoRaPHY** dechirp-and-FFT decoder on the same noisy symbols.
4. Print per-SNR accuracy and the SNR@10% threshold (i.e. the SNR at which
   the symbol error rate drops below 10%).
5. Write `results/results_sf7.csv` and `results/results_sf7.pdf`.

Common options:

| Flag           | Default | Description                                          |
|----------------|---------|------------------------------------------------------|
| `--snr-min`    | -30.0   | Minimum SNR in dB                                    |
| `--snr-max`    | 0.0     | Maximum SNR in dB                                    |
| `--snr-step`   | 2.0     | Step size in dB                                      |
| `--device`     | auto    | `cuda` or `cpu`                                      |
| `--seed`       | 10      | RNG seed for the additive Gaussian noise             |
| `--data`       | -       | Raw NELoRa-bench directory (alternative to `--cache`) |
| `--cache`      | -       | Pre-built `(datax, datay)` pickle (overrides `--data`) |
| `--output`     | results | Directory for CSV/PDF outputs                        |

## Data

`data/sf7_sample.pkl` is a pickled tuple `(datax, datay)`:

- `datax`: list of complex64 IQ tensors of length `num_samples = 2^sf * fs/bw`,
  each a clean LoRa symbol captured over the air at SF=7.
- `datay`: list of ground-truth symbol indices (0 ≤ idx < 2^sf).

Gaussian noise is added by `evaluate.py` at the requested SNR; the underlying
symbols are the same ones used in the original NELoRa benchmark
([source](https://github.com/daibiaoxuwu/NeLoRa_Dataset)).

To regenerate the cache from raw NELoRa-bench `.bin` files, point `--data` at
the dataset root instead of `--cache`.

## Expected results

On the supplied SF=7 sample, you should see SAM-Mixer reach an SNR@10% near
**-18.9 dB** (vs. roughly **-15.4 dB** for LoRaPHY in our paper experiments),
i.e. a ≈ 3.5 dB gain over the conventional decoder.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
