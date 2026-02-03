# Reproducing Results from "A Fourier Explanation of AI-Music Artifacts"

This guide explains how to reproduce **Table 1** (open-source models on FMA) and **Table 2** (closed-source Suno/Udio on SONICS) from the ISMIR 2025 paper.

## Overview

1. **Table 1**: Detect real vs synthetic on FMA medium, using the same split/data as [deepfake-detector](https://github.com/deezer/deepfake-detector). Synthetic = DAC (14kbps), Encodec (24kbps), Musika!.
2. **Table 2**: Detect real vs synthetic using FMA as real reference and SONICS as synthetic (Suno v3.5, Udio 130, etc.), 16 kHz, bandwidth [1 kHz, 8 kHz].

Pipeline: **compute artifact fingerprints (fakeprints)** → **train logistic regression** → **evaluate with provided splits**.

---

## Prerequisites

- **Python** with: `numpy`, `scikit-learn`, `torch`, `torchaudio`, `soxr`, `scipy`, `tqdm`
- **FMA medium** dataset ([github.com/mdeff/fma](https://github.com/mdeff/fma))
- **deepfake-detector** repo for FMA splits and synthetic data generation
- **SONICS** dataset for Table 2 ([Hugging Face](https://huggingface.co/datasets/awsaf49/sonics) or [Kaggle](https://www.kaggle.com/datasets/awsaf49/sonics-dataset))

---

## Part A: Table 1 (Open-source models on FMA)

Uses the same setup as [deepfake-detector](https://github.com/deezer/deepfake-detector): FMA medium split and auto-encoded tracks.

### Step A1: Get FMA data and split

1. Download **FMA medium** and place it (e.g. `fma_medium/` with `**/*.mp3`).
2. Get the **split file** from deepfake-detector:
   - [dataset_medium_split.npy](https://github.com/deezer/deezer/deepfake-detector/blob/main/data/dataset_medium_split.npy)  
   - It is a dict: `{'train': [...], 'validation': [...], 'test': [...]}` with keys = MP3 filenames (e.g. `"000002.mp3"`).

### Step A2: Create real and synthetic audio (same as deepfake-detector)

You need:

- **Real**: Resampled FMA (e.g. `fma_real_medium/resampled/*.mp3`) — same track IDs as FMA.
- **Synthetic**: Auto-encoded FMA with DAC, Encodec, Musika! (e.g. `fma_rebuilt_medium/{lac,encodec,musika}/.../*.mp3`).

Follow deepfake-detector’s README:

- Clone [deepfake-detector](https://github.com/deezer/deepfake-detector), set `HOME` in `loader/global_variables.py`.
- Clone pretrained repos (Musika!, LAC/VampNet) into a `pretrained` folder.
- Run their dataset creation scripts so you get:
  - `fma_real_medium/resampled/` (identity/resampled)
  - `fma_rebuilt_medium/lac/` (DAC, use 14 kbps for Table 1)
  - `fma_rebuilt_medium/encodec/` (Encodec 24 kbps)
  - `fma_rebuilt_medium/musika/` (Musika!)

File names should match the split (e.g. `000002.mp3` in each folder).

### Step A3: Compute fakeprints (artifact fingerprints)

Paper: average spectrum in [5 kHz, 16 kHz], then subtract lower hull and normalise (see `compute_fakeprints.py`). Use **44.1 kHz** and default `--fmin 5000 --fmax 16000` for FMA.

Run once for **real** and once per **synthetic** codec:

```bash
# Real (resampled FMA)
python compute_fakeprints.py --path /path/to/fma_real_medium/resampled --save fp_fma_real.npy --sr 44100 --fmin 5000 --fmax 16000

# DAC 14kbps (lac)
python compute_fakeprints.py --path /path/to/fma_rebuilt_medium/lac --save fp_fma_dac.npy --sr 44100 --fmin 5000 --fmax 16000

# Encodec 24kbps
python compute_fakeprints.py --path /path/to/fma_rebuilt_medium/encodec --save fp_fma_encodec.npy --sr 44100 --fmin 5000 --fmax 16000

# Musika!
python compute_fakeprints.py --path /path/to/fma_rebuilt_medium/musika --save fp_fma_musika.npy --sr 44100 --fmin 5000 --fmax 16000
```

Each `*.npy` is a dict `{ "000002.mp3": fingerprint_array, ... }`. Keys must match the split filenames.

### Step A4: Train and evaluate (Table 1)

Use the **same split** as deepfake-detector (e.g. `dataset_medium_split.npy`) so train/test are comparable to the paper. Run:

```bash
python train_test_regressor.py \
  --real fp_fma_real.npy \
  --synth fp_fma_dac.npy fp_fma_encodec.npy fp_fma_musika.npy \
  --split dataset_medium_split.npy \
  --report table1
```

This will:

- Restrict all real/synth dicts to the track IDs present in the split.
- Use `train` for fitting and `test` for evaluation.
- Train one binary classifier on real vs all synthetics (or real vs pooled synthetics).
- Report **Real** test accuracy and **Synthetic** test accuracy **per codec** (DAC, Encodec, Musika!) to match Table 1.

Target (from paper): Real ~99.87%, DAC ~99.68%, Encodec ~99.81%, Musika! ~99.97%.

---

## Part B: Table 2 (Closed-source: Suno / Udio on SONICS)

Real = FMA (resampled to 16 kHz); Synthetic = SONICS. Bandwidth [1 kHz, 8 kHz] (8 kHz = Nyquist at 16 kHz).

### Step B1: Get SONICS and splits

1. Download **SONICS** (synthetic tracks from Suno/Udio).
2. Use the **splits** from this repo: `sonics/sonics_split.npy`.  
   It contains keys such as `train`, `test`, `valid`, and per-source: `suno_v3.5`, `suno_v3`, `suno_v2`, `udio_v120`, `udio_v30` (values = lists of filenames, e.g. `fake_00001_suno_0.mp3`).

If you generate `sonics_split.npy` yourself, use `sonics/create_splits.py` (requires `sonics/fake_songs.csv` from SONICS metadata).

### Step B2: Real reference (FMA at 16 kHz)

Resample FMA medium to **16 kHz** and place in a folder (e.g. `fma_16k/`). You can use the same track set as in deepfake-detector; for Table 2 the paper uses FMA as real and does not require the exact same split as SONICS train/test — only that real and synthetic use consistent train/test splits.

### Step B3: Compute fakeprints

- **Real**: FMA at 16 kHz, bandwidth [1 kHz, 8 kHz]:

```bash
python compute_fakeprints.py --path /path/to/fma_16k --save fp_sonics_real.npy --sr 16000 --fmin 1000 --fmax 8000
```

- **Synthetic**: SONICS folder (all fake tracks, or per-source subfolders). Same SR and band:

```bash
python compute_fakeprints.py --path /path/to/sonics/fake_songs --save fp_sonics_synth.npy --sr 16000 --fmin 1000 --fmax 8000
```

If your SONICS layout uses one dir with all `*.mp3`, one run is enough. Otherwise run per subfolder and merge dicts so keys match `sonics_split.npy` (e.g. `fake_00001_suno_0.mp3`).

### Step B4: Train and evaluate (Table 2)

Use SONICS split for train/test (and optionally valid):

```bash
python train_test_regressor.py \
  --real fp_sonics_real.npy \
  --synth fp_sonics_synth.npy \
  --split sonics/sonics_split.npy \
  --report table2
```

Script should:

- Use `train` / `test` from `sonics_split.npy`.
- Map synthetic filenames to the same split (train/test).
- Report **Real** test accuracy and **Synthetic** test accuracy **per source** (Suno v3.5, Suno v3, Suno v2, Udio 130, Udio 32) to match Table 2.

Paper notes: performance drops on **Udio 32** (unseen); Table 2 reports ~39.83% for Udio 32†.

---

## Optional: Architecture dependence (Section 4.2, Figure 4)

To reproduce the finding that **artifacts are architecture-dependent** (same peak placement across seeds/datasets):

1. Train **DAC** (LAC) with different seeds and datasets (e.g. FMA twice with different seeds, MTAT, MTG-Jamendo) using deepfake-detector or VampNet.
2. Auto-encode a test set with each model.
3. Compute **fakeprints** for each model’s outputs.
4. Average fakeprints per model and plot; peak positions should align across models (Figure 4).

No change to the provided scripts is strictly required; you only need the extra trained DAC models and their decoded audio.

---

## Summary checklist

| Step | Table 1 (FMA) | Table 2 (SONICS) |
|------|----------------|-------------------|
| Data | FMA medium + deepfake-detector real/synthetic | FMA 16k + SONICS |
| Split | `dataset_medium_split.npy` | `sonics/sonics_split.npy` |
| SR | 44.1 kHz | 16 kHz |
| Bandwidth | [5 kHz, 16 kHz] | [1 kHz, 8 kHz] |
| Fakeprints | Real + DAC + Encodec + Musika | Real + SONICS synth |
| Train/eval | Same split, per-codec report | Same split, per-source report |

If you use the same data and splits as above, the logistic regression in `train_test_regressor.py` (with the updates below) should yield numbers close to Table 1 and Table 2 in the paper.
