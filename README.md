# 🤿 MTL-AQA — WhisperX Edition

**What and How Well You Performed? A Multitask Learning Approach to Action Quality Assessment**

[![arXiv](https://img.shields.io/badge/arXiv-1904.04346-b31b1b.svg)](https://arxiv.org/abs/1904.04346)
![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-orange.svg)
![License: Research](https://img.shields.io/badge/license-research--use-lightgrey.svg)

This is a maintained fork of [ParitoshParmar/MTL-AQA](https://github.com/ParitoshParmar/MTL-AQA) (CVPR 2019) that:

1. **Replaces the original broadcast-caption labels with WhisperX transcripts** — real, time-aligned commentary captions extracted from the source videos (`Raw_Annotations/whisper_srt/`), improving the captioning task's ground truth.
2. **Modernizes the training pipeline** — stable training on current PyTorch (AMP, gradient clipping, dict-based metrics, fixed frame mapping, `weights_only`-safe checkpointing).
3. **Ships a trained best model (no-self-supervised / plain C3D)** — directly runnable, with full evaluation results.

---

## 🧭 Why this fork?

The original MTL-AQA paper jointly learns **three tasks** from a diving clip:

| Task | Head | Output |
|---|---|---|
| 🏆 Final score regression | `score_regressor` | normalized scalar (× 17 → actual score) |
| 🤸 Dive number classification | `dive_classifier` | 5 attributes: position, armstand, rotation, #somersaults, #twists |
| 💬 Comment generation | `caption` (S2VT-GRU) | text description of the dive |

The upstream captions were noisy broadcast subtitles. **This fork re-generates them with WhisperX** — word-level timestamps from the original videos — then retrains the whole multitask model on the cleaner labels, and makes the resulting **best checkpoint** (plain-C3D variant) available out of the box.

---

## 📑 What's new (vs. upstream)

### 🔤 WhisperX caption pipeline
- `Raw_Annotations/whisper_srt/*.srt` — WhisperX transcripts for **all 15 MTL-AQA videos** (`01`–`26`).
- `convert_whisperx_time_aligned_fixed.py` — converts SRT → `final_captions_dict.pkl` + `vocab.json` (with `<START>/<END>/<UNK>`), regenerating the model's caption ground truth.
- `update_captions.py` — merges SRT text into `final_annotations_dict` and rebuilds the caption vocabulary from the train split.
- `verify_captions.py` — sanity-checks the generated caption dict.

### ⚙️ Training modernization (`MTL-AQA_code_release/`)
| File | What changed |
|---|---|
| `train_test_C3DAVG.py` | torch.cuda.AMP + `GradScaler`; **gradient clipping**; **MSE logged with `final_score_std`**; dict-format per-epoch stats (rho/MSE/5 accuracies); `save_model` supports `best_model_*.pth` naming; `torch.load(..., weights_only=False)` for torch ≥ 2.0 compat; optimizer checkpoint save/load |
| `opts.py` | Correct paths, C3D-AVG config restored (`96` frames, `112×112`, `171×128` resize, `vocab 4768`, `caption dim 8192`), `ckpt_dir`/`c3d_base` documented (SSL vs plain) |
| `dataloaders/dataloader_C3DAVG.py` | **Fixed frame-mapping bug** (correct sequential frame indexing), deterministic seeds, cleaned comments |
| `convert_c3d.py` | Converts the official `c3d.pickle` → a PyTorch `.pth` backbone for training |

### 🔬 Self-supervised (MoCo) experiments (reference code)
`pretrain_ssl*.py` + `ssl_dataloader.py` implement the optional MoCo self-supervised C3D pretraining on MTL videos. The released best model below does **not** use it (the *no-SSL* variant), but the code is included for reproducibility.

---

## 🎯 Released best model — plain C3D (no self-supervision)

**Location:** [`best_nonssl/checkpoints/`](best_nonssl/checkpoints/)

The best surviving non-SSL MTL-AQA checkpoint (`run123`), trained 2026-04 on the WhisperX captions:

| Metric | Value |
|---|---|
| **Spearman ρ** (353 test dives) | **0.8871** |
| **Pearson r** | 0.8937 |
| **MSE** | 67.58 |
| **MAE** | 6.02 |
| classification avg | **95.64%** |
| position / armstand / rotation / #somersaults / #twists | 95.18 / 99.43 / 94.33 / 95.75 / 93.48% |

> Curious about SSL? The MoCo-pretrained run reaches **ρ 0.8936** (classification 97.39%). On this
> dataset self-supervision buys only ~0.006 ρ — the plain-C3D model below performs nearly
> identically and needs no pretraining step.

### Files
```
best_nonssl/checkpoints/
├── best_model_CNN.pth                 # C3D backbone (Git LFS)
├── best_model_my_fc6.pth              # fc6 head (Git LFS)
├── best_model_caption.pth             # S2VT caption head (Git LFS)
├── best_model_dive_classifier.pth     # 5-way attribute classifier
├── best_model_score_regressor.pth     # score regressor
├── train_log_run3.txt                 # training log
└── *stats_{train,test}*.txt           # per-epoch metrics
```

> ⚠️ Large files use **Git LFS** (GitHub's 100 MB commit limit). Clone with `git lfs install && git lfs pull`, or download the individual files from the web UI.

---

## 🚀 Quick start

### 0. Setup
```bash
# Linux/CUDA box, Python 3.10, PyTorch ≥ 2.0
conda env create -f MTL-AQA_code_release/environment.yml   # or: pip install torch torchvision scipy pillow
git lfs install && git lfs pull                              # fetch the model weights
```

### 1. Get the data
Download the MTL-AQA frames (per the [upstream README](https://github.com/ParitoshParmar/MTL-AQA)):
```bash
# place frames at: MTL-AQA_dataset_release/frames/<01|02|...>/<frame>.jpg
bash MTL-AQA_dataset_release/frame_extractor.sh  # or extract with your own tooling
```

### 2. Evaluate the released model
```bash
# non-interactive harness (mtl | fine | amateur datasets)
python3 -u eval_ssl_any.py mtl --ckpt best_nonssl/checkpoints --gpu 0
```
The harness loads the 5 checkpoint parts, runs the C3D sliding-window forward pass, and prints
ρ, RMSE, MAE, RL2 plus the 5 attribute accuracies. (MTL-AQA dataset keys/frames must be
reachable — see `MTL-AQA_dataset_release/Ready_2_Use/` and adjust paths in `eval_ssl_any.py`
to your frame root.)

### 3. (Re)train from scratch
```bash
cd MTL-AQA_code_release
# the Sports-1M C3D backbone (c3d.pickle) is the standard MTL-AQA init
python convert_c3d.py --pickle ../c3d.pickle --out ../c3d_pt.pth
# edit opts.py: ckpt_dir, c3d_base, epochs, …
python train_test_C3DAVG.py
```

### 4. Regenerate WhisperX captions
```bash
cd MTL-AQA_code_release
python convert_whisperx_time_aligned_fixed.py         # SRT → final_captions_dict.pkl + vocab.json
python update_captions.py                             # merge SRT text + rebuild vocab
python verify_captions.py                             # sanity check
```

---

## 📊 Full results (353-dive MTL-AQA test split)

| Model | ρ | MSE | MAE | Class avg |
|---|---|---|---|---|
| Prof Morris (paper, epoch 94) | 0.8562 | 88.02 | 6.85 | 95.69% |
| TBZ-Run-2 *(weights lost)* | 0.8984 | 61.31 | 5.73 | 96.71% |
| **this fork — run123 (no-SSL)** | **0.8871** | **67.58** | **6.02** | **95.64%** |
| run 3 (no-SSL) | 0.8838 | 67.95 | 6.11 | 95.81% |
| SSL (MoCo pretrained) | 0.8936 | 69.37 | 5.97 | 97.39% |

Cross-dataset transfer of the SSL variant: FineDiving ρ 0.7490 · AmateurDive ρ −0.2474 — the model trained on elite MTL-AQA dives transfers to professional FineDiving but not to amateur footage (expected domain gap).

---

## 📂 Repository layout

```
MTL_AQA_whisperX/
├── README.md                     ← this readme
├── eval_ssl_any.py                ← non-interactive 3-dataset eval harness
├── best_nonssl/checkpoints/       ← ⭐ best no-SSL model (ρ 0.8871)
├── MTL-AQA_code_release/          ← training/eval/caption code
└── MTL-AQA_dataset_release/       ← annotations + WhisperX captions + frames
    └── Raw_Annotations/whisper_srt/  ← WhisperX SRT transcripts (all 15 videos)
```

---

## 📜 Citation

```bibtex
@inproceedings{mtlaqa,
  title={What and How Well You Performed? A Multitask Learning Approach to Action Quality Assessment},
  author={Parmar, Paritosh and Tran Morris, Brendan},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={304--313},
  year={2019}
}
```

## 🙏 Acknowledgements
- [ParitoshParmar/MTL-AQA](https://github.com/ParitoshParmar/MTL-AQA) — original dataset, code, and paper
- [WhisperX](https://github.com/m-bain/whisperX) — word-level timestamped transcripts used to regenerate caption labels