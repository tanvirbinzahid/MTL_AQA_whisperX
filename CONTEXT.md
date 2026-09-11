# MTL-AQA Project — Master Log

**Owner:** tabz1225 · **Server:** ai.ee.unlv.edu · **Root:** `/home/tabz1225/projects/MTL-AQA-master`
**Last updated:** 2026-09-11

> This is the single source of truth for this project. **Every future run, eval, result, and
> file move gets logged here** (newest section at the bottom of the Timeline, or in the
> relevant table). Keep entries dense: date, what changed, numbers, paths.

---

## 1. What this project is

The classic **MTL-AQA** architecture (Parmar & Tran Morris, CVPR 2019) — *"What and How Well
You Performed?"* — a multitask model that predicts, from a diving video clip:

| Task | Head | Output |
|---|---|---|
| Final score regression | `model_score_regressor` | scalar (normalized by `final_score_std=17`) |
| Dive number classification | `model_dive_classifier` | 5 heads: position(3), armstand(2), rotation(4), #somersaults(10), #twists(8) |
| Comment generation | `model_caption` | S2VT GRU caption |

**Architecture:** C3D backbone → sliding 16-frame clips → fc6 (8192) → mean-pool → heads.
Input: 96-frame window, 112×112 crops from 171×128 resize.

**Two model families in this project:**
1. **Plain C3D** — backbone initialized from `models/c3d.pickle` (Sports-1M pretrained). = "no-SSL".
2. **MoCo SSL C3D** — backbone self-supervised pretrained on the MTL videos (`models/moco_ssl/`), then fine-tuned. = "SSL".

---

## 2. Directory structure (reorganized 2026-09-11)

```
MTL-AQA-master/
├── context.md                  ← THIS FILE (master log)
├── eval_ssl_any.py             ← non-interactive eval harness (mtl|fine|amateur, --ckpt DIR)
├── evaluate.py                 ← original interactive evaluator
├── SSL/checkpoints/            ← ⭐ MoCo self-supervised C3D run (best ρ 0.8936)
├── best_nonssl/checkpoints/    ← ⭐ best surviving no-SSL run = run123 (best ρ 0.8871)
├── models/                     ← backbones: c3d.pickle (no-SSL init), moco_ssl/ (SSL init)
├── MTL-AQA_code_release/       ← source code (train_test_C3DAVG.py, opts.py, dataloaders/, models/)
├── MTL-AQA_dataset_release/    ← annotations + frames (19G)
├── videos/                     ← source mp4s (16G)
├── extracted_dive_frames/      ← extracted dive frames (20M)
└── archive/
    ├── checkpoints/            ← run4_frame103, run5, run6, run7_invalidcaps, run7_cap_fixed
    ├── eval_results/           ← all eval_results_*.txt (8 files)
    ├── logs/                   ← historical training logs + opts.py.pre_reorg.bak
    └── scripts/                ← slowmo/ + experiments/ one-off scripts
```

**Checkpoint file convention (all runs):** `best_model_{CNN,my_fc6,score_regressor,dive_classifier,caption}.pth`,
`best_optimizer.pth`, `model_*_99.pth` (epoch-99 snapshot), `<YYMMDD_HHMMSS>_stats_{train,test}_*.txt`.

⚠️ **Every run writes the SAME fixed filenames into one `ckpt_dir`** → a later run silently
overwrites an earlier run's best weights. **Always copy the current best-model set into a
uniquely named folder BEFORE launching a new run.**

---

## 3. Run timeline (all MTL-AQA test = 353 dives, official split 0)

| # | Run / folder | Date | Backbone | Best test ρ | MSE | Class avg | Status |
|---|---|---|---|---|---|---|---|
| 1 | `archive/checkpoints/run4_frame103` | 2026-01-07 | c3d.pickle | 0.645 | — | — | early |
| 2 | `archive/checkpoints/run5` | 2026-01-08 | c3d.pickle | 0.623 | — | — | early |
| 3 | `archive/checkpoints/run6` | 2026-01-09 | c3d.pickle | 0.888 | — | — | ok |
| 4 | `archive/checkpoints/run7_invalidcaps` | 2026-01-09/10 | c3d.pickle | **0.89945** (raw) | 60.44 | — | ⚠️ captions invalid → no clean eval |
| 5 | **TBZ-Run-2** (dir lost) | 2026-03-16 | c3d.pickle | **0.8984** ✓eval | 61.31 | 96.71% | ❌ **weights destroyed** (overwritten) |
| 6 | `archive/checkpoints/run7_cap_fixed` | 2026-04-16 | c3d.pickle | 0.8848 | — | — | caption-fix retry |
| 7 | **`best_nonssl/checkpoints`** (run123 = "run 3.5 ep129") | 2026-04-18 | c3d.pickle | **0.8871** ✓eval | 67.58 | 95.64% | ✅ **best surviving no-SSL** |
| 8 | **`SSL/checkpoints`** (SSLTRAIN) | 2026-05-25/26 | MoCo C3D | **0.8936** ✓eval | 69.37 | 97.39% | ✅ best overall ρ |

### Milestones
| Date | Event |
|---|---|
| 2025-11-25 | `models/c3d.pickle` (Sports-1M C3D) obtained |
| 2026-01-07 → 01-10 | Runs 4/5/6/7 (plain C3D), iterating on frame extraction & captions |
| 2026-02 | Slow-motion frame-extraction experiments (`archive/scripts/slowmo/`) |
| 2026-03-16 | **TBZ-Run-2** ρ 0.8984 — record no-SSL score. Weights later destroyed. |
| 2026-04-16 | `cap_fixed_run7`; `opts.py` last edited for no-SSL config (points to `c3d.pickle`) |
| 2026-04-18 | **run123** ρ 0.8871 — best non-SSL run that still exists |
| 2026-05-25 17:00 | MoCo SSL **pretraining** on MTL videos → `models/moco_ssl/moco_c3d_backbone_only.pth` (18:28) |
| 2026-05-25 20:11 | `opts.py` switched `c3d_base` → MoCo backbone |
| 2026-05-25/26 | **SSL fine-tune** (SSLTRAIN) → ρ 0.8936 |
| 2026-07-12 22:42 | `/mnt/sdb` unmounted — old disk (incl. the overwritten run2 dir) gone |
| 2026-08-20 | Migration: `MTL-AQA-master` copied to `/home/tabz1225/` |
| 2026-08-24 | **3-dataset SSL eval** (MTL/FineDiving/Amateur) via `eval_ssl_any.py` |
| 2026-09-11 | **Folder reorganization** (SSL/, best_nonssl/, archive/) + `context.md` created |

---

## 4. Results

### 4.1 MTL-AQA test (353 dives) — native

| Model | ρ | MSE | MAE | Position | Armstand | Rotation | #Somers. | #Twists | Avg cls |
|---|---|---|---|---|---|---|---|---|---|
| Prof Morris (paper, ep94) | 0.8562 | 88.02 | 6.85 | 94.05 | 99.43 | 96.32 | 96.60 | 92.07 | 95.69 |
| TBZ-Run-2 *(lost)* | **0.8984** | **61.31** | **5.73** | 97.17 | 99.72 | 95.47 | 97.17 | 94.05 | 96.71 |
| **run123 / best_nonssl** | 0.8871 | 67.58 | 6.02 | 95.18 | 99.43 | 94.33 | 95.75 | 93.48 | 95.64 |
| run 3 | 0.8838 | 67.95 | 6.11 | 95.75 | 99.43 | 94.62 | 95.18 | 94.05 | 95.81 |
| **SSL / SSL/checkpoints** | **0.8936** | 69.37 | 5.97 | 97.17 | 99.72 | 96.03 | 97.73 | 96.32 | **97.39** |

### 4.2 SSL model — 3-dataset transfer (2026-08-24)

| Dataset | n | ρ | Pearson | RMSE | MAE | RL2 |
|---|---|---|---|---|---|---|
| MTL-AQA | 353 | 0.8936 | 0.8936 | 8.33 | 5.97 | 0.0064 |
| FineDiving | 749 | 0.7490 | 0.7645 | 10.07 | 7.13 | 0.0093 |
| AmateurDive | 575 | −0.2474 | −0.4111 | 35.11 | 33.06 | 0.2958 |

Classification (SSL): MTL 97.39% avg · FineDiving 63.18% (somersaults 29.8 / twists 21.5 collapse) · Amateur 55.10% (#somersaults 0.0%).

### 4.3 vs HP-MCoRe (separate project, `/home/tabz1225/projects/HP-MCoRe`)

| Dataset | HP-MCoRe ρ | SSL C3D ρ | HP-MCoRe RMSE | SSL RMSE |
|---|---|---|---|---|
| MTL-AQA | **0.9432** | 0.8936 | **5.36** | 8.33 |
| FineDiving | **0.9302** | 0.7490 | **5.67** | 10.07 |
| AmateurDive | **0.2228** | −0.2474 | **14.22** | 35.11 |

**Takeaway:** HP-MCoRe (pose-guided contrastive) beats SSL-C3D on every dataset, especially
transfer. SSL-C3D's edge is native MTL-AQA *classification* (~97% vs HP-MCoRe's ~46% proxy).

---

## 5. How to run

```bash
cd /home/tabz1225/projects/MTL-AQA-master
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate mmpose   # torch 2.4 (loads new-style ckpts)

# eval any dataset with any checkpoint dir
python3 -u eval_ssl_any.py mtl     --ckpt SSL/checkpoints        --gpu 7
python3 -u eval_ssl_any.py fine    --ckpt SSL/checkpoints        --gpu 7
python3 -u eval_ssl_any.py amateur --ckpt SSL/checkpoints        --gpu 7
python3 -u eval_ssl_any.py mtl     --ckpt best_nonssl/checkpoints --gpu 7   # no-SSL
```
- Save `pred`/`true` arrays to `/shared/rtis_lab/data/AQA/MTL-AQA/prepared/ssl_<ds>_{pred,true}.npy`.
- Eval logs → `archive/logs/ssl_eval/`.
- **Score scale:** the regressor outputs normalized values; multiply by **17** to get real scores
  (`score_std=17`). MTL/Amateur saved arrays are already ×17; FineDiving preds are raw.

**Training (new run):** edit `MTL-AQA_code_release/opts.py` (`ckpt_dir`, `c3d_base`), then
`python3 train_test_C3DAVG.py` from `MTL-AQA_code_release/`. **Copy the current best set elsewhere first.**

**Environments:** `mmpose` (py3.10, torch 2.4.0+cu121) — use for eval/loading modern checkpoints.
`hp-mcore` (py3.7, torch 1.10.1) — older; `torch.load` there chokes on newer pickles.

---

## 6. Known issues / caveats

1. **TBZ-Run-2 (ρ 0.8984, best no-SSL) is unrecoverable.** It wrote to the shared default
   `checkpoints/` and was later overwritten (first by competing runs, finally by the SSL run on
   2026-05-26), then `/mnt/sdb` was unmounted 2026-07-12. Checked: `/shared` (only author's 2021
   models), LVM snapshots (none), ext4 `debugfs` (needs sudo), tarballs (none). **Only its eval
   `.txt` survives** (`archive/eval_results/eval_results_TBZ-RUN-2_best.txt`).
2. `run7_invalidcaps` reached the highest raw ρ (0.89945) but its captions were invalid → no clean eval.
3. Old `/mnt/sdb` paths in `opts.py` were fixed to `/home/...` on 2026-09-11 (backup:
   `archive/logs/opts.py.pre_reorg.bak`).
4. `MTL-AQA_dataset_release/frames` + `/shared/rtis_lab/data/AQA/Videos/whole_videos_frames` are the
   two frame stores — check which a script expects.
5. GPU: all 8 GPUs shared; pick an idle one (`nvidia-smi`). Evaluations use `--gpu N`.

---

## 7. Change log (append newest at bottom)

- **2026-09-11** — Reorganized folders: `SSL/`, `best_nonssl/`, `archive/{checkpoints,eval_results,logs,scripts}`.
  Fixed `opts.py` `/mnt/sdb` → `/home` paths. Repointed `eval_ssl_any.py` default ckpt to `SSL/checkpoints`.
  Verified 91 `.pth` before == after; both best sets load. Created this `context.md`.
