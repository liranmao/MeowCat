# MeowCat Examples

Seven self-contained test cases, each with a `config.yaml` and `run.sh`.
All data lives under `/project/KidneyHE/01_meowcat_test/`.

---

## Quick start

```bash
# install once from the repo root
pip install -e ..

# run any example (from its folder, or point to its config)
cd examples/01_visium_only
bash run.sh
```

Use `--dry-run` to print all commands without executing:

```bash
meowcat run-all --config examples/01_visium_only/config.yaml --dry-run
```

---

## Overview

| Example | Input samples | Training paradigm | CDAN |
|---------|--------------|-------------------|------|
| [01_visium_only](#01-visium-only) | 1 Visium | Recon → Visium MSE | off |
| [02_xenium_only](#02-xenium-only) | 1 Xenium | Recon → Xenium CE | off |
| [03_visium_xenium_single](#03-visium--xenium-single-pair) | 1 Visium + 1 Xenium | Recon → Visium → Xenium | off |
| [04_multi_visium](#04-multiple-visium) | Multiple Visium | Recon → Visium MSE + CDAN | on |
| [04_multi_xenium](#04-multiple-xenium) | Multiple Xenium | Recon → Xenium CE + CDAN | on |
| [05_multi_visium_xenium](#05-multiple-visium--xenium) | 2 Visium + 2 Xenium | Recon → Visium+CDAN → Xenium | on |
| [06_predict_new_sample](#06-predict-new-sample) | New H&E images | Inference using trained model | — |

---

## 01 — Visium only

**Path:** `examples/01_visium_only/`

**Use case:** You have one Visium slide with RCTD-deconvolved cell-type proportions and want to train a soft-label predictor.

**Input:**
```
/project/KidneyHE/01_meowcat_test/01_visium_only/input/
  VIS_S1/
    he_raw.tif
    filtered_feature_bc_matrix/
    spatial/                     <- scalefactors_json.json + tissue_positions
```

> `deconvolution_rctd/major_prop.csv`, `anno-names.txt`, `radius.txt`, etc. are **generated** by the pipeline (`meowcat rctd` and `meowcat preprocess`).

**Output:**
```
/project/KidneyHE/01_meowcat_test/01_visium_only/output/
  batches/
    batch_vis_000_x/y/d.npy      <- tokenized Visium spots
    states/00/model.ckpt
    states/01/model.ckpt
  VIS_S1/
    embeddings-hist.pickle
    pred_fullgrid_outputs.pkl
  results_ex01.pptx
```

**Training phases:**

| Phase | Epochs | Loss |
|-------|--------|------|
| 0 — Reconstruction | 15 | MSE on masked UNI features |
| 1 — Visium | 100 | MSE on RCTD soft proportions |

**Key config settings:**
```yaml
train:
  two_stage: true
  epochs1: 15
  sequential_training: false
  visium_epochs: 100
  xenium_epochs: 0
  adv_lambda: 0    # CDAN off (single patient)
```

---

## 02 — Xenium only

**Path:** `examples/02_xenium_only/`

**Use case:** You have one Xenium slide with manually annotated cell types and want to train a hard-label predictor.

**Input:**
```
/project/KidneyHE/01_meowcat_test/02_xenium_only/input/
  XEN_S1/
    he_raw.tif
    adata_cellbin_HistoSweep.h5ad   <- cell-bin AnnData
    XEN_S1_cell_type_anno.csv       <- per-cell annotations
    anno-names.txt
```

**Output:**
```
/project/KidneyHE/01_meowcat_test/02_xenium_only/output/
  batches/
    batch_xen_000_x/y/d.npy      <- one-hot hard labels in _y.npy
    states/00/model.ckpt
    states/01/model.ckpt
  XEN_S1/
    embeddings-hist.pickle
    pred_fullgrid_outputs.pkl
  results_ex02.pptx
```

**Training phases:**

| Phase | Epochs | Loss |
|-------|--------|------|
| 0 — Reconstruction | 15 | MSE on masked UNI features |
| 1 — Xenium | 100 | Cross-entropy on hard labels |

**Batch preparation:** Uses `meowcat prepare-xenium-batches` (not `prepare-visium-batches`, which is Visium-only).

**Xenium batch size logic (`xenium.keep_frac`):**
- If `keep_frac` is set (e.g. `0.25`): each sample selects `ceil(keep_frac * n_valid_cells)` cells via stratified sampling.
- If `keep_frac` is **omitted** (default `None`): each sample selects the same number of cells as the first Visium batch file (`batch_vis_000_x.npy`'s row count). If no Visium batches exist, all valid cells are used. This keeps Xenium domain sizes balanced with Visium domains.

**Key config settings:**
```yaml
xenium:
  sample_pattern: "XEN*"
  anno_names_path: .../XEN_S1/anno-names.txt
  keep_frac: 0.5

train:
  two_stage: true
  epochs1: 15
  sequential_training: false
  visium_epochs: 0
  xenium_epochs: 100
  xenium_weight: 1.0
```

---

## 03 — Visium + Xenium (single pair)

**Path:** `examples/03_visium_xenium_single/`

**Use case:** You have one Visium and one Xenium slide from a related tissue/disease. Train with 3-phase sequential learning: soft labels first, then fine-tune to single-cell resolution.

**Input:**
```
/project/KidneyHE/01_meowcat_test/03_visium_xenium_single/input/
  VIS_S1/
    he_raw.tif
    filtered_feature_bc_matrix/
    spatial/                         <- scalefactors_json.json + tissue_positions
  XEN_S1/
    he_raw.tif
    adata_cellbin_HistoSweep.h5ad
    XEN_S1_cell_type_anno.csv
    anno-names.txt                   <- cell types must match Visium RCTD columns
```

> Both samples must share the **same cell-type vocabulary**. For Visium, cell-type names are derived from RCTD `major_prop.csv` columns (generated by `meowcat rctd`); for Xenium, from `anno-names.txt`.

**Output:**
```
/project/KidneyHE/01_meowcat_test/03_visium_xenium_single/output/
  batches/
    batch_vis_000_x/y/d.npy      <- Visium (resolution=0, MSE)
    batch_xen_000_x/y/d.npy      <- Xenium (resolution=1, CE)
    states/00/model.ckpt
    states/01/model.ckpt
  VIS_S1/  XEN_S1/
    embeddings-hist.pickle
    pred_fullgrid_outputs.pkl
  results_ex03.pptx
```

**Training phases:**

| Phase | Epochs | Loss | Notes |
|-------|--------|------|-------|
| 0 — Reconstruction | 15 | MSE on masked UNI features | All data |
| 1 — Visium | 100 | MSE on RCTD soft proportions | Only `batch_vis_*` |
| 2 — Xenium | 50 | CE on hard labels | Only `batch_xen_*` |

**Batch preparation:** Two separate steps — `meowcat prepare-visium-batches` for Visium, then `meowcat prepare-xenium-batches` for Xenium. Both write to the same `batches.out_dir` directory. Xenium domain IDs continue from where Visium left off (auto-detected from existing `batch_vis_*_d.npy` files in `batches.out_dir`).

**Key config settings:**
```yaml
xenium:
  sample_pattern: "XEN*"
  anno_names_path: .../VIS_S1/anno-names.txt

train:
  two_stage: true
  epochs1: 15
  sequential_training: true
  visium_epochs: 100
  xenium_epochs: 50
  freeze_encoder_n: 2    # stabilizes fine-tuning at phase transitions
  xenium_weight: 0.01
  adv_lambda: 0
```

---

## 04 — Multiple Visium

**Path:** `examples/04_multi_visium/`

**Use case:** Multiple Visium slides from different patients. CDAN domain adaptation aligns cross-patient representations. Visium-only training (MSE loss).

**Key config settings:** Similar to `01_visium_only` but with multiple samples and `adv_lambda > 0` for CDAN.

---

## 04 — Multiple Xenium

**Path:** `examples/04_multi_xenium/`

**Use case:** Multiple Xenium slides from different patients. CDAN domain adaptation aligns cross-patient representations. Xenium-only training (CE loss).

**Key config settings:** Similar to `02_xenium_only` but with multiple samples and `adv_lambda > 0` for CDAN.

---

## 05 — Multiple Visium + Xenium

**Path:** `examples/05_multi_visium_xenium/`

**Use case:** Multiple patients, both modalities. CDAN domain adaptation aligns cross-patient representations. One sample is held out as an out-of-sample (OOS) generalization monitor.

**Input:**
```
/project/KidneyHE/01_meowcat_test/05_multi_visium_xenium/input/
  VIS_S1/   <- Visium patient 1 (training)
  VIS_S2/   <- Visium patient 2 (training + OOS monitor)
  XEN_S1/   <- Xenium patient 1
  XEN_S2/   <- Xenium patient 2
```

**Output:**
```
/project/KidneyHE/01_meowcat_test/05_multi_visium_xenium/output/
  batches/
    batch_vis_000_x/y/d.npy   <- VIS_S1 (domain 0)
    batch_vis_001_x/y/d.npy   <- VIS_S2 (domain 1)
    batch_xen_000_x/y/d.npy   <- XEN_S1 (domain 2)
    batch_xen_001_x/y/d.npy   <- XEN_S2 (domain 3)
    oos_batch/                 <- OOS batches for VIS_S2
    states/00/model.ckpt
    states/01/model.ckpt
  VIS_S1/  VIS_S2/  XEN_S1/  XEN_S2/
    embeddings-hist.pickle
    pred_fullgrid_outputs.pkl
  results_ex05.pptx
```

**Training phases:**

| Phase | Epochs | Loss | Notes |
|-------|--------|------|-------|
| 0 — Reconstruction | 15 | MSE on masked UNI features | All domains |
| 1 — Visium | 100 | MSE + CDAN adversarial | `adv_lambda=0.005`, OOS monitor active |
| 2 — Xenium | 100 | CE on hard labels | `xenium_weight=0.01` |

**Key config settings:**
```yaml
visium:
  keep_frac: 0.25        # subsample 25% per patient
  domain_map_tsv: null   # auto: one domain per WSI folder

train:
  sequential_training: true
  visium_epochs: 100
  xenium_epochs: 100
  adv_lambda: 0.005      # CDAN cross-patient alignment
  oos_sample: /project/KidneyHE/01_meowcat_test/05_multi_visium_xenium/input/VIS_S2
  oos_tmpdir: /project/KidneyHE/01_meowcat_test/05_multi_visium_xenium/output/oos_batch

visualize:
  save_highlights: true   # saves per-cluster highlight images
```

**Batch preparation:** Two separate steps — `meowcat prepare-visium-batches` for Visium (domains 0–1), then `meowcat prepare-xenium-batches` for Xenium (domains 2–3). Xenium domain IDs continue from where Visium left off (auto-detected from existing `batch_vis_*_d.npy` files in `batches.out_dir`).

**Domain ID assignment (auto):** Visium domains are assigned alphabetically (VIS_S1=0, VIS_S2=1). Xenium domains continue from the max Visium domain + 1 (XEN_S1=2, XEN_S2=3). To override Visium domain assignment, create a TSV:
```
VIS_S1	cohort_A
VIS_S2	cohort_A
XEN_S1	cohort_B
XEN_S2	cohort_B
```
and set `visium.domain_map_tsv` to its path. This groups the two cohorts rather than four individuals.

---

## 06 — Predict New Sample

**Path:** `examples/06_predict_new_sample/`

**Use case:** Run inference on new H&E images using a previously trained model. Uses `meowcat infer` to chain preprocessing, prediction, and visualization.

**Key config settings:** Uses the `inference` section to point to a trained model directory and `anno-names.txt`. See [Inference on New H&E Images](../README.md#inference-on-new-he-images) in the main README.
