# MeowCat Examples

Seven examples, each with a `config.yaml` and `run.sh`.

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
| [01_visium_only](#01--visium-only) | 1 Visium | Recon → Visium MSE | off |
| [02_xenium_only](#02--xenium-only) | 1 Xenium | Recon → Xenium CE | off |
| [03_visium_xenium_single](#03--visium--xenium-single-pair) | 1 Visium + 1 Xenium | Recon → Visium → Xenium | off |
| [04_multi_visium](#04--multiple-visium) | Multiple Visium | Recon → Visium MSE + CDAN | on |
| [04_multi_xenium](#04--multiple-xenium) | Multiple Xenium | Recon → Xenium CE + CDAN | on |
| [05_multi_visium_xenium](#05--multiple-visium--xenium) | Multiple Visium + Xenium | Recon → Visium+CDAN → Xenium | on |
| [06_predict_new_sample](#06--predict-new-sample) | New H&E images | Inference using trained model | — |

---

## 01 — Visium only

**Path:** `examples/01_visium_only/`

**Use case:** You have one Visium slide with RCTD-deconvolved cell-type proportions and want to train a soft-label predictor.

**User-provided input:**
```
/project/KidneyHE/01_meowcat_test/01_visium_only/input/
  VIS_S1/
    he_raw.tif                     <- raw H&E image
    filtered_feature_bc_matrix/    <- 10x Space Ranger output
    spatial/                       <- scalefactors_json.json + tissue_positions
```

**Pipeline steps (from scratch):**
```bash
meowcat rctd                    --config config.yaml   # Step 1: RCTD deconvolution
meowcat check-resolution        --config config.yaml   # Step 2: audit image MPP
meowcat preprocess               --config config.yaml   # Step 3: UNI feature extraction
meowcat prepare-visium           --config config.yaml   # Step 3.5: Visium metadata
meowcat prepare-visium-batches   --config config.yaml   # Step 4: build batch .npy files
meowcat train                    --config config.yaml   # Step 5: train model
meowcat predict                  --config config.yaml   # Step 6: full-grid inference
meowcat visualize                --config config.yaml   # Step 6: argmax + intensity maps
meowcat slide                    --config config.yaml   # Step 7: assemble PowerPoint
# Or run all at once:
meowcat run-all --config config.yaml
```

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

**User-provided input:**
```
/project/KidneyHE/01_meowcat_test/wrapped_data/02_xenium_only/input/
  XEN_P11_LUAD/
    he_raw.tif                       <- raw H&E image
    xenium_raw/                      <- 10x Xenium output (cell_feature_matrix.h5, cells.parquet)
    adata_cellbin_HistoSweep.h5ad    <- cell-bin AnnData (see main README for how to prepare)
    annotation.csv                   <- cell-type annotations (columns: cell_id, cell_state)
  anno-names.txt                     <- shared cell-type list (project-level, referenced by xenium.anno_names_path)
```

> `cell_type_mapping_json` (mapping fine → coarse cell types) is also required — set via `xenium.cell_type_mapping_json` in the config.

**Pipeline steps (from scratch):**
```bash
meowcat preprocess               --config config.yaml   # Step 3: UNI feature extraction → single_super_emb.h5ad
meowcat prepare-xenium-batches   --config config.yaml   # Step 4: merge histology features, build batch .npy files
meowcat train                    --config config.yaml   # Step 5: train model
meowcat predict                  --config config.yaml   # Step 6: full-grid inference
meowcat visualize                --config config.yaml   # Step 6: argmax + intensity maps
meowcat slide                    --config config.yaml   # Step 7: assemble PowerPoint
# Or run all at once:
meowcat run-all --config config.yaml
```

> Note: No `meowcat rctd` or `meowcat prepare-visium` steps — those are Visium-only. The `prepare-xenium-batches` step automatically merges histology features from `single_super_emb.h5ad` into `adata_cellbin_HistoSweep.h5ad`.

**Output:**
```
/project/KidneyHE/01_meowcat_test/wrapped_data/02_xenium_only/output/
  batches/
    batch_xen_000_x/y/d.npy      <- one-hot hard labels in _y.npy
    states/00/model.ckpt
    states/01/model.ckpt
  XEN_P11_LUAD/
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
- If `keep_frac` is set (e.g. `0.5`): each sample selects `ceil(keep_frac * n_valid_cells)` cells via stratified sampling.
- If `keep_frac` is **omitted** (default `None`): each sample selects the same number of cells as the first Visium batch file (`batch_vis_000_x.npy`'s row count). If no Visium batches exist, all valid cells are used. This keeps Xenium domain sizes balanced with Visium domains.

**Key config settings:**
```yaml
xenium:
  sample_pattern: "XEN*"
  anno_names_path: .../anno-names.txt
  cell_type_mapping_json: .../cell_type_mapping_lung.json
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

**User-provided input:**
```
/project/KidneyHE/01_meowcat_test/03_visium_xenium_single/input/
  VIS_P11_LUAD/
    he_raw.tif
    filtered_feature_bc_matrix/
    spatial/                           <- scalefactors_json.json + tissue_positions
  XEN_P11_LUAD/
    he_raw.tif
    xenium_raw/                        <- cell_feature_matrix.h5, cells.parquet
    adata_cellbin_HistoSweep.h5ad
    annotation.csv                     <- cell-type annotations (cell_id, cell_state)
```

> Both samples must share the **same cell-type vocabulary**. For Visium, cell-type names are derived from RCTD `major_prop.csv` columns (generated by `meowcat rctd`); for Xenium, from `anno-names.txt`. The config points `xenium.anno_names_path` to the Visium sample's generated `anno-names.txt` to ensure consistency.

**Pipeline steps (from scratch):**
```bash
meowcat rctd                    --config config.yaml   # Step 1: RCTD deconvolution (Visium only)
meowcat preprocess               --config config.yaml   # Step 3: UNI features for ALL samples (Visium + Xenium)
meowcat prepare-visium           --config config.yaml   # Step 3.5: Visium metadata (generates anno-names.txt)
meowcat prepare-visium-batches   --config config.yaml   # Step 4a: Visium batch files
meowcat prepare-xenium-batches   --config config.yaml   # Step 4b: Xenium batch files (merges histology features)
meowcat train                    --config config.yaml   # Step 5: 3-phase training
meowcat predict                  --config config.yaml   # Step 6: full-grid inference
meowcat visualize                --config config.yaml   # Step 6: argmax + intensity maps
meowcat slide                    --config config.yaml   # Step 7: assemble PowerPoint
# Or run all at once:
meowcat run-all --config config.yaml
```

> Important: `prepare-visium` must run before `prepare-xenium-batches` because the Xenium config references `anno-names.txt` generated by the Visium pipeline.

**Output:**
```
/project/KidneyHE/01_meowcat_test/03_visium_xenium_single/output/
  batches/
    batch_vis_000_x/y/d.npy      <- Visium (resolution=0, MSE)
    batch_xen_000_x/y/d.npy      <- Xenium (resolution=1, CE)
    states/00/model.ckpt
    states/01/model.ckpt
  VIS_P11_LUAD/  XEN_P11_LUAD/
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
  anno_names_path: .../VIS_P11_LUAD/anno-names.txt   # shared vocab from Visium RCTD
  cell_type_mapping_json: .../cell_type_mapping_lung.json

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

**User-provided input:**
```
/project/KidneyHE/01_meowcat_test/04_multi_visium/input/
  P11_LUAD/   P17_LUAD/   P19_LUAD/   P24_LUAD/   ...
    he_raw.tif
    filtered_feature_bc_matrix/
    spatial/
```

**Pipeline steps (from scratch):**
```bash
meowcat rctd                    --config config.yaml   # Step 1: RCTD per sample
meowcat preprocess               --config config.yaml   # Step 3: UNI features per sample
meowcat prepare-visium           --config config.yaml   # Step 3.5: Visium metadata per sample
meowcat prepare-visium-batches   --config config.yaml   # Step 4: batch files (one per sample)
meowcat train                    --config config.yaml   # Step 5: train with CDAN
meowcat predict                  --config config.yaml   # Step 6: predict all samples
meowcat visualize                --config config.yaml   # Step 6: argmax + intensity maps
meowcat slide                    --config config.yaml   # Step 7: assemble PowerPoint
# Or run all at once:
meowcat run-all --config config.yaml
```

**Key config settings:**
```yaml
visium:
  sample_pattern: "P*"
  keep_frac: 0.25        # subsample 25% per patient

train:
  two_stage: true
  epochs1: 15
  sequential_training: false
  visium_epochs: 100
  xenium_epochs: 0
  adv_lambda: 0.005      # CDAN cross-patient alignment
```

---

## 04 — Multiple Xenium

**Path:** `examples/04_multi_xenium/`

**Use case:** Multiple Xenium slides from different patients. CDAN domain adaptation aligns cross-patient representations. Xenium-only training (CE loss).

**User-provided input:**
```
/project/KidneyHE/01_meowcat_test/04_multi_xenium/input/
  P11_LUAD_Xenium/   P17_LUAD_Xenium/   P19_LUAD_Xenium/   P24_LUAD_Xenium/
    he_raw.tif
    xenium_raw/                        <- cell_feature_matrix.h5, cells.parquet
    adata_cellbin_HistoSweep.h5ad
    annotation.csv
  anno-names.txt                       <- project-level shared cell-type list
```

**Pipeline steps (from scratch):**
```bash
meowcat preprocess               --config config.yaml   # Step 3: UNI features per sample
meowcat prepare-xenium-batches   --config config.yaml   # Step 4: merge histology, build batches
meowcat train                    --config config.yaml   # Step 5: train with CDAN
meowcat predict                  --config config.yaml   # Step 6: predict all samples
meowcat visualize                --config config.yaml   # Step 6: argmax + intensity maps
meowcat slide                    --config config.yaml   # Step 7: assemble PowerPoint
# Or run all at once:
meowcat run-all --config config.yaml
```

**Key config settings:**
```yaml
xenium:
  sample_pattern: "P*"
  anno_names_path: .../anno-names.txt
  cell_type_mapping_json: .../cell_type_mapping_lung.json
  keep_frac: 0.03346869815   # balanced with Visium domain size: 59389/(443616*4)

train:
  two_stage: true
  epochs1: 15
  sequential_training: false
  visium_epochs: 0
  xenium_epochs: 100
  xenium_weight: 1.0
  adv_lambda: 0.005      # CDAN cross-patient alignment
```

---

## 05 — Multiple Visium + Xenium

**Path:** `examples/05_multi_visium_xenium/`

**Use case:** Multiple patients, both modalities. CDAN domain adaptation aligns cross-patient representations. 3-phase sequential training: reconstruction → Visium MSE → Xenium CE.

**User-provided input:**
```
/project/KidneyHE/01_meowcat_test/05_multi_visium_xenium/input/
  P11_LUAD/          <- Visium patient 1     (matched by visium.sample_pattern: "P*_LUAD")
  P17_LUAD/          <- Visium patient 2
  P19_LUAD/          <- Visium patient 3
  P24_LUAD/          <- Visium patient 4
    he_raw.tif
    filtered_feature_bc_matrix/
    spatial/
  P11_LUAD_Xenium/   <- Xenium patient 1    (matched by xenium.sample_pattern: "P*_LUAD_Xenium")
  P17_LUAD_Xenium/   <- Xenium patient 2
  P19_LUAD_Xenium/   <- Xenium patient 3
  P24_LUAD_Xenium/   <- Xenium patient 4
    he_raw.tif
    xenium_raw/
    adata_cellbin_HistoSweep.h5ad
    annotation.csv
  anno-names.txt                             <- project-level shared cell-type list
```

> Visium and Xenium samples are distinguished by `sample_pattern` globs: `"P*_LUAD"` matches Visium folders, `"P*_LUAD_Xenium"` matches Xenium folders. Both must share the same cell-type vocabulary.

**Pipeline steps (from scratch):**
```bash
meowcat rctd                    --config config.yaml   # Step 1: RCTD deconvolution (Visium samples)
meowcat preprocess               --config config.yaml   # Step 3: UNI features for ALL samples
meowcat prepare-visium           --config config.yaml   # Step 3.5: Visium metadata
meowcat prepare-visium-batches   --config config.yaml   # Step 4a: Visium batch files
meowcat prepare-xenium-batches   --config config.yaml   # Step 4b: Xenium batch files
meowcat train                    --config config.yaml   # Step 5: 3-phase sequential training
meowcat predict                  --config config.yaml   # Step 6: predict all samples
meowcat visualize                --config config.yaml   # Step 6: argmax + intensity maps
meowcat slide                    --config config.yaml   # Step 7: assemble PowerPoint
# Or run all at once:
meowcat run-all --config config.yaml
```

**Output:**
```
/project/KidneyHE/01_meowcat_test/05_multi_visium_xenium/output/
  batches/
    batch_vis_000_x/y/d.npy   <- Visium sample 1 (domain 0)
    batch_vis_001_x/y/d.npy   <- Visium sample 2 (domain 1)
    ...
    batch_xen_000_x/y/d.npy   <- Xenium sample 1 (domain N)
    batch_xen_001_x/y/d.npy   <- Xenium sample 2 (domain N+1)
    ...
    states/00/model.ckpt
    states/01/model.ckpt
  P11_LUAD/  P17_LUAD/  ...  P11_LUAD_Xenium/  ...
    embeddings-hist.pickle
    pred_fullgrid_outputs.pkl
  results_ex05.pptx
```

**Training phases:**

| Phase | Epochs | Loss | Notes |
|-------|--------|------|-------|
| 0 — Reconstruction | 15 | MSE on masked UNI features | All domains |
| 1 — Visium | 100 | MSE + CDAN adversarial | `adv_lambda=0.005` |
| 2 — Xenium | 100 | CE on hard labels | `xenium_weight=0.01` |

**Batch preparation:** Two separate steps — `meowcat prepare-visium-batches` for Visium, then `meowcat prepare-xenium-batches` for Xenium. Both write to the same `batches.out_dir` directory. Xenium domain IDs continue from where Visium left off (auto-detected from existing `batch_vis_*_d.npy` files in `batches.out_dir`).

**Domain ID assignment (auto):** Visium domains are assigned alphabetically (P11_LUAD=0, P17_LUAD=1, ...). Xenium domains continue from the max Visium domain + 1. To override Visium domain assignment, create a TSV:
```
P11_LUAD	cohort_A
P17_LUAD	cohort_A
P19_LUAD	cohort_B
P24_LUAD	cohort_B
```
and set `visium.domain_map_tsv` to its path. This groups samples into cohorts rather than individual domains.

**Key config settings:**
```yaml
visium:
  sample_pattern: "P*_LUAD"
  keep_frac: 0.25        # subsample 25% per patient

xenium:
  sample_pattern: "P*_LUAD_Xenium"
  anno_names_path: .../anno-names.txt
  cell_type_mapping_json: .../cell_type_mapping_lung.json

train:
  sequential_training: true
  visium_epochs: 100
  xenium_epochs: 100
  adv_lambda: 0.005      # CDAN cross-patient alignment
  xenium_weight: 0.01

visualize:
  save_highlights: true   # saves per-cluster highlight images
```

> **OOS monitoring (optional):** To hold out a sample for out-of-sample generalization monitoring, set `train.oos_sample` to the sample path and `train.oos_tmpdir` to a scratch directory. This config has OOS disabled (`null`) by default.

---

## 06 — Predict New Sample

**Path:** `examples/06_predict_new_sample/`

**Use case:** Run inference on new H&E images using a previously trained model. Uses `meowcat infer` to chain preprocessing, prediction, and visualization.

**User-provided input:**
```
/project/KidneyHE/01_meowcat_test/06_predict_new_sample/input/
  P_NEW_1/
    he_raw.tif       <- new H&E image (no transcriptomics data needed)
  P_NEW_2/
    he_raw.tif
```

**Pipeline steps:**
```bash
meowcat infer --config config.yaml
# This chains: preprocess -> predict -> visualize -> slide
```

**Key config settings:**
```yaml
inference:
  model_dir: .../05_multi_visium_xenium/output/batches   # path to states/ with model.ckpt files
  anno_names: .../anno-names.txt                          # cell-type list from training

visium:
  sample_pattern: "P*"     # discovers new sample folders

preprocess:
  pixel_size_raw: 0.5      # set explicitly if image metadata is unreliable
```

See [Inference on New H&E Images](../README.md#inference-on-new-he-images) in the main README for more details.
