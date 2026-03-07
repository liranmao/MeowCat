<p align="center">
  <img src="image/logo_v2.png" alt="MeowCat Logo" width="400"/>
</p>

A deep learning framework for cell-type annotation in histopathology H&E images, using spatially-registered transcriptomics data as training supervision.

---

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Repository Layout](#repository-layout)
4. [Data Requirements](#data-requirements)
5. [Pipeline Walkthrough](#pipeline-walkthrough)
   - [Step 1 — RCTD Deconvolution](#step-1--rctd-deconvolution)
   - [Step 2 — Resolution Check](#step-2--resolution-check)
   - [Step 3 — Image Preprocessing (training samples)](#step-3--image-preprocessing-training-samples)
   - [Step 3.5 — Visium Metadata Preparation](#step-35--visium-metadata-preparation)
   - [Step 4a — Visium Batch Preparation](#step-4a--visium-batch-preparation)
   - [Step 4b — Xenium Batch Preparation](#step-4b--xenium-batch-preparation)
   - [Step 5 — Training](#step-5--training)
   - [Step 6 — Prediction & Visualization](#step-6--prediction--visualization)
   - [Step 7 — Slide Wrap](#step-7--slide-wrap)
6. [Inference on New H&E Images](#inference-on-new-he-images)
7. [Config Reference](#config-reference)
8. [Training Paradigms](#training-paradigms)
9. [Model Architecture](#model-architecture)
10. [Output Files](#output-files)
11. [Examples](#examples)

---

## Overview

MeowCat predicts cell-type distributions across an entire H&E whole-slide image (WSI) at pixel resolution, trained on spatially-registered transcriptomics data. It supports two types of supervision:

- **Spot data (Visium)**: soft cell-type proportions from RCTD deconvolution → **MSE loss**
- **Single-cell data (Xenium / manually annotated)**: hard one-hot cell-type labels → **cross-entropy loss**

| Component | Role |
|-----------|------|
| **UNI ViT-Large** | Foundation model for patch-level feature extraction |
| **CDAN** (Conditional Domain Adversarial Network) | Cross-patient domain adaptation |
| **Token Encoder** | 4-layer MLP mapping patch features → latent tokens |
| **RCTD** | Deconvolution of spot-level data into cell-type proportions (soft labels) |
| **Multi-Resolution** | Sequential or joint training on spot data and single-cell data |

---

## Installation

```bash
# 1. Clone / navigate to the repo
cd MeowCat

# 2. Install the meowcat CLI (requires Python >= 3.9, PyYAML)
pip install -e .

# 3. Verify
meowcat --help
```

The pipeline uses **three conda environments** for different steps:

| Environment | Steps | How to activate |
|-------------|-------|----------------|
| `rapids_singlecell` | Image preprocessing (Steps 2, 3, 6a) | `micromamba activate rapids_singlecell` |
| `he_anno` | Batch preparation, training, prediction, visualization (Steps 4-7) | `conda activate he_anno` |
| R / `RCTD` | RCTD deconvolution (Step 1) | `conda activate RCTD` |

> **Note:** The `meowcat` CLI generates the correct subprocess commands. Activate the appropriate environment before running each step, or configure cluster job scripts accordingly.

---

## Repository Layout

```
MeowCat/
├── config/
│   ├── default.yaml                 <- all paths & hyperparameters (copy & edit per experiment)
│   └── visualization_cmap.json      <- default cell-type color map for visualization
├── meowcat/
│   ├── __init__.py
│   ├── cli.py                       <- unified CLI entry point
│   ├── config.py                    <- YAML -> dataclass loader
│   ├── pipeline.py                  <- subprocess command builders
│   ├── Preprocess/
│   │   ├── RCTD_deconvolution.R         <- Step 1: RCTD spot deconvolution
│   │   ├── batched_data_preparing.py    <- Step 4: build training batch files
│   │   ├── prepare_inference_new_sample.py
│   │   └── ExtractFeatures/
│   │       ├── audit_resolution.py      <- Step 2: resolution audit
│   │       ├── get_pixel_size.py
│   │       ├── RunPreprocess.py         <- rescale & pad image
│   │       ├── RunHistoSweep.py         <- tissue masking
│   │       ├── UNI_extract_features.py  <- UNI ViT-Large feature extraction
│   │       └── UNI_fuse_features.py     <- global + local feature fusion
│   └── Train_predict/
│       ├── train_by_batch_cdan5.py               <- base classes (MultiTaskModel, GradReverse ...)
│       ├── train_by_batch_cdan5_trainc_final2.py <- Step 5: main training script
│       ├── predict_cdan_multireso.py             <- Step 6: full-grid prediction
│       ├── visualize_prediction_results.py       <- Step 6: cell-type intensity maps
│       └── visualize_slide_wrap.py               <- Step 7: PowerPoint summary
├── normalize_masks.py               <- utility: binarize HistoSweep masks
├── examples/                        <- runnable test cases (see Examples section)
├── tests/
│   └── test_config.py
└── pyproject.toml
```

---

## Data Requirements

### User-provided inputs (per sample)

```
<data_root>/<SAMPLE>/
├── he_raw.<tif|svs|jpg>           <- raw H&E image (filename must contain raw_flag, default: "he_raw")
│
│   -- Visium-specific --------------------------------------------------
├── filtered_feature_bc_matrix/    <- 10x Genomics Space Ranger output
├── spatial/                       <- scalefactors_json.json + tissue_positions_list.csv
│
│   -- Xenium-specific (user-provided) ---------------------------------
├── he_raw.<tif|svs|jpg>           <- raw H&E image (input to meowcat preprocess)
├── xenium_raw/                    <- 10x Xenium output (cell_feature_matrix.h5, cells.parquet)
├── adata_cellbin_HistoSweep.h5ad  <- cell-bin AnnData with bin-to-pixel location mapping
│                                     (.obsm['transformed_pxl_loc_in_morphology'] has pixel coords;
│                                      produced externally via HistoSweep alignment)
└── annotation.csv                 <- cell-type annotations (columns: cell_id, cell_state)
```

### Generated by the pipeline (do NOT create manually)

```
<data_root>/<SAMPLE>/
├── pixel-size-raw.txt       <- raw MPP from image metadata     (Step 3: get_pixel_size.py)
├── pixel-size.txt           <- target MPP                      (Step 3: prepare_visium_inputs.py)
├── embeddings-hist.pickle   <- UNI features [H, W, C]          (Step 3: UNI_fuse_features.py)
├── mask/
│   ├── mask.png             <- tissue mask (full resolution)    (Step 3: RunHistoSweep.py)
│   └── mask-small.png       <- tissue mask (downsampled)        (Step 3: RunHistoSweep.py)
│
│   -- Visium only (generated from spatial/ and RCTD output) ----
├── deconvolution_rctd/
│   └── major_prop.csv       <- RCTD cell-type proportions       (Step 1: RCTD_deconvolution.R)
├── anno-names.txt           <- cell-type names from RCTD cols   (Step 3: prepare_visium_inputs.py)
├── anno_matrix.tsv          <- spot × cell-type proportions     (Step 3: prepare_visium_inputs.py)
├── locs.tsv                 <- spot coords (processed pixels)   (Step 3: prepare_visium_inputs.py)
├── locs-raw.tsv             <- spot coords (raw pixels)         (Step 3: prepare_visium_inputs.py)
├── radius.txt               <- spot radius (processed pixels)   (Step 3: prepare_visium_inputs.py)
├── radius-raw.txt           <- spot radius (raw pixels)         (Step 3: prepare_visium_inputs.py)
│
│   -- Xenium only (generated by prepare-xenium-batches) --------
├── single_super_emb.h5ad                  <- UNI histology features  (Step 3: UNI_fuse_features.py)
├── Xenium_adata_cellbin_analysis_qv20.h5ad <- cellbin + histology_2048 (Step 4)
├── Xenium_adata_cell.h5ad                 <- cell data + annotations   (Step 4)
├── anno-names.txt                         <- copied from xenium.anno_names_path (Step 4)
└── alignment_visualization.png            <- QC overlay cells vs bins  (Step 4)
```

### Batch files (written by Step 4, consumed by Step 5)

```
<batches_dir>/
├── batch_vis_000_x.npy   [N, T, C] float16  -- Visium spot tokens
├── batch_vis_000_y.npy   [N, K]    float32  -- soft proportions from RCTD
├── batch_vis_000_d.npy   [N]       int64    -- domain IDs
├── batch_xen_000_x.npy   [N, T, C] float16  -- Xenium cell tokens
├── batch_xen_000_y.npy   [N, K]    float32  -- one-hot hard labels
└── batch_xen_000_d.npy   [N]       int64    -- domain IDs
```

The `vis` / `xen` prefix in the filename is **how the training script distinguishes Visium from Xenium**.
Files named `batch_vis_*` receive MSE loss; files named `batch_xen_*` receive cross-entropy loss.

---

## Pipeline Walkthrough

All steps are driven by a single config YAML. Copy and edit `config/default.yaml` for each experiment:

```bash
cp config/default.yaml config/my_run.yaml
# edit project.data_root, project.out_root, preprocess.uni_weights, etc.
```

Use `--dry-run` to preview the exact subprocess command without executing it:

```bash
meowcat <step> --config config/my_run.yaml --dry-run
```

---

### Step 1 — RCTD Deconvolution

*Required for Visium data only.*

Decomposes spot-level gene expression into cell-type proportions using RCTD. Configured via the `rctd` section of the config YAML — set `rctd.reference_rds` to your single-cell reference atlas.

```bash
# activate RCTD conda environment, then:
meowcat rctd --config config/my_run.yaml
```

**Group-based subsetting (optional):** If your reference contains multiple tissue/disease subtypes, set `rctd.group_column` and `rctd.groups` to build per-group references. The group is inferred from each sample's folder name. If these are not set, all reference cells are used as a single reference for every sample.

**Output per sample:** `deconvolution_rctd/major_prop.csv` — spot × cell-type proportion matrix. Consumed by Step 3 (`prepare_visium_inputs.py`) to generate `anno-names.txt` and `anno_matrix.tsv`, which are then used in Step 4 to build `batch_vis_*_y.npy`.

---

### Step 2 — Resolution Check

Audits all samples under `data_root` and reports whether images match `target_mpp`.

```bash
# activate rapids_singlecell, then:
meowcat check-resolution --config config/my_run.yaml
```

Controlled by `preprocess.target_mpp` (default `0.5` mpp) and `project.sample_pattern`.

**If auto-detection fails** (e.g. TIFF missing resolution tags), set `preprocess.pixel_size_raw` in your config to the known microns-per-pixel value. This skips metadata detection for both the resolution check and the preprocessing step:

```yaml
preprocess:
  pixel_size_raw: 0.5   # manual override — skip auto-detection
```

**Output:** printed table of actual vs. target mpp per sample; no files written.

---

### Step 3 — Image Preprocessing (training samples)

**This step is data-agnostic** — it processes H&E images identically for both Visium and Xenium samples.

Runs five sub-steps sequentially for each sample:

| Sub-step | Script | What it does |
|----------|--------|--------------|
| 1 | `get_pixel_size.py` | Reads image metadata → writes `pixel-size-raw.txt` |
| 2 | `RunPreprocess.py` | Rescales image to `target_mpp`, pads to multiples of `pad` (default 224) |
| 3 | `RunHistoSweep.py` | Tissue segmentation → `mask/mask.png` + `mask/mask-small.png` |
| 4 | `UNI_extract_features.py` | UNI ViT-Large sliding-window → intermediate feature maps |
| 5 | `UNI_fuse_features.py` | Fuses global + local UNI features → `single_super_emb.h5ad` |

```bash
# activate rapids_singlecell, then:
meowcat preprocess --config config/my_run.yaml

# process a subset of samples only:
meowcat preprocess --config config/my_run.yaml --samples GBM001,GBM002
```

**Output per sample:** `single_super_emb.h5ad` (feature fusion output).

> **For Xenium samples:** `meowcat preprocess` auto-discovers both Visium and Xenium samples — no `--samples` override needed. It produces `single_super_emb.h5ad` (UNI histology features). During `meowcat prepare-xenium-batches` (Step 4b), these features are automatically merged into `adata_cellbin_HistoSweep.h5ad` (the bin-to-pixel location mapping you provide externally via HistoSweep). Both files must be in the sample folder before running batch preparation.

> **Mixed Visium + Xenium setups:** Leave `preprocess.pixel_size_raw: null` (the default) so that MPP is auto-detected per sample from image metadata. H&E images from Visium and Xenium platforms often have different resolutions, so a single manual value would be incorrect for one platform.

> **Per-sample pixel size override:** To manually set the MPP for a specific sample, create a `pixel-size-raw.txt` file in that sample's folder containing the value (e.g., `0.4965`). The preprocess step will detect this file and skip auto-detection for that sample. This takes priority over both auto-detection and the global `preprocess.pixel_size_raw` config setting.

> **For prediction samples (Step 6a):** run the identical command after changing `project.sample_pattern` (or using `--samples`) to target the new slides.

---

### Step 3.5 — Visium Metadata Preparation

*Required for Visium data only.* Must run after Steps 1 and 3.
Non-Visium samples are auto-discovered but skipped gracefully.

Prepares Visium-specific metadata and converts embeddings:
- Reads RCTD output + spatial positions → `anno-names.txt`, `anno_matrix.tsv`, `locs.tsv`, `radius.txt`
- Converts `single_super_emb.h5ad` → `embeddings-hist.pickle` `[H, W, C]`
- Aligns barcodes between RCTD output and tissue positions to ensure row-level consistency

```bash
# activate he_anno, then:
meowcat prepare-visium --config config/my_run.yaml
```

---

### Step 4a — Visium Batch Preparation

Tokenizes Visium spots, applies coreset subsampling, and writes `.npy` batch files.

```bash
# activate he_anno, then:
meowcat prepare-visium-batches --config config/my_run.yaml
```

Key config knobs:

| Key | Default | Effect |
|-----|---------|--------|
| `batches.out_dir` | — | Where batch `.npy` files are written |
| `visium.keep_frac` | `0.25` | Fraction of spots retained by coreset |
| `visium.strategy` | `stratified` | Coreset strategy: `stratified` or `kcenter` |
| `visium.exclude_set` | `[]` | Sample names to skip |
| `visium.domain_map_tsv` | `null` | Optional TSV for multi-patient domain assignment |

**Output:** `batch_vis_XXX_x/y/d.npy` in `batches.out_dir`.

---

### Step 4b — Xenium Batch Preparation

Tokenizes Xenium cells, merges UNI histology features from `single_super_emb.h5ad` into the cellbin, applies coreset subsampling, and writes `.npy` batch files.

```bash
# activate he_anno, then:
meowcat prepare-xenium-batches --config config/my_run.yaml
```

Key config knobs (from the `xenium` section):

| Key | Default | Effect |
|-----|---------|--------|
| `xenium.anno_names_path` | — | Path to the shared `anno-names.txt` (cell-type list must match Visium) |
| `xenium.dapi_pixel_size_raw` | `0.2125` | Xenium DAPI instrument coordinate-to-pixel conversion factor |
| `xenium.keep_frac` | `null` | Fraction of valid cells to keep (`null` = use Visium batch_size or all cells) |

**Output:** `batch_xen_XXX_x/y/d.npy` in `batches.out_dir`.

---

### Step 5 — Training

```bash
# activate he_anno, then:
meowcat train --config config/my_run.yaml
```

The training script auto-detects which data types are present from the batch filenames:
- Only `batch_vis_*` present → Visium-only training (MSE loss)
- Only `batch_xen_*` present → Xenium-only training (CE loss)
- Both present + `sequential_training: true` → 3-phase training (recommended)

**Output:** `<batches.out_dir>/states/00/model.ckpt`, `01/model.ckpt`, … (one checkpoint per `n_states` replica).

For per-paradigm configuration details, see [Training Paradigms](#training-paradigms) and the [examples/](examples/) folder.

---

### Step 6 — Prediction & Visualization

**Step 6a** — preprocess new (prediction) samples, same as Step 3:

```bash
# activate rapids_singlecell, then:
meowcat preprocess --config config/my_run.yaml --samples P_new1,P_new2
```

**Step 6b** — full-grid prediction and visualization:

```bash
# activate he_anno, then:
meowcat predict   --config config/my_run.yaml --samples P_new1,P_new2
meowcat visualize --config config/my_run.yaml --samples P_new1,P_new2
```

`predict` loads all `n_states` checkpoints, runs the model over every pixel of `embeddings-hist.pickle`, and saves:

```
<data_root>/<SAMPLE>/pred_fullgrid_outputs.pkl
  z_map:  [H, W, D]  -- L2-normalized latent embeddings (median across states)
  p_map:  [H, W, K]  -- per-pixel cell-type probabilities (mean across states)
  ctypes: list[str]  -- cell-type names
```

`visualize` reads that pickle and generates figures in `<out_root>/<SAMPLE>/`:

- `<SAMPLE>_predicted_celltype_map.png` — predicted dominant cell type per pixel + legend
- `celltype_intensity_percentiles/masked_<ct>.png` — per-cell-type probability maps (percentile scaled)

**Custom color maps:** Cell-type colors are controlled by a JSON file (`visualize.cmap_json`). By default the bundled `config/visualization_cmap.json` is used. To customize, create a JSON with this format:

```json
{
  "cmap_name": "my_colors",
  "type_to_color": {
    "CellType1": "#ff0000",
    "CellType2": "#00ff00"
  }
}
```

Cell types not listed in the JSON fall back to the `tab20` matplotlib palette.

Key prediction config knobs:

| Key | Default | Effect |
|-----|---------|--------|
| `predict.tokens_per_chunk` | `70000` | GPU chunk size (reduce if OOM) |
| `predict.chunks_per_batch` | `2` | Chunks batched per forward pass |
| `visualize.n_clusters` | `6` | Number of KMeans clusters |
| `visualize.p_lo` / `p_hi` | `5` / `95` | Percentile clipping for intensity maps |
| `visualize.cmap_json` | `null` | Custom color map JSON (`null` = bundled default) |

---

### Step 7 — Slide Wrap

Assembles all per-sample result images into a single PowerPoint deck.

```bash
# activate he_anno, then:
meowcat slide --config config/my_run.yaml
```

**Output:** `<out_root>/results.pptx` (or path set by `slide.pptx`).

---

## Inference on New H&E Images

Once you have a trained model, use `meowcat infer` to predict cell types on new H&E images in a single command. This chains preprocessing, embedding extraction, prediction, visualization, and slide generation.

```bash
# activate he_anno (use rapids_singlecell if preprocessing is needed), then:
meowcat infer --config inference.yaml

# specific samples only:
meowcat infer --config inference.yaml --samples HE001,HE002

# skip preprocessing (already done):
meowcat infer --config inference.yaml --start-from 6

# preview commands without running:
meowcat infer --config inference.yaml --dry-run
```

### Inference config

The inference config is a minimal YAML with an `inference` section pointing to the trained model:

```yaml
inference:
  model_dir: /path/to/training/batches    # contains states/*/model.ckpt
  anno_names: /path/to/anno-names.txt     # cell-type list from training

project:
  data_root: /path/to/new_samples
  out_root:  /path/to/output
  sample_pattern: "HE*"

preprocess:
  raw_flag: he_raw
  target_mpp: 0.5
  pixel_size_raw: null
  uni_weights: /path/to/uni_weights.bin

predict:
  n_states: 2
  device: cuda

visualize:
  cmap_json: null
```

The command automatically copies `anno-names.txt` into each sample directory (required by the predict script). See [`examples/06_predict_new_sample/`](examples/06_predict_new_sample/) for a complete example.

---

## Config Reference

All parameters live in `config/default.yaml`. Commonly changed keys:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `project` | `name` | `meowcat_run` | Experiment name (for logging) |
| `project` | `data_root` | — | Root folder with per-sample subfolders |
| `project` | `out_root` | — | Root for all outputs |
| `visium` | `sample_pattern` | `VIS*` | Glob pattern to find Visium sample dirs |
| `rctd` | `reference_rds` | — | Path to single-cell reference RDS (Seurat v5) |
| `rctd` | `cell_type_column` | `MainType` | Metadata column for cell-type labels |
| `rctd` | `group_column` | `""` | Metadata column for group subsetting (`""` = use all cells) |
| `rctd` | `groups` | `[]` | Groups of interest (inferred from folder names) |
| `rctd` | `max_cores` | `5` | Parallel cores for RCTD fitting |
| `rctd` | `doublet_mode` | `full` | RCTD doublet mode |
| `rctd` | `min_umi` | `10` | Minimum UMI for reference cells |
| `preprocess` | `raw_flag` | `he_raw` | Substring in raw image filename |
| `preprocess` | `target_mpp` | `0.5` | Target resolution (microns-per-pixel) |
| `preprocess` | `pixel_size_raw` | `null` | Manual raw H&E image MPP override (`null` = auto-detect from image metadata; distinct from `xenium.dapi_pixel_size_raw` which is the Xenium instrument coordinate conversion factor) |
| `preprocess` | `pad` | `224` | Padding multiple for rescaled image |
| `preprocess` | `uni_weights` | — | Path to UNI ViT-Large `.bin` weights |
| `preprocess` | `fusion_mode` | `single` | Feature fusion: `single` or `multi` |
| `visium` | `include_only` | `null` | Sample names to include (null = all matching pattern) |
| `visium` | `exclude_set` | `[]` | Sample names to skip |
| `visium` | `domain_map_tsv` | `null` | TSV mapping sample names to domain strings |
| `visium` | `fixed_radius` | `null` | Override spot radius (null = from `radius.txt`) |
| `visium` | `keep_frac` | `0.25` | Coreset fraction of spots to keep |
| `visium` | `strategy` | `stratified` | `stratified` or `kcenter` |
| `batches` | `out_dir` | — | Where batch `.npy` files are written (shared by Visium and Xenium) |
| `train` | `n_states` | `2` | Independent model replicas (ensemble) |
| `train` | `two_stage` | `true` | Phase 0 reconstruction pretraining |
| `train` | `epochs1` | `15` | Reconstruction epochs |
| `train` | `sequential_training` | `true` | Sequential Visium -> Xenium |
| `train` | `visium_epochs` | `100` | Visium phase epochs |
| `train` | `xenium_epochs` | `100` | Xenium phase epochs |
| `train` | `freeze_encoder_n` | `2` | Encoder layers frozen at phase transitions |
| `train` | `xenium_weight` | `0.01` | Relative CE loss weight |
| `train` | `adv_lambda` | `0` | CDAN adversarial weight (0 = disabled) |
| `train` | `monitor_metric` | `val_weak_mse` | Metric for best-checkpoint selection |
| `predict` | `n_states` | `2` | Checkpoints to ensemble at inference |
| `predict` | `tokens_per_chunk` | `70000` | GPU chunk size |
| `visualize` | `n_clusters` | `6` | KMeans clusters for latent map |
| `visualize` | `cmap_json` | `null` | Custom cell-type color map JSON (`null` = bundled default) |
| `slide` | `pptx` | `results.pptx` | Output PowerPoint filename |

---

## Training Paradigms

### Visium only (soft labels, MSE)

All batch files are `batch_vis_*`. Use 2-phase training: reconstruction pretraining then Visium MSE.

```yaml
train:
  two_stage: true
  epochs1: 15
  sequential_training: false
  visium_epochs: 100
  xenium_epochs: 0
```

### Xenium only (hard labels, cross-entropy)

All batch files are `batch_xen_*`. Use 2-phase training: reconstruction pretraining then Xenium CE.

```yaml
train:
  two_stage: true
  epochs1: 15
  sequential_training: false
  visium_epochs: 0
  xenium_epochs: 100
```

### Multi-resolution: Visium then Xenium (recommended)

Both `batch_vis_*` and `batch_xen_*` present. 3-phase sequential training:

```yaml
train:
  two_stage: true
  epochs1: 15            # Phase 0: reconstruction pretraining
  sequential_training: true
  visium_epochs: 100     # Phase 1: Visium MSE
  xenium_epochs: 100     # Phase 2: Xenium CE fine-tuning
  freeze_encoder_n: 2    # freeze first 2 layers at each phase transition
  xenium_weight: 0.01
```

**GRL schedule (CDAN):** when `adv_lambda > 0`, lambda ramps from 0 to `adv_lambda` via sigmoid:
`lambda = adv_lambda * (2 / (1 + exp(-10 * p)) - 1)` where `p` = fraction of current phase done.

---

## Model Architecture

### MultiTaskModel (single-resolution)

```
Input x: [B, T, C]          (B spots, T tokens per disk, C UNI features)
         |
         v
net_lat  (4x FeedForward)   C -> token_dim=256
         |                  ELU activation, no residual
         v
z_tok:   [B, T, 256]        per-token latent
         |
   +-----+---------------------+
   v                           v
ct_head_tok               recon_head_tok      (optional Phase 0)
Linear(256, K)            Linear(256, C)
softmax -> [B,T,K]        -> x_recon [B,T,C]
mean over T ->
p_agg [B, K]             +-- GRL(lambda) --+
                           v                |
                       z_tok x p_tok        |   CDAN outer-product
                           v                |   feature [B,T,256*K]
                     domain_head            |   norm by sqrt(256*K)
                     -> logits [B,T,D]     <--
```

**Losses:**

| Loss | When | Formula |
|------|------|---------|
| `L_weak` | Phase 1/2 | `MSE(mean_T(softmax(z·W_ct)), y_soft)` for Visium; `NLL(argmax(y_hard))` for Xenium |
| `L_CDAN` | Phase 1/2, `adv_lambda > 0` | `CE(domain_head(GRL(z x p)), d)` |
| `L_recon` | Phase 0 | `MSE(x_recon, x_masked)` |
| **Total** | | `L_weak + lambda·L_CDAN + recon_weight·L_recon` |

### MultiResolutionModel (adds resolution embedding)

Identical to MultiTaskModel plus:
- `res_embed`: `Embedding(4, 256)` — resolution type embedding added to `z_tok` with 0.1 scaling
- **Spot data** (resolution=0): MSE on soft proportions from RCTD
- **Single-cell data** (resolution=1): NLL on `argmax(y_hard)` (cross-entropy on hard labels)
- `training_mode`: `'joint'` | `'spot_only'` | `'sc_only'`
- `sc_loss_weight`: relative weight of the Xenium CE loss in joint or sequential training

---

## Output Files

| File | Location | Shape | Description |
|------|----------|-------|-------------|
| `pixel-size-raw.txt` | `<sample>/` | — | Microns-per-pixel from image metadata |
| `embeddings-hist.pickle` | `<sample>/` | `[H, W, C]` float32 | UNI patch features |
| `batch_vis_XXX_x.npy` | `batches/` | `[N, T, C]` float16 | Visium spot input tokens |
| `batch_vis_XXX_y.npy` | `batches/` | `[N, K]` float32 | Soft cell-type proportions (RCTD) |
| `batch_xen_XXX_x.npy` | `batches/` | `[N, T, C]` float16 | Xenium cell input tokens |
| `batch_xen_XXX_y.npy` | `batches/` | `[N, K]` float32 | One-hot hard cell-type labels |
| `batch_*_d.npy` | `batches/` | `[N]` int64 | Domain IDs per spot/cell |
| `states/XX/model.ckpt` | `batches/` | — | PyTorch Lightning checkpoint |
| `pred_fullgrid_outputs.pkl` | `<sample>/` | dict | `z_map [H,W,D]`, `p_map [H,W,K]`, `ctypes` |

---

## Examples

See the [`examples/`](examples/) folder for seven ready-to-run test cases. All use paths under `/project/KidneyHE/01_meowcat_test/`.

| Folder | Training data | Training paradigm |
|--------|--------------|-------------------|
| [`examples/01_visium_only/`](examples/01_visium_only/) | 1 Visium sample | Recon → Visium MSE |
| [`examples/02_xenium_only/`](examples/02_xenium_only/) | 1 Xenium sample | Recon → Xenium CE |
| [`examples/03_visium_xenium_single/`](examples/03_visium_xenium_single/) | 1 Visium + 1 Xenium | Recon → Visium → Xenium (3-phase) |
| [`examples/04_multi_visium/`](examples/04_multi_visium/) | Multiple Visium samples | Multi-sample Visium MSE with CDAN |
| [`examples/04_multi_xenium/`](examples/04_multi_xenium/) | Multiple Xenium samples | Multi-sample Xenium CE with CDAN |
| [`examples/05_multi_visium_xenium/`](examples/05_multi_visium_xenium/) | 2 Visium + 2 Xenium | Multi-sample 3-phase with CDAN |
| [`examples/06_predict_new_sample/`](examples/06_predict_new_sample/) | New H&E images | Inference using trained model |

Each example contains a `config.yaml` and a `run.sh`. See [`examples/README.md`](examples/README.md) for details.
