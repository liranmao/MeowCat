# MeowCat — Claude Code Context

MeowCat is a CLI tool (`meowcat`) for cell-type annotation in H&E histopathology images,
supervised by spatially registered transcriptomics (Visium spots or Xenium single cells).
Single conda environment: **`he_anno`** — all steps run in it.

## Key files

| File | Role |
|------|------|
| `meowcat/cli.py` | Unified CLI; `_build_parser()` defines all subcommands |
| `meowcat/pipeline.py` | Builds subprocess `List[str]` for each pipeline step |
| `meowcat/config.py` | YAML → nested dataclass loader; all config fields live here |
| `config/default.yaml` | Master config template — copy and edit per experiment |
| `examples/01–06/` | Seven ready-to-run examples with `config.yaml` + `run.sh` |

## Pipeline steps (in order)

```bash
meowcat rctd                   # Step 1:   RCTD deconvolution (Visium only)
meowcat check-resolution       # Step 2:   audit image MPP across samples
meowcat preprocess             # Step 3:   rescale + tissue mask + UNI features
meowcat prepare-visium         # Step 3.5: Visium metadata + embeddings-hist.pickle/.npy
meowcat prepare-visium-batches # Step 4a:  tokenize Visium spots → batch_vis_*.npy
meowcat prepare-xenium-batches # Step 4b:  tokenize Xenium cells → batch_xen_*.npy
meowcat train                  # Step 5:   train model (auto-detects vis/xen from filenames)
meowcat predict                # Step 6:   full-grid cell-type prediction
meowcat visualize              # Step 6:   generate argmax map + intensity maps
meowcat slide                  # Step 7:   PowerPoint summary
meowcat infer                  # Steps 3–7 in one command (new H&E images, no omics needed)
meowcat slim-xenium            # Utility:  shrink Xenium cellbin h5ad (~100GB → ~5GB)
```

All subcommands accept `--config <yaml>` and `--dry-run`.

## Training paradigms (set in `train:` section)

| Scenario | `sequential_training` | `visium_epochs` | `xenium_epochs` | `adv_lambda` |
|----------|-----------------------|-----------------|-----------------|--------------|
| Visium only, 1 sample | `false` | 100 | 0 | 0 |
| Visium only, multi | `false` | 100 | 0 | 0.005 |
| Xenium only, 1 sample | `false` | 0 | 100 | 0 |
| Xenium only, multi | `false` | 0 | 100 | 0.005 |
| Both, 1 pair | `true` | 100 | 50 | 0 |
| Both, multi | `true` | 100 | 100 | 0.005 |

Always keep: `two_stage: true`, `epochs1: 15`, `n_states: 2`.

## Required config fields (must be set before running)

```yaml
project:
  data_root: /abs/path/to/samples    # folder with per-sample subfolders
  out_root:  /abs/path/to/outputs
preprocess:
  uni_weights: /abs/path/to/pytorch_model.bin   # UNI ViT-Large weights
batches:
  out_dir: /abs/path/to/batches

# Visium only:
rctd:
  reference_rds: /abs/path/to/reference.rds
visium:
  sample_pattern: "VIS*"

# Xenium only / mixed:
xenium:
  sample_pattern: "XEN*"
  anno_names_path: /abs/path/to/anno-names.txt
```

## Required input files per sample

**Visium:** `he_raw.*`, `filtered_feature_bc_matrix/`, `spatial/scalefactors_json.json`, `spatial/tissue_positions*.csv`

**Xenium:** `he_raw.*`, `xenium_raw/cell_feature_matrix.h5`, `xenium_raw/cells.parquet`, `adata_cellbin_HistoSweep.h5ad`, `annotation.csv`

## Available skills

- `/meowcat-setup` — interactive config generator; asks about your data and writes a ready-to-run `config.yaml`
- `/meowcat-check` — validates all paths and per-sample file layout before running the pipeline

## Common tasks

**Add a new subcommand:** add a handler in `cli.py`, register in `_HANDLERS`, add the subprocess builder in `pipeline.py`.

**Change config defaults:** edit the dataclass in `config.py` and `config/default.yaml`.

**Debug a failed step:** check `log.txt` in the example folder; re-run with `--dry-run` to print commands; use `--start-from N` to resume preprocessing from a specific substep.
