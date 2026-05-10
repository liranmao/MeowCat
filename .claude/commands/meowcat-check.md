Validate a MeowCat config file and data layout before running the pipeline.

If $ARGUMENTS is provided, treat it as the path to the config YAML. Otherwise ask the user for the path.

---

## What to check

### 1. Read the config
Use the Read tool to load the YAML. Parse it to understand which modalities are active:
- **Visium active** if `visium.sample_pattern` is set (non-null, non-empty)
- **Xenium active** if `xenium.sample_pattern` is set (non-null, non-empty)

---

### 2. Top-level path checks

Use Bash `test -d` / `test -f` to check each path. Report ✓ / ✗ / ⚠:

| Check | Command | Status if missing |
|-------|---------|-------------------|
| `project.data_root` exists as directory | `test -d <path>` | ✗ ERROR |
| `project.out_root` exists or is creatable | `test -d <path>` | ⚠ will be created |
| `preprocess.uni_weights` exists as file | `test -f <path>` | ✗ ERROR |
| `batches.out_dir` exists or is creatable | `test -d <path>` | ⚠ will be created |
| `rctd.reference_rds` (if non-empty) | `test -f <path>` | ✗ ERROR (Visium only) |
| `xenium.anno_names_path` (if set) | `test -f <path>` | ✗ ERROR (Xenium only) |
| `xenium.cell_type_mapping_json` (if set) | `test -f <path>` | ✗ ERROR |
| `visualize.cmap_json` (if set) | `test -f <path>` | ✗ ERROR |
| `inference.model_dir` (if non-empty) | `test -d <path>` | ✗ ERROR (infer mode) |
| `inference.anno_names` (if non-empty) | `test -f <path>` | ✗ ERROR (infer mode) |

---

### 3. Discover samples

Use Bash glob to find matching directories:
```bash
ls -d <data_root>/<visium_pattern>/ 2>/dev/null
ls -d <data_root>/<xenium_pattern>/ 2>/dev/null
```

Report how many Visium and Xenium samples were found. If zero found, report ✗ ERROR — check `data_root` and `sample_pattern`.

Apply `include_only` and `exclude_set` filters if configured, and report which samples are excluded.

---

### 4. Per-sample file checks

**For each Visium sample** (matching `visium.sample_pattern`):

| File / folder | Required? | Notes |
|---------------|-----------|-------|
| `he_raw.*` (any file matching `preprocess.raw_flag`) | ✗ ERROR | Raw H&E image |
| `filtered_feature_bc_matrix/` directory | ✗ ERROR | 10x Space Ranger output |
| `spatial/scalefactors_json.json` | ✗ ERROR | Spot scale factors |
| `spatial/tissue_positions_list.csv` or `tissue_positions.csv` | ✗ ERROR | Spot coordinates |
| `deconvolution_rctd/major_prop.csv` | ⚠ INFO | Present if Step 1 (RCTD) already run |
| `single_super_emb.h5ad` | ⚠ INFO | Present if Step 3 (preprocess) already run |
| `embeddings-hist.pickle` or `embeddings-hist.npy` | ⚠ INFO | Present if Step 3.5 already run |

**For each Xenium sample** (matching `xenium.sample_pattern`):

| File / folder | Required? | Notes |
|---------------|-----------|-------|
| `he_raw.*` | ✗ ERROR | Raw H&E image |
| `xenium_raw/cell_feature_matrix.h5` | ✗ ERROR | 10x Xenium output |
| `xenium_raw/cells.parquet` | ✗ ERROR | Cell coordinates |
| `adata_cellbin_HistoSweep.h5ad` | ✗ ERROR | Cell-bin location mapping (external alignment) |
| `annotation.csv` | ✗ ERROR | Cell-type annotations (cell_id, cell_state columns) |
| `single_super_emb.h5ad` | ⚠ INFO | Present if Step 3 (preprocess) already run |

---

### 5. Print summary report

```
════════════════════════════════════════
 MeowCat Config Check
════════════════════════════════════════
 Config:      /path/to/config.yaml
 Modalities:  Visium + Xenium   (or: Visium only / Xenium only)

 Global paths
 ────────────
 data_root:      ✓  /path/to/data
 out_root:       ⚠  /path/to/out  (will be created)
 uni_weights:    ✓  /path/to/pytorch_model.bin
 batches.out_dir:✓  /path/to/batches
 reference_rds:  ✓  /path/to/ref.rds
 anno_names:     ✓  /path/to/anno-names.txt

 Visium samples found: 2
 ───────────────────────────────────────────────
  VIS_P01   ✓ H&E  ✓ SpaceRanger  ✓ spatial  ⚠ RCTD not yet run
  VIS_P02   ✓ H&E  ✗ filtered_feature_bc_matrix/ MISSING

 Xenium samples found: 1
 ───────────────────────────────────────────────
  XEN_P01   ✓ H&E  ✓ xenium_raw  ✓ cellbin  ✓ annotation

════════════════════════════════════════
 Result: 1 error, 1 warning
════════════════════════════════════════
```

Use ✓ = found/ok, ✗ = missing/blocks pipeline (ERROR), ⚠ = optional or informational (WARNING).

Count total errors and warnings. If errors > 0, tell the user what must be fixed before running. If warnings only, explain they are informational and the pipeline can proceed.
