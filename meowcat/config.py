"""
meowcat/config.py
Loads config/default.yaml (or a user-supplied YAML) into a nested dataclass.
No heavy dependencies — only PyYAML + stdlib.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class ProjectConfig:
    name: str = "meowcat_run"
    data_root: str = "/path/to/data"
    out_root: str = "/path/to/outputs"


@dataclass
class RctdConfig:
    # Absolute path to single-cell reference RDS file (Seurat v5 object)
    reference_rds: str = ""
    # Metadata column in reference for cell-type labels
    cell_type_column: str = "MainType"
    # Metadata column for group-based subsetting ("" = use all cells as one reference)
    group_column: str = ""
    # Groups of interest for subsetting ([] = use all cells)
    groups: List[str] = field(default_factory=list)
    # Maximum parallel cores for RCTD fitting
    max_cores: int = 5
    # RCTD doublet mode: "full", "doublet", or "multi"
    doublet_mode: str = "full"
    # Minimum UMI count for reference cells
    min_umi: int = 10


@dataclass
class PreprocessConfig:
    raw_flag: str = "he_raw"
    target_mpp: float = 0.5
    # Manual pixel size override (microns-per-pixel of the raw image).
    # null = auto-detect from image metadata; set a float value to skip detection.
    pixel_size_raw: Optional[float] = None
    pad: int = 224
    uni_weights: str = "/path/to/uni_weights.bin"
    fusion_mode: str = "single"


@dataclass
class VisiumConfig:
    sample_pattern: str = "VIS*"
    include_only: Optional[List[str]] = None
    exclude_set: List[str] = field(default_factory=list)
    domain_map_tsv: Optional[str] = None
    fixed_radius: Optional[float] = None


@dataclass
class BatchesConfig:
    out_dir: str = "/path/to/batches"
    keep_frac: float = 0.25
    strategy: str = "stratified"
    seed: int = 0


@dataclass
class XeniumConfig:
    # Uses project.data_root for sample discovery. Each Xenium sample folder should contain:
    #   xenium_raw/                    — raw xenium data (cell_feature_matrix.h5, cells.parquet)
    #   adata_cellbin_HistoSweep.h5ad  — cellbin h5ad with HistoSweep features
    #   annotation.csv                 — cell type annotations
    #   (optional) single_super_emb.h5ad — histology embeddings (if cellbin lacks histology_2048)
    #
    # Glob pattern for Xenium sample folders under data_root
    sample_pattern: Optional[str] = None
    # Samples to include (null = all matching pattern)
    include_only: Optional[List[str]] = None
    # Samples to exclude
    exclude_set: List[str] = field(default_factory=list)
    # Output directory for batch_xen_*_{x,y,d}.npy files
    out_dir: str = "/path/to/xenium/batches"
    # Pixel size of Xenium DAPI images (microns per pixel; 0.2125 for standard Xenium).
    # Used to convert Xenium instrument coordinates (microns) to pixel coordinates.
    dapi_pixel_size_raw: float = 0.2125
    # Fixed radius in pixels for KDTree bin-to-cell mapping
    fixed_radius: float = 75.20
    # Path to shared anno-names.txt (defines cell type order for all Xenium samples)
    anno_names_path: str = "/path/to/anno-names.txt"
    # Path to cell type mapping JSON (fine -> coarse labels)
    cell_type_mapping_json: Optional[str] = None
    # Visium batch directory (to determine domain ID offset; null = start from 0)
    visium_batch_dir: Optional[str] = None
    seed: int = 42


@dataclass
class TrainConfig:
    n_states: int = 2
    two_stage: bool = True
    epochs1: int = 15
    sequential_training: bool = True
    visium_epochs: int = 100
    xenium_epochs: int = 100
    adv_lambda: float = 0.0
    freeze_encoder_n: int = 2
    recon_weight: float = 0.1
    recon_mask_ratio: float = 0.3
    save_every_n_epochs: int = 10
    xenium_weight: float = 0.01
    monitor_metric: str = "val_weak_mse"
    oos_sample: Optional[str] = None
    oos_tmpdir: Optional[str] = None
    device: str = "cuda"


@dataclass
class PredictConfig:
    n_states: int = 2
    tokens_per_chunk: int = 70000
    chunks_per_batch: int = 2
    out_pkl_name: str = "pred_fullgrid_outputs.pkl"
    device: str = "cuda"


@dataclass
class VisualizeConfig:
    n_clusters: int = 6
    pca_comp: int = 100
    random_seed: int = 0
    p_lo: int = 5
    p_hi: int = 95
    save_highlights: bool = False
    # Path to custom cell-type color map JSON.
    # null = use bundled default (config/visualization_cmap.json)
    cmap_json: Optional[str] = None


@dataclass
class InferenceConfig:
    # Path to batches directory containing states/*/model.ckpt from training
    model_dir: str = ""
    # Path to anno-names.txt from training (copied into each new sample dir)
    anno_names: str = ""


@dataclass
class SlideConfig:
    pptx: str = "results.pptx"
    intensity_cols: int = 3
    intensity_rows: int = 2
    highlight_cols: int = 4
    highlight_rows: int = 3


@dataclass
class MeowCatConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    rctd: RctdConfig = field(default_factory=RctdConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    visium: VisiumConfig = field(default_factory=VisiumConfig)
    batches: BatchesConfig = field(default_factory=BatchesConfig)
    xenium: XeniumConfig = field(default_factory=XeniumConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)
    visualize: VisualizeConfig = field(default_factory=VisualizeConfig)
    slide: SlideConfig = field(default_factory=SlideConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


def _update_dataclass(obj, data: Dict[str, Any]) -> None:
    """Recursively update a dataclass from a dict, ignoring unknown keys."""
    valid = {f.name for f in fields(obj)}
    for k, v in data.items():
        if k not in valid:
            continue
        setattr(obj, k, v)


def load_config(path: Optional[str] = None) -> MeowCatConfig:
    """
    Load a MeowCatConfig from a YAML file.
    Falls back to config/default.yaml relative to this file's package root.

    Parameters
    ----------
    path : str or None
        Path to YAML config file. If None, loads config/default.yaml.
    """
    if path is None:
        # default.yaml lives one level up from this file (project root/config/)
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "..", "config", "default.yaml")

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    cfg = MeowCatConfig()

    # ── Backward compat: migrate old keys into 'visium' section ──────────
    if "visium" not in raw:
        raw["visium"] = {}
    visium_raw = raw["visium"] or {}

    # project.sample_pattern -> visium.sample_pattern
    proj_raw = raw.get("project") or {}
    if "sample_pattern" in proj_raw and "sample_pattern" not in visium_raw:
        visium_raw["sample_pattern"] = proj_raw.pop("sample_pattern")

    # batches.{include_only,exclude_set,domain_map_tsv,fixed_radius} -> visium.*
    batches_raw = raw.get("batches") or {}
    for key in ("include_only", "exclude_set", "domain_map_tsv", "fixed_radius"):
        if key in batches_raw and key not in visium_raw:
            visium_raw[key] = batches_raw.pop(key)

    raw["visium"] = visium_raw

    section_map = {
        "project": cfg.project,
        "rctd": cfg.rctd,
        "preprocess": cfg.preprocess,
        "visium": cfg.visium,
        "batches": cfg.batches,
        "xenium": cfg.xenium,
        "train": cfg.train,
        "predict": cfg.predict,
        "visualize": cfg.visualize,
        "slide": cfg.slide,
        "inference": cfg.inference,
    }

    for section, obj in section_map.items():
        if section in raw and raw[section]:
            data = raw[section]
            # Backward compat: xenium.pixel_size_raw -> xenium.dapi_pixel_size_raw
            if section == "xenium" and "pixel_size_raw" in data and "dapi_pixel_size_raw" not in data:
                data["dapi_pixel_size_raw"] = data.pop("pixel_size_raw")
            _update_dataclass(obj, data)

    return cfg
