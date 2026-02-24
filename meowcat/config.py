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
    sample_pattern: str = "GBM*"


@dataclass
class PreprocessConfig:
    raw_flag: str = "he_raw"
    target_mpp: float = 0.5
    pad: int = 224
    uni_weights: str = "/path/to/uni_weights.bin"
    histosweep_mask_dir: str = "./mask/"
    fusion_mode: str = "single"


@dataclass
class BatchesConfig:
    out_dir: str = "/path/to/batches"
    keep_frac: float = 0.25
    strategy: str = "stratified"
    seed: int = 0
    include_only: Optional[List[str]] = None
    exclude_set: List[str] = field(default_factory=list)
    domain_map_tsv: Optional[str] = None
    fixed_radius: Optional[float] = None
    radius_multiplier: float = 2.0


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
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    batches: BatchesConfig = field(default_factory=BatchesConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)
    visualize: VisualizeConfig = field(default_factory=VisualizeConfig)
    slide: SlideConfig = field(default_factory=SlideConfig)


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

    section_map = {
        "project": cfg.project,
        "preprocess": cfg.preprocess,
        "batches": cfg.batches,
        "train": cfg.train,
        "predict": cfg.predict,
        "visualize": cfg.visualize,
        "slide": cfg.slide,
    }

    for section, obj in section_map.items():
        if section in raw and raw[section]:
            _update_dataclass(obj, raw[section])

    return cfg
