"""
tests/test_config.py
Unit tests for the MeowCat config layer.
No GPU, no data files required — pure Python.
"""

import os
import sys
import tempfile
import textwrap

import pytest

# Make the repo root importable when running tests from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from meowcat.config import load_config, MeowCatConfig


DEFAULT_YAML = os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml")


class TestDefaultConfig:
    def test_load_default_yaml(self):
        cfg = load_config(DEFAULT_YAML)
        assert isinstance(cfg, MeowCatConfig)

    def test_default_values(self):
        cfg = load_config(DEFAULT_YAML)
        assert cfg.preprocess.target_mpp == 0.5
        assert cfg.preprocess.pad == 224
        assert cfg.train.n_states == 2
        assert cfg.train.two_stage is True
        assert cfg.visium.keep_frac == 0.25
        assert cfg.visium.strategy == "stratified"
        assert cfg.visualize.n_clusters == 6

    def test_section_types(self):
        from meowcat.config import (
            ProjectConfig, PreprocessConfig, BatchesConfig,
            TrainConfig, PredictConfig, VisualizeConfig, SlideConfig,
        )
        cfg = load_config(DEFAULT_YAML)
        assert isinstance(cfg.project, ProjectConfig)
        assert isinstance(cfg.preprocess, PreprocessConfig)
        assert isinstance(cfg.batches, BatchesConfig)
        assert isinstance(cfg.train, TrainConfig)
        assert isinstance(cfg.predict, PredictConfig)
        assert isinstance(cfg.visualize, VisualizeConfig)
        assert isinstance(cfg.slide, SlideConfig)


class TestCustomConfig:
    def _write_yaml(self, content: str) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        f.write(textwrap.dedent(content))
        f.close()
        return f.name

    def test_override_data_root(self):
        path = self._write_yaml("""
            project:
              data_root: /my/custom/data
        """)
        cfg = load_config(path)
        assert cfg.project.data_root == "/my/custom/data"
        # untouched field keeps default
        assert cfg.preprocess.target_mpp == 0.5
        os.unlink(path)

    def test_override_train_params(self):
        path = self._write_yaml("""
            train:
              n_states: 5
              visium_epochs: 200
              adv_lambda: 0.01
        """)
        cfg = load_config(path)
        assert cfg.train.n_states == 5
        assert cfg.train.visium_epochs == 200
        assert cfg.train.adv_lambda == pytest.approx(0.01)
        os.unlink(path)

    def test_unknown_keys_ignored(self):
        path = self._write_yaml("""
            project:
              data_root: /data
              nonexistent_key: should_be_ignored
        """)
        cfg = load_config(path)
        assert cfg.project.data_root == "/data"
        assert not hasattr(cfg.project, "nonexistent_key")
        os.unlink(path)

    def test_empty_yaml_uses_defaults(self):
        path = self._write_yaml("")
        cfg = load_config(path)
        assert cfg.train.n_states == 2
        os.unlink(path)

    def test_exclude_set_is_list(self):
        path = self._write_yaml("""
            visium:
              exclude_set: [P21_LUAD, P_bad]
        """)
        cfg = load_config(path)
        assert cfg.visium.exclude_set == ["P21_LUAD", "P_bad"]
        os.unlink(path)

    def test_backward_compat_batches_keep_frac(self):
        path = self._write_yaml("""
            batches:
              keep_frac: 0.5
              strategy: kcenter
              seed: 42
        """)
        cfg = load_config(path)
        assert cfg.visium.keep_frac == 0.5
        assert cfg.visium.strategy == "kcenter"
        assert cfg.visium.seed == 42
        os.unlink(path)


class TestCLIDryRun:
    """Smoke-test the CLI argument parser without subprocess execution."""

    def test_help_does_not_crash(self):
        from meowcat.cli import _build_parser
        p = _build_parser()
        with pytest.raises(SystemExit) as exc:
            p.parse_args(["--help"])
        assert exc.value.code == 0

    def test_subcommand_parsed(self):
        from meowcat.cli import _build_parser
        p = _build_parser()
        args = p.parse_args(["train", "--dry-run"])
        assert args.step == "train"
        assert args.dry_run is True

    def test_preprocess_samples_arg(self):
        from meowcat.cli import _build_parser
        p = _build_parser()
        args = p.parse_args(["preprocess", "--samples", "S1,S2"])
        assert args.step == "preprocess"
        assert args.samples == "S1,S2"

    def test_pipeline_cmd_train_contains_prefix(self):
        from meowcat.pipeline import cmd_train
        cfg = load_config(DEFAULT_YAML)
        cfg.batches.out_dir = "/tmp/batches"
        cmd = cmd_train(cfg)
        assert "/tmp/batches" in cmd
        assert "--n-states" in cmd
        assert "--two-stage" in cmd

    def test_pipeline_cmd_predict_contains_sample(self):
        from meowcat.pipeline import cmd_predict_sample
        cfg = load_config(DEFAULT_YAML)
        cfg.project.data_root = "/tmp/data"
        cmd = cmd_predict_sample(cfg, "GBM001")
        assert "GBM001" in cmd
        assert "--tokens-per-chunk" in cmd
