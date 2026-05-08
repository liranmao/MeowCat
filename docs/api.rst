Config API
==========

The ``MeowCatConfig`` dataclass tree mirrors the YAML structure section-by-section.
All parameters have sensible defaults; override them in your config YAML.

.. contents:: On this page
   :local:
   :depth: 1

Config dataclasses
------------------

.. autoclass:: meowcat.config.MeowCatConfig
   :members:
   :undoc-members:

.. autoclass:: meowcat.config.ProjectConfig
   :members:
   :undoc-members:

.. autoclass:: meowcat.config.RctdConfig
   :members:
   :undoc-members:

.. autoclass:: meowcat.config.PreprocessConfig
   :members:
   :undoc-members:

.. autoclass:: meowcat.config.VisiumConfig
   :members:
   :undoc-members:

.. autoclass:: meowcat.config.BatchesConfig
   :members:
   :undoc-members:

.. autoclass:: meowcat.config.XeniumConfig
   :members:
   :undoc-members:

.. autoclass:: meowcat.config.TrainConfig
   :members:
   :undoc-members:

.. autoclass:: meowcat.config.PredictConfig
   :members:
   :undoc-members:

.. autoclass:: meowcat.config.VisualizeConfig
   :members:
   :undoc-members:

.. autoclass:: meowcat.config.InferenceConfig
   :members:
   :undoc-members:

.. autoclass:: meowcat.config.SlideConfig
   :members:
   :undoc-members:

.. autofunction:: meowcat.config.load_config

Pipeline functions
------------------

These functions build the subprocess command for each pipeline step.
Each returns a ``List[str]`` that ``cli.py`` passes to ``subprocess.run()``.

.. autofunction:: meowcat.pipeline.cmd_rctd

.. autofunction:: meowcat.pipeline.cmd_check_resolution

.. autofunction:: meowcat.pipeline.cmds_preprocess_sample

.. autofunction:: meowcat.pipeline.cmds_prepare_visium_sample

.. autofunction:: meowcat.pipeline.cmd_visualize_visium

.. autofunction:: meowcat.pipeline.cmd_prepare_visium_batches

.. autofunction:: meowcat.pipeline.cmd_prepare_xenium_batches

.. autofunction:: meowcat.pipeline.cmd_slim_xenium

.. autofunction:: meowcat.pipeline.cmd_train

.. autofunction:: meowcat.pipeline.cmd_predict_sample

.. autofunction:: meowcat.pipeline.cmd_visualize_sample

.. autofunction:: meowcat.pipeline.cmd_slide
