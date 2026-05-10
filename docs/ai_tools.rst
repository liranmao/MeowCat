AI Tools
========

MeowCat ships several resources that make AI coding assistants significantly more useful
when adapting the pipeline to a new dataset.

Claude Code Skills
------------------

If you use `Claude Code <https://claude.ai/code>`_, two slash commands are available
automatically when you open the MeowCat repository:

``/meowcat-setup``
~~~~~~~~~~~~~~~~~~

An interactive config generator. Invoke it and Claude will ask about your data modality
(Visium / Xenium / both), sample count, file paths, and reference files — then write a
complete, ready-to-run ``config.yaml`` with the correct training paradigm selected
automatically. It also tells you the exact ``meowcat`` commands to run next.

.. code-block:: text

   /meowcat-setup

``/meowcat-check``
~~~~~~~~~~~~~~~~~~

Validates your config and data layout before running the pipeline. It checks every
required path and file for each sample and prints a clear pass/fail summary with
actionable error messages.

.. code-block:: text

   /meowcat-check config/my_run.yaml

Skills are defined in ``.claude/commands/`` in the repository root and work with any
version of Claude Code.

Config Scaffolding Prompt
--------------------------

If you use a different AI assistant (ChatGPT, Gemini, Cursor, etc.), copy the prompt
below and paste it into your chat. Fill in the ``[bracketed]`` fields, and the AI will
generate a correct ``config.yaml`` for your dataset.

.. code-block:: text

   I am using MeowCat, a CLI pipeline for cell-type annotation in H&E images.
   Help me generate a config.yaml for my dataset.

   My dataset:
   - Data modality: [Visium only / Xenium only / Visium + Xenium]
   - Number of samples: [single sample / N samples from different patients]
   - data_root: [/absolute/path/to/samples]
   - out_root:  [/absolute/path/to/outputs]
   - UNI weights (.bin file): [/absolute/path/to/pytorch_model.bin]

   Visium (if applicable):
   - Sample folder name pattern (glob): [e.g. VIS*, P*]
   - Single-cell reference RDS (Seurat v5): [/path/to/reference.rds]
   - Cell-type label column in reference: [e.g. MainType]
   - Group column for per-group reference subsetting: [e.g. Group_ID, or leave blank]
   - Groups to include: [e.g. LUAD, or leave blank to use all cells]

   Xenium (if applicable):
   - Sample folder name pattern (glob): [e.g. XEN*, *_Xenium]
   - anno-names.txt path: [/path/to/anno-names.txt]
   - Cell-type mapping JSON (fine→coarse): [/path/to/mapping.json, or leave blank]

   Training paradigm rules (apply automatically):
   - Visium only, 1 sample:  sequential_training=false, visium_epochs=100, xenium_epochs=0,  adv_lambda=0
   - Visium only, multi:     sequential_training=false, visium_epochs=100, xenium_epochs=0,  adv_lambda=0.005
   - Xenium only, 1 sample:  sequential_training=false, visium_epochs=0,   xenium_epochs=100, adv_lambda=0
   - Xenium only, multi:     sequential_training=false, visium_epochs=0,   xenium_epochs=100, adv_lambda=0.005
   - Both, 1 pair:           sequential_training=true,  visium_epochs=100, xenium_epochs=50,  adv_lambda=0
   - Both, multi:            sequential_training=true,  visium_epochs=100, xenium_epochs=100, adv_lambda=0.005
   Always set: two_stage=true, epochs1=15, n_states=2, freeze_encoder_n=2.
   Use monitor_metric=val_weak_mse for Visium; val_loss for Xenium-only.

   Please generate the complete config.yaml (all sections, with comments).
   Then list the exact meowcat commands I need to run in order.

CLAUDE.md
---------

The repository root contains a ``CLAUDE.md`` file that is automatically loaded by
Claude Code when you open the project. It gives the assistant instant context about:

- The role of every key file (``cli.py``, ``pipeline.py``, ``config.py``)
- All pipeline steps and their order
- Training paradigm selection rules
- Required input files per sample
- Common tasks (adding a subcommand, changing defaults, debugging)

No setup needed — Claude Code reads it automatically on project open.
