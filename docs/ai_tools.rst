AI Tools
========

All AI helper files live in ``agent_helpers/`` at the repository root:

.. code-block:: text

   agent_helpers/
   ├── CLAUDE.md                  <- project context (auto-loaded by Claude Code from repo root)
   ├── meowcat-setup.md           <- /meowcat-setup skill definition
   ├── meowcat-check.md           <- /meowcat-check skill definition
   └── config_scaffold_prompt.md  <- copy-paste prompt for any AI assistant

Claude Code Skills
------------------

If you use `Claude Code <https://claude.ai/code>`_, two slash commands are available
automatically when you open the MeowCat repository:

``/meowcat-setup``
~~~~~~~~~~~~~~~~~~

An interactive config generator. Claude will ask about your data modality
(Visium / Xenium / both), sample count, file paths, and reference files — then write a
complete, ready-to-run ``config.yaml`` with the correct training paradigm selected
automatically. It finishes by printing the exact ``meowcat`` commands to run next.

.. code-block:: text

   /meowcat-setup

Skill definition: ``agent_helpers/meowcat-setup.md``

``/meowcat-check``
~~~~~~~~~~~~~~~~~~

Validates your config and data layout before running the pipeline. It reads the config,
checks every required path and file for each sample, and prints a ✓/✗/⚠ summary report
with actionable error messages.

.. code-block:: text

   /meowcat-check config/my_run.yaml

Skill definition: ``agent_helpers/meowcat-check.md``

.. note::
   Claude Code skills must be placed in ``.claude/commands/`` to be recognised. The
   repository ships with copies there automatically — ``agent_helpers/`` is the
   human-readable source of truth.

Config Scaffolding Prompt
--------------------------

For any other AI assistant (ChatGPT, Gemini, Cursor, etc.), use the ready-made prompt in
``agent_helpers/config_scaffold_prompt.md``. Fill in the ``[bracketed]`` fields and the
AI will generate a correct ``config.yaml`` and list the commands to run.

.. literalinclude:: ../agent_helpers/config_scaffold_prompt.md
   :language: text
   :start-after: ---

CLAUDE.md
---------

``agent_helpers/CLAUDE.md`` (also at the repo root as ``CLAUDE.md``) is loaded
automatically by Claude Code when you open the project. It gives the assistant instant
context about:

- The role of every key file (``cli.py``, ``pipeline.py``, ``config.py``)
- All pipeline steps and their order
- Training paradigm selection rules
- Required input files per sample
- Common development tasks

No setup needed — Claude Code reads it automatically on project open.
