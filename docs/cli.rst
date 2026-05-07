CLI Reference
=============

All commands follow the pattern::

   meowcat <command> --config config/my_run.yaml [--samples S1,S2] [--dry-run]

.. argparse::
   :module: meowcat.cli
   :func: _build_parser
   :prog: meowcat
