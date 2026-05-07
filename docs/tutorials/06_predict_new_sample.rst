Example 06: Predict on New H&E Images
======================================

**Scenario:** Apply a previously trained MeowCat model to new H&E images (no omics data needed).

**Command:** ``meowcat infer`` chains preprocessing → embeddings → prediction → visualization → slide.

Prerequisites
-------------

- A trained MeowCat model from any of examples 01–05 (or your own training run)
- New H&E images placed under ``<data_root>/<SAMPLE>/he_raw.<ext>``

Data Setup
----------

.. code-block:: text

   <data_root>/
   └── HE001/
       └── he_raw.tif   <- new H&E image (no omics data needed)

Config
------

.. literalinclude:: ../../examples/06_predict_new_sample/config.yaml
   :language: yaml

Running
-------

.. literalinclude:: ../../examples/06_predict_new_sample/run.sh
   :language: bash

Useful Flags
------------

.. code-block:: bash

   # Skip preprocessing (already done):
   meowcat infer --config config.yaml --start-from 6

   # Process specific samples only:
   meowcat infer --config config.yaml --samples HE001,HE002

   # Dry run to preview commands:
   meowcat infer --config config.yaml --dry-run

Expected Output
---------------

.. code-block:: text

   <out_root>/HE001/
   ├── embeddings-hist.pickle or .npy
   ├── pred_fullgrid_outputs.pkl
   ├── argmax_map.png
   └── *_intensity.png
   results.pptx
