Example 01: Visium Only
=======================

**Scenario:** Single Visium sample, no Xenium data.

**Training paradigm:** Reconstruction pretraining (Phase 0) → Visium MSE (Phase 1).

Data Setup
----------

.. code-block:: text

   <data_root>/VIS_S1/
   ├── he_raw.tif                   <- raw H&E image
   ├── filtered_feature_bc_matrix/  <- 10x Space Ranger output
   └── spatial/                     <- scalefactors_json.json + tissue_positions_list.csv

Config
------

.. literalinclude:: ../../examples/01_visium_only/config.yaml
   :language: yaml

Running
-------

.. literalinclude:: ../../examples/01_visium_only/run.sh
   :language: bash

Expected Output
---------------

.. code-block:: text

   <out_root>/
   ├── batches/
   │   ├── batch_vis_000_x/y/d.npy      <- tokenized training batches
   │   └── states/
   │       ├── 00/model.ckpt            <- recon checkpoint
   │       └── 01/model.ckpt            <- Visium MSE checkpoint
   └── VIS_S1/
       ├── embeddings-hist.pickle or .npy  <- UNI features
       ├── pred_fullgrid_outputs.pkl    <- prediction results
       ├── argmax_map.png               <- cell-type argmax map
       └── *_intensity.png              <- per-cell-type intensity maps
