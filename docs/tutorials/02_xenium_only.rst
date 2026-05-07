Example 02: Xenium Only
=======================

**Scenario:** Single Xenium sample with hard cell-type labels. No Visium data; no RCTD needed.

**Training paradigm:** Reconstruction pretraining (Phase 0) → Xenium cross-entropy (Phase 1).

Data Setup
----------

.. code-block:: text

   <data_root>/XEN_P11_LUAD/
   ├── he_raw.tif                     <- raw H&E image
   ├── xenium_raw/                    <- 10x Xenium output
   │   ├── cell_feature_matrix.h5
   │   └── cells.parquet
   ├── adata_cellbin_HistoSweep.h5ad  <- cell-bin location mapping (external alignment)
   └── annotation.csv                 <- cell-type annotations (cell_id, cell_state)

   # also required (shared across samples):
   anno-names.txt                     <- cell-type names list

Config
------

.. literalinclude:: ../../examples/02_xenium_only/config.yaml
   :language: yaml

Running
-------

.. literalinclude:: ../../examples/02_xenium_only/run.sh
   :language: bash

Expected Output
---------------

.. code-block:: text

   <out_root>/
   ├── batches/
   │   ├── batch_xen_000_x/y/d.npy
   │   └── states/00/model.ckpt, states/01/model.ckpt
   └── XEN_P11_LUAD/
       ├── Xenium_adata_cellbin_analysis_qv20.h5ad
       ├── pred_fullgrid_outputs.pkl
       └── argmax_map.png
