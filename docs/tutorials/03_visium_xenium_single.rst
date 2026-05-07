Example 03: Visium + Xenium (Single Pair)
==========================================

**Scenario:** One Visium sample and one Xenium sample from the same patient/tissue.

**Training paradigm:** Reconstruction (Phase 0) → Visium MSE (Phase 1) → Xenium CE fine-tuning (Phase 2).

The Xenium phase fine-tunes the encoder trained on Visium, transferring spot-level spatial knowledge to single-cell resolution.

Data Setup
----------

.. code-block:: text

   <data_root>/
   ├── VIS_P11_LUAD/
   │   ├── he_raw.tif
   │   ├── filtered_feature_bc_matrix/
   │   └── spatial/
   └── XEN_P11_LUAD/
       ├── he_raw.tif
       ├── xenium_raw/
       ├── adata_cellbin_HistoSweep.h5ad
       └── annotation.csv

Config
------

.. literalinclude:: ../../examples/03_visium_xenium_single/config.yaml
   :language: yaml

Running
-------

.. literalinclude:: ../../examples/03_visium_xenium_single/run.sh
   :language: bash

Expected Output
---------------

.. code-block:: text

   <out_root>/
   ├── batches/
   │   ├── batch_vis_000_x/y/d.npy
   │   ├── batch_xen_000_x/y/d.npy
   │   └── states/00/model.ckpt, states/01/model.ckpt
   ├── VIS_P11_LUAD/
   │   └── pred_fullgrid_outputs.pkl, argmax_map.png
   └── XEN_P11_LUAD/
       └── pred_fullgrid_outputs.pkl, argmax_map.png
