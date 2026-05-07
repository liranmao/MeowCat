Example 04b: Multiple Xenium Samples
=====================================

**Scenario:** Multiple Xenium samples from different patients (e.g., 4 LUAD samples).

**Training paradigm:** Reconstruction (Phase 0) → Xenium CE with CDAN domain adaptation (Phase 1).

CDAN aligns patient-level domains so the encoder learns patient-invariant features.

Data Setup
----------

.. code-block:: text

   <data_root>/
   ├── P24_LUAD_Xenium/
   │   ├── xenium_raw/
   │   ├── adata_cellbin_HistoSweep.h5ad
   │   ├── annotation.csv
   │   └── single_super_emb.h5ad   <- pre-prepared UNI features
   ├── P19_LUAD_Xenium/  (same layout)
   ├── P17_LUAD_Xenium/  (same layout)
   ├── P11_LUAD_Xenium/  (same layout)
   └── anno-names.txt

Run ``setup_data.sh`` first to create symlinks from pre-prepared data.

Config
------

.. literalinclude:: ../../examples/04_multi_xenium/config.yaml
   :language: yaml

Running
-------

.. literalinclude:: ../../examples/04_multi_xenium/run.sh
   :language: bash

Expected Output
---------------

.. code-block:: text

   <out_root>/batches/
   ├── batch_xen_000_x/y/d.npy   <- domain 0 (P24)
   ├── batch_xen_001_x/y/d.npy   <- domain 1 (P19)
   ├── batch_xen_002_x/y/d.npy   <- domain 2 (P17)
   ├── batch_xen_003_x/y/d.npy   <- domain 3 (P11)
   └── states/00/model.ckpt, states/01/model.ckpt
