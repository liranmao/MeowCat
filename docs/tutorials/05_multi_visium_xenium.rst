Example 05: Multi-Sample Visium + Xenium
=========================================

**Scenario:** Multiple Visium + multiple Xenium samples (full multi-resolution setup).

**Training paradigm:** Reconstruction (Phase 0) → Visium MSE + CDAN (Phase 1) → Xenium CE (Phase 2).

This is the recommended paradigm for multi-patient datasets with both data modalities.

Data Setup
----------

.. code-block:: text

   <data_root>/
   ├── P*/                   <- Visium samples (symlinked from 04_multi_visium)
   └── P*_LUAD_Xenium/       <- Xenium samples (symlinked from 04_multi_xenium)

   <batches_dir>/
   └── batch_vis_*_x/y/d.npy <- Visium batches (copied by setup_data.sh from 04_multi_visium)

Run ``setup_data.sh`` first to set up the directory structure.

Config
------

.. literalinclude:: ../../examples/05_multi_visium_xenium/config.yaml
   :language: yaml

Running
-------

.. literalinclude:: ../../examples/05_multi_visium_xenium/run.sh
   :language: bash

Expected Output
---------------

.. code-block:: text

   <out_root>/batches/
   ├── batch_vis_*_x/y/d.npy   <- Visium domains
   ├── batch_xen_*_x/y/d.npy   <- Xenium domains (domain IDs continue after Visium)
   └── states/00/model.ckpt, states/01/model.ckpt
