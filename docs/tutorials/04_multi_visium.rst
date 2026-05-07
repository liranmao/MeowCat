Example 04a: Multiple Visium Samples
=====================================

**Scenario:** Multiple Visium samples from different patients.

**Training paradigm:** Reconstruction (Phase 0) → Visium MSE with CDAN domain adaptation (Phase 1).

CDAN aligns patient-level domains so the encoder learns patient-invariant features.

Data Setup
----------

.. code-block:: text

   <data_root>/
   ├── P001/   <- Visium patient 1 (embeddings-hist.pickle or .npy, anno-names.txt, mask/)
   ├── P002/   <- Visium patient 2
   └── ...

Run ``setup_data.sh`` first to symlink pre-prepared data.

Config
------

.. literalinclude:: ../../examples/04_multi_visium/config.yaml
   :language: yaml

Running
-------

.. literalinclude:: ../../examples/04_multi_visium/run.sh
   :language: bash

Expected Output
---------------

.. code-block:: text

   <out_root>/batches/
   ├── batch_vis_000_x/y/d.npy   <- domain 0 (P001)
   ├── batch_vis_001_x/y/d.npy   <- domain 1 (P002)
   └── states/00/model.ckpt, states/01/model.ckpt
