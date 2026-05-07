Examples
========

Seven ready-to-run examples covering common training scenarios.
Each has a ``config.yaml`` and a ``run.sh`` in the ``examples/`` folder of the repository.

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Example
     - Training data
     - Training paradigm
   * - :doc:`tutorials/01_visium_only`
     - 1 Visium sample
     - Recon → Visium MSE
   * - :doc:`tutorials/02_xenium_only`
     - 1 Xenium sample
     - Recon → Xenium CE
   * - :doc:`tutorials/03_visium_xenium_single`
     - 1 Visium + 1 Xenium
     - Recon → Visium → Xenium (3-phase)
   * - :doc:`tutorials/04_multi_visium`
     - Multiple Visium samples
     - Multi-sample Visium MSE with CDAN
   * - :doc:`tutorials/04_multi_xenium`
     - Multiple Xenium samples
     - Multi-sample Xenium CE with CDAN
   * - :doc:`tutorials/05_multi_visium_xenium`
     - 2 Visium + 2 Xenium
     - Multi-sample 3-phase with CDAN
   * - :doc:`tutorials/06_predict_new_sample`
     - New H&E images
     - Inference using trained model

.. toctree::
   :maxdepth: 1
   :hidden:

   tutorials/01_visium_only
   tutorials/02_xenium_only
   tutorials/03_visium_xenium_single
   tutorials/04_multi_visium
   tutorials/04_multi_xenium
   tutorials/05_multi_visium_xenium
   tutorials/06_predict_new_sample
