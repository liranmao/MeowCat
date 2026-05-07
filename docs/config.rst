Config Reference
================

All parameters live in ``config/default.yaml``. Copy and edit it for each experiment:

.. code-block:: bash

   cp config/default.yaml config/my_run.yaml

Parameter Table
---------------

.. list-table::
   :header-rows: 1
   :widths: 15 25 20 40

   * - Section
     - Key
     - Default
     - Description
   * - ``project``
     - ``name``
     - ``meowcat_run``
     - Experiment name (for logging)
   * - ``project``
     - ``data_root``
     - —
     - Root folder with per-sample subfolders
   * - ``project``
     - ``out_root``
     - —
     - Root for all outputs
   * - ``rctd``
     - ``reference_rds``
     - —
     - Path to single-cell reference RDS (Seurat v5)
   * - ``rctd``
     - ``cell_type_column``
     - ``MainType``
     - Metadata column for cell-type labels
   * - ``rctd``
     - ``group_column``
     - ``""``
     - Metadata column for group subsetting (``""`` = use all cells)
   * - ``rctd``
     - ``groups``
     - ``[]``
     - Groups of interest for subsetting
   * - ``rctd``
     - ``max_cores``
     - ``5``
     - Parallel cores for RCTD fitting
   * - ``rctd``
     - ``doublet_mode``
     - ``full``
     - RCTD doublet mode (``full``, ``doublet``, or ``multi``)
   * - ``rctd``
     - ``min_umi``
     - ``10``
     - Minimum UMI count for reference cells
   * - ``preprocess``
     - ``raw_flag``
     - ``he_raw``
     - Substring matched in raw image filename
   * - ``preprocess``
     - ``target_mpp``
     - ``0.5``
     - Target resolution (microns-per-pixel)
   * - ``preprocess``
     - ``pixel_size_raw``
     - ``null``
     - Manual MPP override (``null`` = auto-detect from image metadata)
   * - ``preprocess``
     - ``pad``
     - ``224``
     - Padding multiple for rescaled image
   * - ``preprocess``
     - ``uni_weights``
     - —
     - Path to UNI ViT-Large ``.bin`` weights
   * - ``preprocess``
     - ``fusion_mode``
     - ``single``
     - Feature fusion mode: ``single`` or ``multi``
   * - ``batches``
     - ``out_dir``
     - —
     - Where batch ``.npy`` files are written
   * - ``visium``
     - ``sample_pattern``
     - ``VIS*``
     - Glob pattern to find Visium sample dirs
   * - ``visium``
     - ``include_only``
     - ``null``
     - Sample names to include (``null`` = all matching pattern)
   * - ``visium``
     - ``exclude_set``
     - ``[]``
     - Sample names to skip
   * - ``visium``
     - ``domain_map_tsv``
     - ``null``
     - TSV mapping sample names to domain strings
   * - ``visium``
     - ``fixed_radius``
     - ``null``
     - Override spot radius (``null`` = from ``radius.txt``)
   * - ``visium``
     - ``keep_frac``
     - ``0.25``
     - Coreset fraction of spots to keep
   * - ``visium``
     - ``strategy``
     - ``stratified``
     - Coreset strategy: ``stratified`` or ``kcenter``
   * - ``xenium``
     - ``sample_pattern``
     - ``null``
     - Glob pattern for Xenium sample folders
   * - ``xenium``
     - ``anno_names_path``
     - —
     - Path to shared ``anno-names.txt``
   * - ``xenium``
     - ``dapi_pixel_size_raw``
     - ``0.2125``
     - Xenium DAPI coordinate-to-pixel conversion factor
   * - ``xenium``
     - ``keep_frac``
     - ``null``
     - Fraction of valid cells to keep
   * - ``train``
     - ``n_states``
     - ``2``
     - Independent model replicas (ensemble)
   * - ``train``
     - ``two_stage``
     - ``true``
     - Phase 0 reconstruction pretraining
   * - ``train``
     - ``epochs1``
     - ``15``
     - Reconstruction epochs
   * - ``train``
     - ``sequential_training``
     - ``true``
     - Sequential Visium → Xenium training
   * - ``train``
     - ``visium_epochs``
     - ``100``
     - Visium phase epochs
   * - ``train``
     - ``xenium_epochs``
     - ``100``
     - Xenium phase epochs
   * - ``train``
     - ``freeze_encoder_n``
     - ``2``
     - Encoder layers frozen at phase transitions
   * - ``train``
     - ``xenium_weight``
     - ``0.01``
     - Relative CE loss weight
   * - ``train``
     - ``adv_lambda``
     - ``0``
     - CDAN adversarial weight (``0`` = disabled)
   * - ``train``
     - ``monitor_metric``
     - ``val_weak_mse``
     - Metric for best-checkpoint selection
   * - ``predict``
     - ``n_states``
     - ``2``
     - Checkpoints to ensemble at inference
   * - ``predict``
     - ``tokens_per_chunk``
     - ``70000``
     - GPU chunk size (reduce if OOM)
   * - ``predict``
     - ``chunks_per_batch``
     - ``2``
     - Chunks batched per forward pass
   * - ``visualize``
     - ``n_clusters``
     - ``6``
     - KMeans clusters for latent map
   * - ``visualize``
     - ``p_lo`` / ``p_hi``
     - ``5`` / ``95``
     - Percentile clipping for intensity maps
   * - ``visualize``
     - ``cmap_json``
     - ``null``
     - Custom cell-type color map JSON
   * - ``slide``
     - ``pptx``
     - ``results.pptx``
     - Output PowerPoint filename
   * - ``inference``
     - ``model_dir``
     - ``""``
     - Path to batches dir containing trained checkpoints
   * - ``inference``
     - ``anno_names``
     - ``""``
     - Path to ``anno-names.txt`` from training

Training Paradigms
------------------

Visium only (soft labels, MSE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   train:
     two_stage: true
     epochs1: 15
     sequential_training: false
     visium_epochs: 100
     xenium_epochs: 0

Xenium only (hard labels, cross-entropy)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   train:
     two_stage: true
     epochs1: 15
     sequential_training: false
     visium_epochs: 0
     xenium_epochs: 100

Multi-resolution: Visium then Xenium (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3-phase sequential training: reconstruction pretraining → Visium MSE → Xenium CE fine-tuning.

.. code-block:: yaml

   train:
     two_stage: true
     epochs1: 15
     sequential_training: true
     visium_epochs: 100
     xenium_epochs: 100
     freeze_encoder_n: 2
     xenium_weight: 0.01

When ``adv_lambda > 0``, CDAN domain adaptation is enabled with a sigmoid ramp schedule.
