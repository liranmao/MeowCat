Installation
============

Prerequisites
-------------

- `conda <https://docs.conda.io/en/latest/>`_ (Miniconda or Anaconda)
- GPU recommended for preprocessing and training (not required for small datasets)

Clone and Install
-----------------

.. code-block:: bash

   git clone https://github.com/liranmao/MeowCat.git
   cd MeowCat
   conda env create -f env/MeowCat_env.yml
   conda activate he_anno
   pip install -e .
   meowcat --help

All pipeline steps run inside the single ``he_anno`` environment — no environment switching is needed.

UNI Model Weights
-----------------

MeowCat uses the `UNI ViT-Large <https://huggingface.co/MahmoodLab/UNI>`_ foundation model for
histology patch embeddings. You must download the pretrained weights separately:

1. Request access on HuggingFace: https://huggingface.co/MahmoodLab/UNI
2. Download ``pytorch_model.bin`` (or ``model.safetensors``)
3. Set the path in your config YAML:

.. code-block:: yaml

   preprocess:
     uni_weights: /path/to/uni_weights.bin

Quick Verification
------------------

.. code-block:: bash

   # Verify the CLI is installed
   conda activate he_anno
   meowcat --help

   # Dry-run a config to check paths without executing
   meowcat run-all --config config/my_run.yaml --dry-run
