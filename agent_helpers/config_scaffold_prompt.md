# MeowCat Config Scaffolding Prompt

Copy the prompt below and paste it into any AI assistant (ChatGPT, Gemini, Cursor, etc.).
Fill in the `[bracketed]` fields and the AI will generate a correct `config.yaml` for your dataset.

---

```
I am using MeowCat, a CLI pipeline for cell-type annotation in H&E images.
Help me generate a config.yaml for my dataset.

My dataset:
- Data modality: [Visium only / Xenium only / Visium + Xenium]
- Number of samples: [single sample / N samples from different patients]
- data_root: [/absolute/path/to/samples]
- out_root:  [/absolute/path/to/outputs]
- UNI weights (.bin file): [/absolute/path/to/pytorch_model.bin]

Visium (if applicable):
- Sample folder name pattern (glob): [e.g. VIS*, P*]
- Single-cell reference RDS (Seurat v5): [/path/to/reference.rds]
- Cell-type label column in reference: [e.g. MainType]
- Group column for per-group reference subsetting: [e.g. Group_ID, or leave blank]
- Groups to include: [e.g. LUAD, or leave blank to use all cells]

Xenium (if applicable):
- Sample folder name pattern (glob): [e.g. XEN*, *_Xenium]
- anno-names.txt path: [/path/to/anno-names.txt]
- Cell-type mapping JSON (fine→coarse): [/path/to/mapping.json, or leave blank]

Training paradigm rules (apply automatically):
- Visium only, 1 sample:  sequential_training=false, visium_epochs=100, xenium_epochs=0,  adv_lambda=0
- Visium only, multi:     sequential_training=false, visium_epochs=100, xenium_epochs=0,  adv_lambda=0.005
- Xenium only, 1 sample:  sequential_training=false, visium_epochs=0,   xenium_epochs=100, adv_lambda=0
- Xenium only, multi:     sequential_training=false, visium_epochs=0,   xenium_epochs=100, adv_lambda=0.005
- Both, 1 pair:           sequential_training=true,  visium_epochs=100, xenium_epochs=50,  adv_lambda=0
- Both, multi:            sequential_training=true,  visium_epochs=100, xenium_epochs=100, adv_lambda=0.005
Always set: two_stage=true, epochs1=15, n_states=2, freeze_encoder_n=2.
Use monitor_metric=val_weak_mse for Visium; val_loss for Xenium-only.

Please generate the complete config.yaml (all sections, with comments).
Then list the exact meowcat commands I need to run in order.
```
