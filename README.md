# PFIR-SAM2

PFIR-SAM2 performs prompt-free instance segmentation of bright-field organoid
images. It combines a LoRA-adapted SAM2.1 Hiera-L image encoder with a parallel
CNN local branch, residual multi-scale fusion, dense foreground/boundary/center
cues, and cue-guided instance reconstruction with small-object rescue.

## Installation

Create the project environment:

```bash
conda env create -f environment.yml
conda activate pfir-sam2
```

Alternatively, install the Python dependencies into an existing CUDA-compatible
environment:

```bash
pip install -r requirements.txt
```

## SAM2 dependency

PFIR-SAM2 depends on the official
[facebookresearch/sam2](https://github.com/facebookresearch/sam2) package. Clone
and install that project from its official source; do not substitute a copied
SAM2 source tree from this repository.

```bash
git clone https://github.com/facebookresearch/sam2.git sam2
pip install -e sam2
```

Download the official SAM2.1 Hiera-L checkpoint according to the SAM2
instructions and place it at `weights/sam2.1_hiera_large.pt`, or override its
path on the command line.

## Dataset preparation

OrganoID was obtained from
[OSF project xmes4](https://osf.io/xmes4/). The author-distributed Train, Val,
and Test directories are used directly without repartitioning. Dataset files
are not included. See [docs/DATASETS.md](docs/DATASETS.md).

```bash
python scripts/prepare_organoid.py --data-root data/OrganoID --output outputs/organoid_dataset_manifest.csv --size-groups-output-dir outputs/organoid_size_groups
```

## Pretrained PFIR-SAM2 checkpoint

The retained OrganoID checkpoint is approximately 1.03 GiB and is distributed
as a separate release asset, not as a Git object. Place it at
`weights/PFIR-SAM2_OrganoID_best_model.pth`. The permanent download URL will be
added when the release asset is deposited.

## Inference

```bash
python scripts/infer.py --config configs/inference_organoid.yaml --checkpoint weights/PFIR-SAM2_OrganoID_best_model.pth --input data/OrganoID/Test/Images --output outputs/organoid_test
```

The command strictly loads the retained architecture and writes final
integer-label TIFF maps to `outputs/organoid_test/instances/`.

## Evaluation

```bash
python scripts/evaluate.py --gt-mask-dir data/OrganoID/Test/Masks --method PFIR-SAM2=outputs/organoid_test/instances --gt-boxes-csv outputs/organoid_size_groups/bbox_instances_test.csv --output-dir outputs/organoid_test_evaluation
```

The evaluator reports pooled and mean per-image precision/recall/F1, P@0.75,
mP@0.50:0.95, Count/Total Area/Mean Area MAPE, and size-stratified recall.
`mP@0.50:0.95` is a mean fixed-IoU-threshold precision metric and is **not
confidence-ranked COCO mAP**.

## Training

```bash
python scripts/train.py --config configs/train_organoid.yaml
```

The config reproduces the retained 100-epoch full-model protocol and selects
the best checkpoint by Val foreground Dice. It requires the official SAM2 base
checkpoint and the retained PFIR checkpoint for architecture/default provenance
validation. Test data are not used for model or parameter selection.

## Reproducing OrganoID evaluation

1. Download OrganoID from OSF project xmes4.
2. Validate the author-distributed folders with `scripts/prepare_organoid.py`.
3. Obtain the official SAM2.1 Hiera-L and retained PFIR-SAM2 checkpoints.
4. Run the inference and evaluation commands above.
5. Compare the generated CSV definitions with the manuscript values; smoke-test
   outputs must not replace the frozen manuscript results.

## OrgaSegment zero-shot evaluation

Download OrgaSegment from its official
[Zenodo record](https://doi.org/10.5281/zenodo.10278229). No OrgaSegment
training, fine-tuning, or calibration is performed.

```bash
python scripts/infer.py --config configs/orgasegment_zero_shot.yaml --checkpoint weights/PFIR-SAM2_OrganoID_best_model.pth --input data/OrgaSegment/Test/Images --output outputs/orgasegment_zero_shot_test
python scripts/evaluate.py --gt-mask-dir data/OrgaSegment/Test/Masks --method PFIR-SAM2-zero-shot=outputs/orgasegment_zero_shot_test/instances --gt-boxes-csv data/OrgaSegment/bbox_instances_test.csv --output-dir outputs/orgasegment_zero_shot_test_evaluation
```

## Repository scope

This repository contains the model, inference, reconstruction, and evaluation
code required to reproduce the core PFIR-SAM2 workflow. Publication-layout and
figure-rendering scripts are not required for running or evaluating the method
and are not included. Third-party baseline repositories are also not copied.

For baseline provenance, Cellpose-SAM denotes the official default pretrained
Cellpose-SAM model used without fine-tuning on OrganoID.

## Citation

Please cite the accompanying PFIR-SAM2 manuscript. Citation metadata will be
updated when the manuscript record is public; see `CITATION.cff`.

## License

Project code is released under the Apache License 2.0. SAM2 remains governed by
its own upstream license and distribution terms. Dataset licenses and terms are
those of their original repositories.
