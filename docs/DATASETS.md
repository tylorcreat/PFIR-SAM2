# Datasets

## OrganoIDNetData

PFIR-SAM2 uses OrganoIDNetData from Kulkarni et al., *Scientific Data* (2024),
available from [Zenodo record 10643410](https://doi.org/10.5281/zenodo.10643410).
Download the dataset from the source repository. Do not copy its images or
annotations into this Git repository.

The author-distributed directories are used directly without repartitioning:

```text
data/OrganoID/  # legacy local alias for OrganoIDNetData
  Train/
    Images/
    Masks/
  Val/
    Images/
    Masks/
  Test/
    Images/
    Masks/
```

Validate the download and generate the local instance-size table with:

```bash
python scripts/prepare_organoid.py --data-root data/OrganoID --output outputs/organoid_dataset_manifest.csv --size-groups-output-dir outputs/organoid_size_groups
```

The retained size-stratification protocol excludes Training instances smaller
than 10 pixels before deriving the 1/3 and 2/3 area quantiles. The resulting
fixed OrganoIDNetData cutoffs are 201 and 457 pixels and are reused unchanged
for Validation and Test.

## OrgaSegment

OrgaSegment is described by Lefferts et al., *Communications Biology* (2024),
[doi:10.1038/s42003-024-05966-4](https://doi.org/10.1038/s42003-024-05966-4).
The segmented microscopy dataset is available from
[Zenodo record 10278229](https://doi.org/10.5281/zenodo.10278229).

The PFIR-SAM2 OrgaSegment experiment is zero-shot: there is no OrgaSegment
training, fine-tuning, or calibration. Download the data from its official
record and provide the desired split paths to `scripts/infer.py` and
`scripts/evaluate.py`. Raw OrgaSegment data are not redistributed here.
