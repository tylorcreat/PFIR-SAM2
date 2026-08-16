# Examples

Datasets and microscopy images are intentionally not bundled. After downloading
OrganoID, use one local image directory for a smoke run:

```bash
python scripts/infer.py --config configs/inference_organoid.yaml --checkpoint weights/PFIR-SAM2_OrganoID_best_model.pth --input data/OrganoID/Test/Images --output outputs/organoid_test
```

The final instance maps are written as integer-label TIFF files under
`outputs/organoid_test/instances/`.
