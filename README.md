# Minimal Path_to_Space Inference

This folder contains a stripped-down inference-only version of the original Path2Space code.

The `scripts/` folder now has a single entrypoint, `run_inference.py`. Helper modules live in `scripts/common/`, using normal package imports.

## License

This repository is distributed under the Apache License 2.0. See [LICENSE](LICENSE).

It is a simplified inference-focused adaptation related to the official
`path2space-companion` project:

- upstream repository: `https://github.com/eldadshulman/path2space-companion`
- upstream license: Apache License 2.0

See [NOTICE](NOTICE) for attribution details.

## Included pipeline

1. Read one H&E slide image.
2. Sample tiles from the top-left corner in a seamless, non-overlapping grid.
3. Generate slide-level `metadata` and spot-level `x`, `y`, `pixel_x`, `pixel_y`.
4. Crop a fixed `80` micron square at each sampled spot.
5. Resize each crop to `tile_size x tile_size`.
6. Save a combined preview image with `img_mask` and `processed_mask`.
7. Normalize stain and extract CTransPath features.
8. Run one trained ST prediction model to predict spatial transcriptomics.
9. Smooth the ST predictions with a KDTree neighborhood mean in grid coordinates.
10. Optionally run one trained cell type model to predict `TILs`, `Stromal`, and `Epithelial`.

## Required input

- One slide image file
- The image resolution as `--microns-per-pixel`

The script automatically builds sampling metadata from that image:

- `slide_name`: parsed automatically from the input image filename
- `x`, `y`: grid indices
- `pixel_x`, `pixel_y`: pixel-center positions of each sampled tile on the image

Sampling starts at the top-left corner of the image and advances by one full sample width each time, so neighboring tiles have no gaps or overlap.

Each sampled crop covers a fixed `80` microns, converted to pixels using:

`sample_size_pixels = 80 / microns_per_pixel`

After cropping, the image patch is resized to `--tile-size` before feature extraction.

The preprocessing step also saves one PNG preview similar to the original Path2Space script:

- left panel: `img_mask`
- right panel: `processed_mask`

This preview is saved as PNG with `dpi=300`.

## Required model files

- `ctranspath.pth`
- `mlp_ensemble.tar.gz` (extract it first, then pass the extracted `mlp_ensemble/` directory to `--st-model-dir`)
- `genes.txt` or `gene_file.pkl`

Optional cell type files:

- `cell_type_prediction/model_fold_0.joblib`
- `cell_type_prediction/best_features_fold_0.txt`

Model sources used in this repo setup:

- ST inference weights (`ctranspath.pth`, `mlp_ensemble.tar.gz`, `genes.txt`) are downloaded from Zenodo record `20174301`: https://zenodo.org/records/20174301
- The cell type prediction files in this setup are documented as downloaded from Zenodo record `20171390`: https://zenodo.org/records/20171390

Limitations of cell type prediction:

- The cell type prediction in this repository is limited by model availability.
- The original release does not provide a full set of cell type models for all datasets or scenarios; it only provides one model set associated with a single dataset.
- As a result, the optional cell type prediction here should be treated as a dataset-specific reference output rather than a universally reliable cell type annotation model.
- Predictions for `TILs`, `Stromal`, and `Epithelial` may not generalize well to slides from other datasets, cohorts, or experimental settings.

Cell type prediction is skipped unless both files are provided explicitly:

- `--cell-model`
- `--cell-feature-list`

## Run

Using Pixi:

```bash
pixi install
pixi run run-inference
```

Direct command:

```bash
python scripts/run_inference.py \
  --slide-image /path/to/slide.tif \
  --microns-per-pixel 0.5 \
  --ctranspath-weights /path/to/ctranspath.pth \
  --st-model-dir /path/to/mlp_ensemble \
  --gene-file /path/to/genes.txt \
  --smooth-radius 2.0 \
  --output-dir /path/to/output
```

Run with optional cell type prediction:

```bash
python scripts/run_inference.py \
  --slide-image /path/to/slide.tif \
  --microns-per-pixel 0.5 \
  --ctranspath-weights /path/to/ctranspath.pth \
  --st-model-dir /path/to/mlp_ensemble \
  --gene-file /path/to/genes.txt \
  --smooth-radius 2.0 \
  --cell-model /path/to/cell_type_prediction/model_fold_0.joblib \
  --cell-feature-list /path/to/cell_type_prediction/best_features_fold_0.txt \
  --output-dir /path/to/output
```

## Outputs

- `<slide_name>_metadata.csv`
- `<slide_name>_spots.csv`
- `<slide_name>_processed_spots.csv`
- `<slide_name>_mask.png`
- `<slide_name>_st_predictions.pkl`
- `<slide_name>_st_predictions.csv`
- `<slide_name>_st_predictions_smoothed.pkl`
- `<slide_name>_st_predictions_smoothed.csv`
- optional `<slide_name>_cell_type_predictions.pkl`
- optional `<slide_name>_cell_type_predictions.csv`
