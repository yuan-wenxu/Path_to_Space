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
2. Sample spot centers on a regular `100` micron grid.
3. Generate slide-level `metadata` and spot-level `x`, `y`, `pixel_x`, `pixel_y`.
4. Crop a fixed `80` micron square at each sampled spot.
5. Resize each crop to `tile_size x tile_size`.
6. Save a combined preview image with `img_mask` and `processed_mask`.
7. Normalize stain and extract CTransPath features.
8. Run one trained ST prediction model to predict spatial transcriptomics.
9. Optionally run one trained cell type model to predict `TILs`, `Stromal`, and `Epithelial`.

## Required input

- One slide image file
- The image resolution as `--microns-per-pixel`

The script automatically builds sampling metadata from that image:

- `slide_name`: parsed automatically from the input image filename
- `x`, `y`: grid indices
- `pixel_x`, `pixel_y`: pixel-center positions on the image

Neighboring spot centers are spaced by `100` microns by default, converted to pixels using:

`spot_spacing_pixels = 100 / microns_per_pixel`

Each sampled crop covers a fixed `80` microns, converted to pixels using:

`sample_size_pixels = 80 / microns_per_pixel`

After cropping, the image patch is resized to `--tile-size` before feature extraction.

The preprocessing step also saves one PNG preview similar to the original Path2Space script:

- left panel: `img_mask`
- right panel: `processed_mask`

This preview is saved as PNG with `dpi=300`.

## Required model files

- `ctranspath_feature_extractor.pth`
- `st_prediction_model.pth`
- `st_gene_list.pkl`

Optional cell type files:

- `cell_type_prediction_model.joblib`
- `cell_type_feature_list.txt`

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
  --ctranspath-weights /path/to/ctranspath_feature_extractor.pth \
  --st-model /path/to/st_prediction_model.pth \
  --gene-file /path/to/st_gene_list.pkl \
  --output-dir /path/to/output
```

Run with optional cell type prediction:

```bash
python scripts/run_inference.py \
  --slide-image /path/to/slide.tif \
  --microns-per-pixel 0.5 \
  --ctranspath-weights /path/to/ctranspath_feature_extractor.pth \
  --st-model /path/to/st_prediction_model.pth \
  --gene-file /path/to/st_gene_list.pkl \
  --cell-model /path/to/cell_type_prediction_model.joblib \
  --cell-feature-list /path/to/cell_type_feature_list.txt \
  --output-dir /path/to/output
```

## Outputs

- `<slide_name>_metadata.csv`
- `<slide_name>_spots.csv`
- `<slide_name>_processed_spots.csv`
- `<slide_name>_mask.png`
- `<slide_name>_st_predictions.pkl`
- `<slide_name>_st_predictions.csv`
- optional `<slide_name>_cell_type_predictions.pkl`
- optional `<slide_name>_cell_type_predictions.csv`
