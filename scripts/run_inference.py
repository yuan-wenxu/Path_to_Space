#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from common.cell_type_model import predict_cell_types
from common.image_features import extract_spot_features, save_mask_figure
from common.st_model import predict_st


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal Path2Space inference pipeline.")
    parser.add_argument("--slide-image",
        required=True,
        help="Path to a single slide image file"
    )
    parser.add_argument(
        "--microns-per-pixel",
        type=float,
        required=True,
        help="Physical size of one pixel in microns, e.g. 0.5",
    )
    parser.add_argument(
        "--ctranspath-weights",
        required=True,
        help="Path to the CTransPath feature extractor weights, e.g. ctranspath_feature_extractor.pth",
    )
    parser.add_argument(
        "--st-model",
        required=True,
        help="Path to the spatial transcriptomics prediction model, e.g. st_prediction_model.pth",
    )
    parser.add_argument(
        "--gene-file",
        required=True,
        help="Path to the spatial transcriptomics gene list, e.g. st_gene_list.pkl",
    )
    parser.add_argument("--output-dir", required=True, help="Directory where predictions will be saved")
    parser.add_argument("--batch-size", type=int, default=128, help="Feature extraction batch size")
    parser.add_argument("--tile-size", type=int, default=224, help="Tile size in pixels")
    parser.add_argument("--spot-spacing-microns", type=float, default=100.0, help="Sampling interval in microns")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda, cuda:0, or cpu")
    parser.add_argument(
        "--cell-model",
        default=None,
        help="Optional cell type prediction model path, e.g. cell_type_prediction_model.joblib",
    )
    parser.add_argument(
        "--cell-feature-list",
        default=None,
        help="Optional explicit cell type feature list path, e.g. cell_type_feature_list.txt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    slide_name = Path(args.slide_image).stem

    features_list, slide_metadata, spots_metadata, processed_spots, img_mask, processed_mask = extract_spot_features(
        slide_image_path=args.slide_image,
        slide_name=slide_name,
        microns_per_pixel=args.microns_per_pixel,
        ctranspath_weights=args.ctranspath_weights,
        batch_size=args.batch_size,
        device_name=args.device,
        tile_size=args.tile_size,
        spacing_microns=args.spot_spacing_microns,
    )

    slide_metadata.to_csv(os.path.join(args.output_dir, f"{slide_name}_metadata.csv"), index=False)
    spots_metadata.to_csv(os.path.join(args.output_dir, f"{slide_name}_spots.csv"), index=False)
    processed_spots.to_csv(os.path.join(args.output_dir, f"{slide_name}_processed_spots.csv"), index=False)
    save_mask_figure(
        img_mask=img_mask,
        processed_mask=processed_mask,
        output_path=os.path.join(args.output_dir, f"{slide_name}_mask.png"),
        slide_name=slide_name,
    )

    st_df = predict_st(
        features_list=features_list,
        model_path=args.st_model,
        gene_file=args.gene_file,
        device_name=args.device,
    )
    st_path = os.path.join(args.output_dir, f"{slide_name}_st_predictions.pkl")
    st_df.to_pickle(st_path)
    st_df.to_csv(os.path.join(args.output_dir, f"{slide_name}_st_predictions.csv"))
    print(f"Saved ST predictions to {st_path}")

    if args.cell_model and args.cell_feature_list:
        cell_df = predict_cell_types(st_df, args.cell_model, args.cell_feature_list)
        cell_path = os.path.join(args.output_dir, f"{slide_name}_cell_type_predictions.pkl")
        cell_df.to_pickle(cell_path)
        cell_df.to_csv(os.path.join(args.output_dir, f"{slide_name}_cell_type_predictions.csv"))
        print(f"Saved cell type predictions to {cell_path}")
    elif args.cell_model and not args.cell_feature_list:
        print("Skipped cell type prediction because --cell-feature-list was not provided.")


if __name__ == "__main__":
    main()
