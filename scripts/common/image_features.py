from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.io
import torch
import torchvision.transforms as transforms
from PIL import Image

from common.color_norm import macenko_normalizer
from common.ctranspath import CTransPath


FeatureList = List[Tuple[str, np.ndarray]]
SAMPLE_SIZE_MICRONS = 80.0


def get_device(device_name: str | None = None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_tile(img_np: np.ndarray, edge_mag_threshold: int = 15, edge_fraction_threshold: float = 0.5) -> bool:
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0)
    sobely = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1)
    sobelx1 = cv2.convertScaleAbs(sobelx)
    sobely1 = cv2.convertScaleAbs(sobely)
    mag = cv2.addWeighted(sobelx1, 0.5, sobely1, 0.5, 0)
    unique, counts = np.unique(mag, return_counts=True)
    low_edge_fraction = counts[np.argwhere(unique < edge_mag_threshold)].sum() / (img_np.shape[0] * img_np.shape[1])
    return low_edge_fraction <= edge_fraction_threshold


def tile_transform(
    tiles_list: List[Image.Image],
    data_mean: list[float],
    data_std: list[float],
    device: torch.device,
    tile_size: int,
) -> torch.Tensor:
    data_transform = transforms.Compose(
        [
            transforms.Resize((tile_size, tile_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_mean, std=data_std),
        ]
    )
    tiles = [data_transform(tile).unsqueeze(0).to(device) for tile in tiles_list]
    return torch.cat(tiles, dim=0)


def load_ctrans_model(weights_path: str, device: torch.device) -> torch.nn.Module:
    model = CTransPath(num_classes=0).to(device)
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model


def tiles_to_features(
    tiles_list: List[Image.Image],
    weights_path: str,
    batch_size: int,
    device: torch.device,
    tile_size: int,
) -> np.ndarray:
    if not tiles_list:
        raise ValueError("No tiles passed feature extraction after filtering.")

    model = load_ctrans_model(weights_path, device)
    tiles = tile_transform(tiles_list, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], device, tile_size)

    outputs = []
    for idx_start in range(0, tiles.shape[0], batch_size):
        idx_end = min(idx_start + batch_size, tiles.shape[0])
        with torch.no_grad():
            y = model(tiles[idx_start:idx_end])
        outputs.append(y.detach().cpu().numpy())

    return np.concatenate(outputs, axis=0)


def build_slide_sampling_metadata(
    slide_image_path: str,
    slide_name: str,
    microns_per_pixel: float,
    spacing_microns: float = 100.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if microns_per_pixel <= 0:
        raise ValueError("microns_per_pixel must be greater than 0.")

    image = skimage.io.imread(slide_image_path)
    image_height, image_width = image.shape[:2]
    spacing_pixels = spacing_microns / microns_per_pixel
    sample_size_pixels = SAMPLE_SIZE_MICRONS / microns_per_pixel
    if spacing_pixels <= 0:
        raise ValueError("Computed spacing in pixels must be greater than 0.")
    if sample_size_pixels <= 0:
        raise ValueError("Computed sample size in pixels must be greater than 0.")

    half_tile = int(round(sample_size_pixels / 2.0))
    min_x = half_tile
    max_x = image_width - half_tile
    min_y = half_tile
    max_y = image_height - half_tile

    if min_x >= max_x or min_y >= max_y:
        raise ValueError("The input image is smaller than one tile and cannot be sampled.")

    x_centers = np.arange(min_x, max_x + 1e-6, spacing_pixels)
    y_centers = np.arange(min_y, max_y + 1e-6, spacing_pixels)

    spot_rows = []
    for y_index, pixel_y in enumerate(y_centers):
        for x_index, pixel_x in enumerate(x_centers):
            spot_rows.append(
                {
                    "slide_name": slide_name,
                    "selected": 1,
                    "x": x_index,
                    "y": y_index,
                    "pixel_x": int(round(float(pixel_x))),
                    "pixel_y": int(round(float(pixel_y))),
                }
            )

    metadata_df = pd.DataFrame(
        [
            {
                "slide_name": slide_name,
                "slide_file_name": Path(slide_image_path).stem,
                "image_width": image_width,
                "image_height": image_height,
                "microns_per_pixel": microns_per_pixel,
                "spot_spacing_microns": spacing_microns,
                "spot_spacing_pixels": spacing_pixels,
                "sample_size_microns": SAMPLE_SIZE_MICRONS,
                "sample_size_pixels": sample_size_pixels,
            }
        ]
    )
    spots_df = pd.DataFrame(spot_rows)
    return metadata_df, spots_df


def _figure_size_for_image(image_width: int, image_height: int, max_long_side_inches: float = 16.0) -> tuple[float, float]:
    long_side = max(image_width, image_height)
    scale = max_long_side_inches / long_side
    width_inches = max(6.0, image_width * scale)
    height_inches = max(6.0, image_height * scale)
    return width_inches, height_inches


def save_mask_figure(
    img_mask: np.ndarray,
    processed_mask: np.ndarray,
    output_path: str,
    slide_name: str,
) -> None:
    image_height, image_width = img_mask.shape[:2]
    fig_width, fig_height = _figure_size_for_image(image_width, image_height)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width * 2, fig_height))
    axes[0].imshow(img_mask)
    axes[1].imshow(processed_mask)
    axes[0].set_title(slide_name)
    axes[0].set_axis_off()
    axes[1].set_axis_off()
    fig.tight_layout(h_pad=0.4, w_pad=0.5)
    fig.savefig(output_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def extract_spot_features(
    slide_image_path: str,
    slide_name: str,
    microns_per_pixel: float,
    ctranspath_weights: str,
    batch_size: int = 128,
    device_name: str | None = None,
    tile_size: int = 224,
    spacing_microns: float = 100.0,
) -> tuple[FeatureList, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    device = get_device(device_name)
    color_normalizer = macenko_normalizer()
    metadata, spots = build_slide_sampling_metadata(
        slide_image_path=slide_image_path,
        slide_name=slide_name,
        microns_per_pixel=microns_per_pixel,
        spacing_microns=spacing_microns,
    )
    slide_img = skimage.io.imread(slide_image_path)
    spots = spots[spots["selected"] == 1].sort_values(["x", "y"])
    sample_size_pixels = int(round(float(metadata.iloc[0]["sample_size_pixels"])))
    img_mask = np.full(slide_img.shape, 255, dtype=np.uint8)
    processed_mask = np.full(slide_img.shape, 255, dtype=np.uint8)

    tile_coords = spots[["pixel_x", "pixel_y"]].round().astype(int).values
    grid_coords = spots[["x", "y"]].values

    tiles_list: List[Image.Image] = []
    tile_names: list[str] = []
    kept_rows: list[dict[str, int | str]] = []

    for grid_coord, tile_coord in zip(grid_coords, tile_coords):
        grid_x, grid_y = grid_coord
        pixel_x, pixel_y = tile_coord
        spot_name = f"{slide_name}_{grid_x}x{grid_y}"

        start_x = pixel_x - sample_size_pixels // 2
        end_x = start_x + sample_size_pixels
        start_y = pixel_y - sample_size_pixels // 2
        end_y = start_y + sample_size_pixels

        if start_x < 0 or end_x > slide_img.shape[1] or start_y < 0 or end_y > slide_img.shape[0]:
            continue

        tile = slide_img[start_y:end_y, start_x:end_x, :]
        img_mask[start_y:end_y, start_x:end_x, :] = tile
        if evaluate_tile(tile):
            normalized_tile = Image.fromarray(color_normalizer.transform(tile))
            tiles_list.append(normalized_tile)
            tile_names.append(spot_name)
            processed_mask[start_y:end_y, start_x:end_x, :] = np.array(normalized_tile)
            kept_rows.append(
                {
                    "slide_name": slide_name,
                    "selected": 1,
                    "x": int(grid_x),
                    "y": int(grid_y),
                    "pixel_x": int(pixel_x),
                    "pixel_y": int(pixel_y),
                }
            )

    features = tiles_to_features(tiles_list, ctranspath_weights, batch_size, device, tile_size)
    line_color = np.array([0, 255, 0], dtype=np.uint8)
    img_mask[:, ::tile_size, :] = line_color
    img_mask[::tile_size, :, :] = line_color
    processed_mask[:, ::tile_size, :] = line_color
    processed_mask[::tile_size, :, :] = line_color
    processed_spots = pd.DataFrame(
        kept_rows,
        columns=["slide_name", "selected", "x", "y", "pixel_x", "pixel_y"],
    )
    return list(zip(tile_names, features)), metadata, spots, processed_spots, img_mask, processed_mask
