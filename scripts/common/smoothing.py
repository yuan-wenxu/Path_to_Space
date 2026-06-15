from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree


def smooth_genes_kdtree(
    slide_df: pd.DataFrame,
    genes: Sequence[str],
    radius: float = 2.0,
    coord_cols: tuple[str, str] = ("x", "y"),
    weights: str | float = "uniform",
) -> pd.DataFrame:
    coords = slide_df.loc[:, list(coord_cols)].to_numpy(dtype=float)
    gene_data = slide_df.loc[:, list(genes)].to_numpy()

    tree = KDTree(coords)
    indices = tree.query_radius(coords, r=radius)

    smoothed = np.zeros_like(gene_data)
    for idx, neighbors in enumerate(indices):
        if len(neighbors) == 0:
            smoothed[idx] = gene_data[idx]
            continue

        mean_values = gene_data[neighbors].mean(axis=0)
        if weights != "uniform":
            weight = float(weights)
            neighbor_count = len(neighbors)
            mean_values = (
                neighbor_count * mean_values + (weight - 1) * gene_data[idx]
            ) / (neighbor_count + weight - 1)
        smoothed[idx] = mean_values

    out = slide_df.copy()
    out.loc[:, list(genes)] = smoothed
    return out
