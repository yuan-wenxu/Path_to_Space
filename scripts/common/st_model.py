from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn

from common.smoothing import smooth_genes_kdtree

N_IK_FOLDS = 22
N_IL_FOLDS = 7


class MLPRegression(nn.Module):
    def __init__(self, n_inputs: int, n_hiddens: int, n_outputs: int, dropout: float = 0.2):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Linear(n_inputs, n_hiddens),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layer1 = nn.Sequential(
            nn.Linear(n_hiddens, n_outputs),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        return x


def get_device(device_name: str | None = None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_gene_names(gene_file: str) -> np.ndarray:
    if gene_file.endswith(".pkl"):
        genes = pd.read_pickle(gene_file)
        return genes["gene"].values
    if gene_file.endswith(".txt"):
        with open(gene_file) as f:
            return np.array([line.strip() for line in f if line.strip()])
    raise ValueError(f"Unsupported gene file format: {gene_file}")


def parse_spot_coordinates(names: Iterable[str]) -> pd.DataFrame:
    rows = []
    for name in names:
        try:
            slide_name, coord_text = name.rsplit("_", 1)
            x_text, y_text = coord_text.split("x", 1)
            rows.append(
                {
                    "spot_name": name,
                    "slide_name": slide_name,
                    "x": int(x_text),
                    "y": int(y_text),
                }
            )
        except ValueError:
            rows.append({"spot_name": name, "slide_name": None, "x": np.nan, "y": np.nan})
    return pd.DataFrame(rows).set_index("spot_name")


def _load_ensemble_models(
    ensemble_dir: str,
    n_genes: int,
    device: torch.device,
) -> dict[tuple[int, int], MLPRegression]:
    ensemble_path = Path(ensemble_dir)
    models: dict[tuple[int, int], MLPRegression] = {}
    missing: list[Path] = []

    for ik in range(N_IK_FOLDS):
        for il in range(N_IL_FOLDS):
            ckpt = ensemble_path / f"result_{ik}_{il}_0" / "model_trained.pth"
            if not ckpt.exists():
                missing.append(ckpt)
                continue

            model = MLPRegression(n_inputs=768, n_hiddens=768, n_outputs=n_genes)
            state_dict = torch.load(ckpt, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            models[(ik, il)] = model

    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} ensemble checkpoint(s). First missing file: {missing[0]}"
        )

    return models


def predict_st(features_list, model_dir: str, gene_file: str, device_name: str | None = None) -> pd.DataFrame:
    device = get_device(device_name)
    genes = load_gene_names(gene_file)
    feature_matrix = np.stack([feature for _name, feature in features_list]).astype(np.float32)
    x = torch.as_tensor(feature_matrix, dtype=torch.float32, device=device)
    models = _load_ensemble_models(model_dir, len(genes), device)

    preds_ik = np.zeros((N_IK_FOLDS, len(features_list), len(genes)), dtype=np.float32)
    with torch.no_grad():
        for ik in range(N_IK_FOLDS):
            preds_il = np.zeros((N_IL_FOLDS, len(features_list), len(genes)), dtype=np.float32)
            for il in range(N_IL_FOLDS):
                preds_il[il] = models[(ik, il)](x).detach().cpu().numpy()
            preds_ik[ik] = preds_il.mean(axis=0)

    pred_matrix = preds_ik.mean(axis=0)
    spot_names = [name for name, _feature in features_list]
    df_pred = pd.DataFrame(pred_matrix, columns=genes, index=spot_names)
    coord_df = parse_spot_coordinates(spot_names)
    return coord_df.join(df_pred)


def smooth_st_predictions(
    st_predictions: pd.DataFrame,
    gene_file: str,
    radius: float = 2.0,
    weights: str | float = "uniform",
) -> pd.DataFrame:
    genes = load_gene_names(gene_file)
    missing_genes = sorted(set(genes) - set(st_predictions.columns))
    if missing_genes:
        raise ValueError(f"ST prediction is missing {len(missing_genes)} required genes. Example: {missing_genes[:5]}")

    return smooth_genes_kdtree(
        slide_df=st_predictions,
        genes=genes,
        radius=radius,
        coord_cols=("x", "y"),
        weights=weights,
    )
