from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


class SlideDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, index):
        return torch.Tensor(self.features[index][1]).float().unsqueeze(0)

    def __len__(self):
        return len(self.features)


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
        return torch.mean(x, dim=0)


def get_device(device_name: str | None = None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_gene_names(gene_file: str) -> np.ndarray:
    genes = pd.read_pickle(gene_file)
    return genes["gene"].values


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


def predict_st(features_list, model_path: str, gene_file: str, device_name: str | None = None) -> pd.DataFrame:
    device = get_device(device_name)
    genes = load_gene_names(gene_file)
    dataset = SlideDataset(features_list)
    model = MLPRegression(n_inputs=768, n_hiddens=768, n_outputs=len(genes))
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    preds = []
    with torch.no_grad():
        for x in dataset:
            y = model(x.float().to(device))
            if y.dim() == 1:
                y = y.unsqueeze(0)
            preds.append(y.detach().cpu().numpy())

    pred_matrix = np.concatenate(preds, axis=0)
    spot_names = [name for name, _feature in features_list]
    df_pred = pd.DataFrame(pred_matrix, columns=genes, index=spot_names)
    coord_df = parse_spot_coordinates(spot_names)
    return coord_df.join(df_pred)
