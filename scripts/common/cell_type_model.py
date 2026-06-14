import joblib
import numpy as np
import pandas as pd


def predict_cell_types(
    st_predictions: pd.DataFrame,
    model_path: str,
    feature_list_path: str,
    cell_types: list[str] | None = None,
) -> pd.DataFrame:
    if cell_types is None:
        cell_types = ["TILs", "Stromal", "Epithelial"]

    with open(feature_list_path, "r", encoding="utf-8") as handle:
        best_features = [line.strip() for line in handle if line.strip()]

    missing_features = sorted(set(best_features) - set(st_predictions.columns))
    if missing_features:
        raise ValueError(f"ST prediction is missing {len(missing_features)} required genes. Example: {missing_features[:5]}")

    model = joblib.load(model_path)
    x_subset = st_predictions.loc[:, best_features]
    y_pred = model.predict(x_subset)

    base_cols = [col for col in ["slide_name", "x", "y"] if col in st_predictions.columns]
    result = st_predictions.loc[:, base_cols].copy()
    result[cell_types] = pd.DataFrame(y_pred, index=st_predictions.index, columns=cell_types)
    return result
