"""Read the raw SDSS17 catalog exported by the Kaggle dataset."""

import numpy as np
import pandas as pd


# The original plots use this numeric class order.
CLASS_NAMES = ("GALAXY", "STAR", "QSO")
MEASUREMENTS = {"alpha", "delta", "u", "g", "r", "i", "z", "redshift"}


def load_catalog(path):
    catalog = pd.read_csv(path)
    missing = sorted((MEASUREMENTS | {"class"}) - set(catalog.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    labels = catalog.pop("class").astype("string").str.strip().str.upper()
    if labels.isna().any():
        raise ValueError("The class column contains missing labels.")
    unknown = sorted(set(labels) - set(CLASS_NAMES))
    if unknown:
        raise ValueError("Unrecognized classes: " + ", ".join(unknown))

    # Keep the survey metadata used in the original experiment.
    features = catalog.apply(pd.to_numeric, errors="raise")
    if features.empty:
        raise ValueError("The catalog contains no observations.")
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Feature columns must contain finite numeric values.")

    return features, labels.map(dict(zip(CLASS_NAMES, range(len(CLASS_NAMES)))))
