"""Prepare photometric and position features using training-set statistics."""

import numpy as np
import pandas as pd


def prepare_features(train, test):
    train = train.copy()
    test = test.copy()

    # Preserve the original range checks; the r band was used without imputation.
    ranges = {"redshift": (0, 7), **dict.fromkeys(("u", "g", "i", "z"), (-10, 30))}
    for column, (lower, upper) in ranges.items():
        valid = train[column].between(lower, upper)
        if not valid.any():
            raise ValueError(f"No valid training values for {column} in [{lower}, {upper}].")
        mean = train.loc[valid, column].mean()
        for frame in (train, test):
            frame[column] = frame[column].where(frame[column].between(lower, upper), mean)

    minimum = train["redshift"].min()
    span = train["redshift"].max() - minimum
    for frame in (train, test):
        frame["redshift"] = (frame["redshift"] - minimum) / span if span else 0.0

    for column in ("alpha", "delta"):
        _, edges = pd.cut(train[column], bins=12, retbins=True)
        # A held-out position beyond the training range belongs to the outermost bin.
        edges[0], edges[-1] = -np.inf, np.inf
        for frame in (train, test):
            frame[f"{column}_cut"] = pd.cut(frame[column], bins=edges, labels=False)

    for frame in (train, test):
        frame["coordinate_norm"] = np.hypot(frame["alpha"], frame["delta"])
        for left, right in (("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")):
            frame[f"{left}-{right}"] = frame[left] - frame[right]

    return train, test
