"""Compatibility split shim for the Method 3 logistic-regression study."""

from __future__ import annotations

import numpy as np
import pandas as pd

from splitting import split_data as root_split_data


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Delegate to the repository-wide split implementation."""
    return root_split_data(
        y=y,
        df=df,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )
