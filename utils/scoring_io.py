from __future__ import annotations
from pathlib import Path
import pandas as pd

def read_input(per90_csv: Path | None = None, df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is not None:
        return df.copy()
    if per90_csv is None:
        raise ValueError("Necesito df o per90_csv")
    return pd.read_csv(per90_csv, low_memory=False, encoding="utf-8-sig")
