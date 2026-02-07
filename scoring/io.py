# scoring/io.py
# -*- coding: utf-8 -*-
"""
I/O utilities for scoring module.
Moved from utils/scoring_io.py
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def read_input(per90_csv: Path | None = None, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Read input data from CSV or DataFrame.
    
    Parameters
    ----------
    per90_csv : Path, optional
        Path to per90 CSV file
    df : pd.DataFrame, optional
        DataFrame with per90 data
    
    Returns
    -------
    pd.DataFrame
        Copy of input data
    
    Raises
    ------
    ValueError
        If neither per90_csv nor df is provided
    
    Examples
    --------
    >>> df = read_input(df=my_dataframe)
    >>> df = read_input(per90_csv=Path("data.csv"))
    """
    if df is not None:
        return df.copy()
    if per90_csv is None:
        raise ValueError("Necesito df o per90_csv")
    return pd.read_csv(per90_csv, low_memory=False, encoding="utf-8-sig")
