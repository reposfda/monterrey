# utils/loaders.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st


def _fix_mojibake_series(s: pd.Series) -> pd.Series:
    if s.dtype != "object":
        return s

    x = s.astype("string")
    mask = x.str.contains("Ã|Â|�", na=False)

    def _fix_one(val):
        if val is None:
            return val
        try:
            return val.encode("latin1", errors="strict").decode("utf-8", errors="strict")
        except Exception:
            return val

    x.loc[mask] = x.loc[mask].apply(_fix_one)
    return x


@st.cache_data
def load_per90(path: Path) -> pd.DataFrame:
    """
    Carga CSV per90 con fallback simple + fix mojibake.
    """
    try:
        df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    except Exception:
        try:
            df = pd.read_csv(path, low_memory=False, encoding="utf-8")
        except Exception:
            df = pd.read_csv(path, low_memory=False, encoding="cp1252")


    for col in ["player_name", "team_name", "player", "team", "Jugador", "Equipo", "Flags", "Perfil"]:
        if col in df.columns:
            before = df[col].astype("string")
            df[col] = _fix_mojibake_series(df[col])
            after = df[col].astype("string")
            if (before != after).any():
                st.warning(f"Se detectó y corrigió mojibake en columna '{col}' para {path.name}")


    return df
