# utils/scoring.py
from __future__ import annotations

import tempfile
from pathlib import Path
import pandas as pd
import streamlit as st

from utils.loaders import load_per90

from position_scoring_golero import run_goalkeeper_scoring
from position_scoring_defensor_central import run_cb_scoring
from position_scoring_delantero import run_delantero_scoring
from position_scoring_extremos import run_extremo_scoring
from position_scoring_interior import run_interior_scoring
from position_scoring_lateral import run_position_scoring
from position_scoring_volante import run_volante_scoring


@st.cache_data(show_spinner=False)
def compute_scoring(
    per90_path: str,
    position_key: str,
    min_minutes: int,
    min_matches: int,
    selected_teams: list[str],
) -> pd.DataFrame:
    df = load_per90(Path(per90_path))

    # filtro equipos (antes del scoring)
    if selected_teams and "teams" in df.columns:
        df = df[df["teams"].astype(str).isin([str(t) for t in selected_teams])].copy()

    # temp csv para reutilizar scripts run_*
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    df.to_csv(tmp_path, index=False, encoding="utf-8")

    out_tmp = (Path("outputs") / "_tmp_scoring.csv")
    out_tmp.parent.mkdir(parents=True, exist_ok=True)

    if position_key == "Golero":
        return run_goalkeeper_scoring(
            per90_csv=tmp_path, 
            out_csv=out_tmp,
            position_group="Golero",
            min_minutes=min_minutes,
            min_matches=min_matches,
            flag_q=0.75,
        )


    if position_key == "Zaguero":
        return run_cb_scoring(tmp_path, out_tmp, "Zaguero", min_minutes, min_matches, flag_q=0.75)

    if position_key == "Lateral":
        return run_position_scoring(
            per90_csv=tmp_path,
            out_csv=out_tmp,
            position_group="Lateral",
            min_minutes=min_minutes,
            min_matches=min_matches,
            flag_q=0.75,
            def_exec_w=0.60,
            def_obv_w=0.40,
        )

    if position_key == "Volante":
        return run_volante_scoring(tmp_path, out_tmp, "Volante", min_minutes, min_matches, flag_q=0.75)

    if position_key == "Interior/Mediapunta":
        return run_interior_scoring(tmp_path, out_tmp, "Interior/Mediapunta", min_minutes, min_matches, flag_q=0.75)

    if position_key == "Extremo":
        return run_extremo_scoring(tmp_path, out_tmp, "Extremo", min_minutes, min_matches, flag_q=0.75)

    if position_key == "Delantero":
        return run_delantero_scoring(tmp_path, out_tmp, "Delantero", min_minutes, min_matches, flag_q=0.75)

    raise ValueError(f"Posición no soportada: {position_key}")
