# -*- coding: utf-8 -*-
"""
Scoring – Lateral (DataFrame-first)

- Función principal: score_lateral_df(per90_df, ...)
- Wrapper legacy: run_position_scoring(per90_csv, out_csv, ...) para correr standalone

Notas:
- Calcula percentiles 0-100 por métrica (invertidas donde aplica)
- Score por categoría = wavg de percentiles
- Score_Defensivo = combinación Exec/OBV con renormalización (sin "0 fantasma")
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from positions_config import normalize_group, sb_positions_for


# =========================
# HELPERS
# =========================
def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def pct_rank_0_100(s: pd.Series) -> pd.Series:
    """
    Percentil 0-100: mayor = mejor.
    """
    x = s.copy()
    m = x.notna()
    out = pd.Series(np.nan, index=x.index, dtype="float64")
    out.loc[m] = x.loc[m].rank(pct=True, method="average") * 100.0
    return out


def wavg(df: pd.DataFrame, cols_weights):
    """
    Promedio ponderado ignorando NaN.
    cols_weights: [(col, w), ...]
    """
    cols = [c for c, _ in cols_weights if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)

    w = np.array([w for c, w in cols_weights if c in df.columns], dtype="float64")
    if w.sum() == 0:
        return pd.Series(np.nan, index=df.index)

    w = w / w.sum()
    mat = np.vstack([df[c].to_numpy(dtype="float64") for c in cols]).T
    num = np.nansum(mat * w, axis=1)
    den = np.nansum((~np.isnan(mat)) * w, axis=1)
    return pd.Series(np.where(den > 0, num / den, np.nan), index=df.index)


def filter_by_position_group(df: pd.DataFrame, group: str) -> pd.DataFrame:
    """
    Filtra por primary_position usando positions_config.sb_positions_for(group)
    """
    group = normalize_group(group)
    valid_positions = sb_positions_for(group)

    if "primary_position" not in df.columns:
        raise KeyError(
            "No existe la columna 'primary_position' en el DataFrame. "
            "Asegurate de que el per90 la incluya (StatsBomb primary position)."
        )

    return df[df["primary_position"].isin(valid_positions)].copy()


# =========================
# MÉTRICAS / CATEGORÍAS (LATERAL)
# =========================
DEPTH = [
    ("pass_into_final_third_per90", 0.15, False),
    ("carry_into_final_third_per90", 0.25, False),
    ("n_events_third_attacking_pass_per90", 0.20, False),
    ("n_events_third_attacking_pass_cross_openplay_per90", 0.20, False),
    ("xa_per90", 0.20, False),
]

QUALITY = [
    ("obv_total_net_type_pass_per90", 0.45, False),
    ("obv_total_net_type_dribble_per90", 0.25, False),
    ("obv_total_net_type_carry_per90", 0.20, False),
    ("total_turnovers_per90", 0.10, True),
]

PRESS = [
    ("pressure_per90", 0.35, False),
    ("n_events_third_attacking_pressure_per90", 0.35, False),
    ("ball_recovery_offensive_per90", 0.15, False),
    ("counterpress_per90", 0.15, False),
]

DEF_EXEC = [
    ("duel_success_rate", 0.25, False),
    ("tackle_success_pct", 0.25, False),
    ("ball_recovery_success_pct", 0.15, False),
    ("interception_success_rate", 0.10, False),
    ("clearances_total_per90", 0.10, False),
    ("blocks_total_per90", 0.05, False),
    ("defensive_actions_lost_per90", 0.10, True),
]

DEF_OBV = [
    ("obv_total_net_type_duel_per90", 0.20, False),
    ("obv_total_net_duel_type_tackle_per90", 0.25, False),
    ("obv_total_net_type_interception_per90", 0.20, False),
    ("obv_total_net_type_ball_recovery_per90", 0.20, False),
    ("obv_total_net_type_clearance_per90", 0.15, False),
]

# Categorías "macro" (4 finales en app)
CATS = {
    "Score_Profundidad": DEPTH,
    "Score_Calidad": QUALITY,
    "Score_Presion": PRESS,
    # Score_Defensivo se arma aparte (Exec/OBV)
}

CAT_W = {
    "Score_Profundidad": 0.30,
    "Score_Calidad": 0.30,
    "Score_Presion": 0.20,
    "Score_Defensivo": 0.20,
}


# =========================
# FUNCIÓN PRINCIPAL (DF)
# =========================
def score_lateral_df(
    per90_df: pd.DataFrame,
    position_group: str = "Lateral",
    min_minutes: int = 450,
    min_matches: int = 3,
    flag_q: float = 0.75,
    def_exec_w: float = 0.60,
    def_obv_w: float = 0.40,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Devuelve DataFrame con scores de laterales.
    NO escribe archivo. Ideal para Streamlit.

    Requiere columnas:
    - primary_position
    - total_minutes, matches_played, player_id, player_name, teams
    """

    base = per90_df.copy()

    # filtros por posición
    base = filter_by_position_group(base, position_group)

    # filtros minutos / partidos
    if "total_minutes" not in base.columns or "matches_played" not in base.columns:
        raise KeyError("Faltan columnas total_minutes y/o matches_played en el per90_df.")

    base = base[base["total_minutes"] >= min_minutes].copy()
    base = base[base["matches_played"] >= min_matches].copy()

    if base.empty:
        raise ValueError(f"No hay jugadores de {position_group} que cumplan filtros.")

    # compatibilidad nombres salida
    base = base.rename(columns={
        "teams": "team_name",
        "matches_played": "matches",
        "total_minutes": "minutes",
    })

    # --- preparar métricas
    ALL_ITEMS = DEPTH + QUALITY + PRESS + DEF_EXEC + DEF_OBV

    # a numérico
    for col, _, _ in ALL_ITEMS:
        if col in base.columns:
            base[col] = safe_numeric(base[col])

    # percentiles por métrica
    missing_cols = []
    for col, _, inv in ALL_ITEMS:
        if col not in base.columns:
            missing_cols.append(col)
            continue
        x = -base[col] if inv else base[col]
        base[f"pct__{col}"] = pct_rank_0_100(x)

    if verbose and missing_cols:
        print(f"[Lateral] Columnas faltantes ignoradas: {len(missing_cols)}")
        for c in missing_cols[:15]:
            print(" -", c)

    # scores por categoría macro (3)
    for cat, items in CATS.items():
        pct_items = [(f"pct__{col}", w) for col, w, _ in items if f"pct__{col}" in base.columns]
        base[cat] = wavg(base, pct_items) if pct_items else np.nan

    # subscores defensivos
    def _calc_subscore(items, out_col):
        pct_items = [(f"pct__{col}", w) for col, w, _ in items if f"pct__{col}" in base.columns]
        base[out_col] = wavg(base, pct_items) if pct_items else np.nan

    _calc_subscore(DEF_EXEC, "Score_Def_Exec")
    _calc_subscore(DEF_OBV, "Score_Def_OBV")

    # defensivo final: renormaliza pesos según disponibilidad (sin 0 fantasma)
    a = base["Score_Def_Exec"]
    b = base["Score_Def_OBV"]

    w_exec = np.where(a.notna(), def_exec_w, 0.0)
    w_obv  = np.where(b.notna(), def_obv_w,  0.0)
    w_sum  = w_exec + w_obv

    base["Score_Defensivo"] = np.where(
        w_sum > 0,
        (a.fillna(0) * w_exec + b.fillna(0) * w_obv) / w_sum,
        np.nan,
    )

    # Score overall con pesos CAT_W (usando columnas reales)
    # OJO: CAT_W tiene Score_Defensivo; CATS no.
    # Validación simple:
    num = 0.0
    den = 0.0
    for c, w in CAT_W.items():
        if c not in base.columns:
            continue
        valid = base[c].notna()
        num += base[c].fillna(0) * w * valid
        den += w * valid
    base["Score_Overall"] = np.where(den > 0, num / den, np.nan)

    # flags
    for flag_name, score_col in [
        ("Flag_Profundos", "Score_Profundidad"),
        ("Flag_Tecnicos", "Score_Calidad"),
        ("Flag_Presionantes", "Score_Presion"),
        ("Flag_Protectores", "Score_Defensivo"),
    ]:
        if score_col in base.columns and base[score_col].notna().any():
            thr = base[score_col].quantile(flag_q)
            base[flag_name] = base[score_col] >= thr
        else:
            base[flag_name] = False

    def tags(r):
        t = []
        if r.get("Flag_Profundos", False): t.append("Profundos")
        if r.get("Flag_Tecnicos", False): t.append("Técnicos")
        if r.get("Flag_Presionantes", False): t.append("Presionantes")
        if r.get("Flag_Protectores", False): t.append("Protectores")
        return " | ".join(t) if t else "Balanceados"

    base["Flags"] = base.apply(tags, axis=1)

    # columnas output (mantengo las que usa tu app)
    cols = [
        "player_id", "player_name", "team_name", "matches", "minutes",
        "primary_position", "primary_position_share",
        "Score_Profundidad", "Score_Calidad", "Score_Presion",
        "Score_Defensivo", "Score_Overall",
        "Flag_Profundos", "Flag_Tecnicos", "Flag_Presionantes", "Flag_Protectores",
        "Flags",
    ]
    cols = [c for c in cols if c in base.columns]

    out = base[cols].sort_values("Score_Overall", ascending=False).reset_index(drop=True)

    if verbose:
        print("[Lateral] notna Def_Exec:", out.get("Score_Def_Exec", pd.Series(dtype=float)).notna().sum() if "Score_Def_Exec" in base.columns else "n/a")
        print("[Lateral] notna Def_OBV :", out.get("Score_Def_OBV", pd.Series(dtype=float)).notna().sum() if "Score_Def_OBV" in base.columns else "n/a")
        print("[Lateral] notna Defensivo:", out["Score_Defensivo"].notna().sum() if "Score_Defensivo" in out.columns else "n/a")

    return out


# =========================
# WRAPPER CSV (legacy)
# =========================
def run_position_scoring(
    per90_csv: Path,
    out_csv: Path,
    position_group: str = "Lateral",
    min_minutes: int = 450,
    min_matches: int = 3,
    flag_q: float = 0.75,
    def_exec_w: float = 0.60,
    def_obv_w: float = 0.40,
) -> pd.DataFrame:
    """
    Mantiene tu interfaz vieja (lee CSV -> score_lateral_df -> escribe CSV).
    """
    print("=" * 70)
    print(f"SCORING DE {position_group.upper()} (CSV wrapper)")
    print("=" * 70)

    per90 = pd.read_csv(per90_csv, low_memory=False, encoding="utf-8-sig")
    out = score_lateral_df(
        per90_df=per90,
        position_group=position_group,
        min_minutes=min_minutes,
        min_matches=min_matches,
        flag_q=flag_q,
        def_exec_w=def_exec_w,
        def_obv_w=def_obv_w,
        verbose=True,
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"✅ Output guardado en: {out_csv}")
    print("=" * 70)
    return out


# =========================
# EJEMPLO DE USO
# =========================
if __name__ == "__main__":
    per90_csv = Path("outputs/all_players_complete_2025_2026.csv")
    out_csv = Path("outputs/lateral_scores_2025_2026.csv")

    run_position_scoring(
        per90_csv=per90_csv,
        out_csv=out_csv,
        position_group="Lateral",
        min_minutes=450,
        min_matches=3,
        flag_q=0.75,
        def_exec_w=0.60,
        def_obv_w=0.40,
    )
