# -*- coding: utf-8 -*-
"""
Scoring – Volante (DataFrame-first)

- Función principal: score_volante_df(per90_df, ...)
- Wrapper legacy: run_volante_scoring(per90_csv, out_csv, ...)

Incluye fallbacks razonables para:
- complete_passes_per90
- completed_passes_under_pressure_per90
- duel_tackle_per90
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from utils.scoring_io import read_input

from positions_config import normalize_group, sb_positions_for


# =========================
# HELPERS
# =========================
def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def pct_rank_0_100(s: pd.Series) -> pd.Series:
    x = s.copy()
    m = x.notna()
    out = pd.Series(np.nan, index=x.index, dtype="float64")
    out.loc[m] = x.loc[m].rank(pct=True, method="average") * 100.0
    return out


def wavg(df: pd.DataFrame, cols_weights):
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
    group = normalize_group(group)
    valid_positions = sb_positions_for(group)

    if "primary_position" not in df.columns:
        raise KeyError("Falta columna 'primary_position' en el DataFrame.")

    return df[df["primary_position"].isin(valid_positions)].copy()


def ensure_col_from_first_available(df: pd.DataFrame, target: str, candidates: list[str], verbose: bool = False):
    """
    Si target no existe, intenta crearlo copiando la primera columna disponible en candidates.
    """
    if target in df.columns:
        return
    for c in candidates:
        if c in df.columns:
            df[target] = df[c]
            if verbose:
                print(f"[Volante] '{target}' mapeado desde '{c}'")
            return
    df[target] = np.nan
    if verbose:
        print(f"[Volante] '{target}' no encontrado. Queda NaN.")


# =========================
# MÉTRICAS / CATEGORÍAS
# =========================
POSESION = [
    ("complete_passes_per90",                  0.20, False),
    ("completed_passes_under_pressure_per90",  0.30, False),
    ("total_turnovers_per90",                  0.20, True),
    ("obv_total_net_type_pass_per90",          0.30, False),
]

PROGRESION = [
    ("pass_into_final_third_per90",    0.20, False),
    ("carry_into_final_third_per90",   0.20, False),
    ("obv_total_net_type_carry_per90", 0.20, False),
    ("pass_switch_per90",              0.20, False),
    ("pass_through_ball_per90",        0.20, False),
]

TERRITORIALES = [
    ("n_events_third_defensive_pressure_per90",      0.12, False),
    ("n_events_third_middle_pressure_per90",         0.18, False),
    ("counterpress_per90",                           0.05, False),

    ("n_events_third_defensive_ball_recovery_per90", 0.15, False),
    ("n_events_third_middle_ball_recovery_per90",    0.20, False),

    ("obv_total_net_type_interception_per90",        0.15, False),
    ("obv_total_net_type_ball_recovery_per90",       0.15, False),
]

CONTENCION = [
    ("duel_tackle_per90",                            0.22, False),
    ("obv_total_net_duel_type_tackle_per90",         0.23, False),

    ("interception_success_rate",                    0.17, False),
    ("obv_total_net_type_interception_per90",        0.23, False),

    ("n_events_third_defensive_interception_per90",  0.15, False),
]

CATS = {
    "Score_Posesion": POSESION,
    "Score_Progresion": PROGRESION,
    "Score_Territoriales": TERRITORIALES,
    "Score_Contencion": CONTENCION,
}

CAT_W = {
    "Score_Posesion": 0.25,
    "Score_Progresion": 0.30,
    "Score_Territoriales": 0.25,
    "Score_Contencion": 0.20,
}


# =========================
# FUNCIÓN PRINCIPAL (DF)
# =========================
def score_volante_df(
    per90_df: pd.DataFrame,
    position_group: str = "Volante",
    min_minutes: int = 450,
    min_matches: int = 3,
    flag_q: float = 0.75,
    verbose: bool = True,
) -> pd.DataFrame:

    base = per90_df.copy()

    if verbose:
        print("=" * 70)
        print("SCORING DE VOLANTE")
        print("=" * 70)
        print(f"✓ Total jugadores en input: {len(base)}\n")
        print(f"🔍 Filtrando por posición: {position_group}")

    # filtro posición (SIEMPRE)
    base = filter_by_position_group(base, position_group)

    if verbose:
        print(f"✓ Jugadores en posición {position_group}: {len(base)}\n")
        print("⏱️  Aplicando filtros:")
        print(f"  - Minutos mínimos: {min_minutes}")
        print(f"  - Partidos mínimos: {min_matches}")

    # filtros minutos / partidos
    if "total_minutes" not in base.columns or "matches_played" not in base.columns:
        raise KeyError("Faltan columnas total_minutes y/o matches_played.")

    base = base[base["total_minutes"] >= min_minutes].copy()
    base = base[base["matches_played"] >= min_matches].copy()

    if verbose:
        print(f"✓ Jugadores después de filtros: {len(base)}\n")

    if base.empty:
        raise ValueError(f"No hay jugadores de {position_group} que cumplan filtros.")

    # renombres
    base = base.rename(
        columns={
            "teams": "team_name",
            "matches_played": "matches",
            "total_minutes": "minutes",
        }
    )

    if verbose:
        print("🔧 Calculando métricas derivadas (si aplica)...\n")

    # =========================
    # DERIVADAS / FALLBACKS
    # =========================
    # complete_passes_per90
    if "complete_passes_per90" not in base.columns:
        if "complete_passes" in base.columns:
            base["complete_passes"] = safe_numeric(base["complete_passes"])
            base["complete_passes_per90"] = np.where(
                base["minutes"] > 0,
                base["complete_passes"] / base["minutes"] * 90.0,
                np.nan,
            )
            if verbose:
                print("[Volante] complete_passes_per90 calculado desde complete_passes")
        else:
            ensure_col_from_first_available(
                base,
                "complete_passes_per90",
                candidates=["completed_passes_per90", "passes_completed_per90"],
                verbose=verbose,
            )

    # completed_passes_under_pressure_per90 (alias comunes)
    ensure_col_from_first_available(
        base,
        "completed_passes_under_pressure_per90",
        candidates=[
            "complete_passes_under_pressure_per90",
            "completed_passes_under_pressure_per90",
            "passes_completed_under_pressure_per90",
            "completed_passes_under_pressure",
        ],
        verbose=verbose,
    )

    # duel_tackle_per90 (alias comunes)
    ensure_col_from_first_available(
        base,
        "duel_tackle_per90",
        candidates=[
            "n_events_duel_type_tackle_per90",
            "tackles_total_per90",
            "tackles_per90",
        ],
        verbose=verbose,
    )

    if verbose:
        print("\n🎯 Calculando scores...\n")

    # =========================
    # CÁLCULO DE SCORES
    # =========================
    ALL_ITEMS = POSESION + PROGRESION + TERRITORIALES + CONTENCION

    # a numérico
    for col, _, _ in ALL_ITEMS:
        if col in base.columns:
            base[col] = safe_numeric(base[col])

    # percentiles
    missing_cols = []
    for col, _, inv in ALL_ITEMS:
        if col not in base.columns:
            missing_cols.append(col)
            continue
        x = -base[col] if inv else base[col]
        base[f"pct__{col}"] = pct_rank_0_100(x)

    if missing_cols and verbose:
        print(f"⚠️  Columnas no encontradas (serán ignoradas): {len(missing_cols)}")
        for c in missing_cols:
            print(f"  - {c}")

    # score por categoría
    for cat, items in CATS.items():
        pct_items = [(f"pct__{col}", w) for col, w, _ in items if f"pct__{col}" in base.columns]
        base[cat] = wavg(base, pct_items) if pct_items else np.nan

    # overall (ponderado por disponibilidad)
    num = pd.Series(0.0, index=base.index)
    den = pd.Series(0.0, index=base.index)
    for c, w in CAT_W.items():
        if c not in base.columns:
            continue
        valid = base[c].notna()
        num = num + base[c].fillna(0) * (w * valid.astype(float))
        den = den + (w * valid.astype(float))
    base["Score_Overall"] = np.where(den > 0, num / den, np.nan)

    if verbose:
        print("✓ Scores calculados\n")

    # =========================
    # FLAGS / TAGS
    # =========================
    for flag_name, score_col in [
        ("Flag_Posesion", "Score_Posesion"),
        ("Flag_Progresion", "Score_Progresion"),
        ("Flag_Territoriales", "Score_Territoriales"),
        ("Flag_Contencion", "Score_Contencion"),
    ]:
        if score_col in base.columns and base[score_col].notna().any():
            thr = base[score_col].quantile(flag_q)
            base[flag_name] = base[score_col] >= thr
        else:
            base[flag_name] = False

    def tags(r):
        t = []
        if r.get("Flag_Posesion", False): t.append("Posesión")
        if r.get("Flag_Progresion", False): t.append("Progresión")
        if r.get("Flag_Territoriales", False): t.append("Territoriales")
        if r.get("Flag_Contencion", False): t.append("Contención")
        return " | ".join(t) if t else "Balanceados"

    base["Flags"] = base.apply(tags, axis=1)

    # output
    cols = [
        "player_id", "player_name", "team_name", "matches", "minutes",
        "primary_position", "primary_position_share",
        "Score_Posesion", "Score_Progresion", "Score_Territoriales", "Score_Contencion",
        "Score_Overall",
        "Flag_Posesion", "Flag_Progresion", "Flag_Territoriales", "Flag_Contencion",
        "Flags",
    ]
    cols = [c for c in cols if c in base.columns]

    out = base[cols].sort_values("Score_Overall", ascending=False).reset_index(drop=True)

    if verbose:
        print("🏷️  Asignando flags (top 25%)...")
        print("=" * 70)
        print(f"📊 Jugadores evaluados: {len(out)}")

    return out


# =========================
# RUN (df-first) + WRAPPER legacy
# =========================
def run_volante_scoring(
    per90_csv: Path | None = None,
    out_csv: Path | None = None,
    df: pd.DataFrame | None = None,
    position_group: str = "Volante",
    min_minutes: int = 450,
    min_matches: int = 3,
    flag_q: float = 0.75,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Entry-point estándar (igual que CB/Interior/Extremo/etc):
    - acepta df o per90_csv
    - out_csv opcional
    """
    print("=" * 70)
    print(f"SCORING DE {position_group.upper()} (df-first)")
    print("=" * 70)

    per90 = read_input(per90_csv=per90_csv, df=df)

    out = score_volante_df(
        per90_df=per90,
        position_group=position_group,
        min_minutes=min_minutes,
        min_matches=min_matches,
        flag_q=flag_q,
        verbose=verbose,
    )

    if out_csv is not None:
        out_csv = Path(out_csv)
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
    out_csv = Path("outputs/volante_scores_2025_2026.csv")

    run_volante_scoring(
        per90_csv=per90_csv,
        out_csv=out_csv,
        position_group="Volante",
        min_minutes=450,
        min_matches=3,
        flag_q=0.75,
    )
