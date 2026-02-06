# turnover_scoring.py
from __future__ import annotations

import pandas as pd


# -------------------------
# Config editable
# -------------------------
OPEN_PLAY_PATTERN = {"Regular Play"}

RESTART_PATTERNS = {
    "From Corner",
    "From Free Kick",
    "From Throw In",
    "From Goal Kick",
    "From Keeper",
    "From Kick Off",
    "Other",
}

DEFAULT_EXCLUDE_TYPES = {
    "Shot",
    "Foul Won",
    "Foul Committed",
    "Injury Stoppage",
    "Substitution",
}


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")


def classify_turnover(row: pd.Series) -> str:
    """
    Etiqueta el "cómo fue" la pérdida con type + outcomes (si existen columnas).
    """
    t = row.get("type")

    if t == "Pass":
        outc = row.get("pass_outcome")
        return f"Pass: {outc}" if pd.notna(outc) and outc != "" else "Pass"

    if t == "Dribble":
        outc = row.get("dribble_outcome")
        return f"Dribble: {outc}" if pd.notna(outc) and outc != "" else "Dribble"

    if t == "Duel":
        outc = row.get("duel_outcome")
        return f"Duel: {outc}" if pd.notna(outc) and outc != "" else "Duel"

    if t == "Ball Receipt*":
        outc = row.get("ball_receipt_outcome")
        return f"Ball Receipt: {outc}" if pd.notna(outc) and outc != "" else "Ball Receipt"

    if t in {"Miscontrol", "Dispossessed"}:
        return t

    if t == "Carry":
        if bool(row.get("out")) is True:
            return "Carry: Out"
        return "Carry"

    return str(t)


def compute_player_turnovers(
    events: pd.DataFrame,
    *,
    open_play_only: bool = True,
    exclude_restart_patterns: bool = True,
    exclude_types: set[str] | None = None,
) -> pd.DataFrame:
    """
    Detecta pérdidas por cambio de possession_team entre eventos consecutivos (dentro del partido),
    y se la atribuye al último evento del equipo en posesión (team == possession_team).

    Devuelve un DF a nivel evento.
    """
    _require_cols(events, ["match_id", "period", "index", "team", "possession_team", "type"])

    df = events.copy()
    df = df.sort_values(["match_id", "period", "index"]).reset_index(drop=True)

    # shift dentro del match
    df["next_possession_team"] = df.groupby("match_id")["possession_team"].shift(-1)
    df["next_play_pattern"] = df.groupby("match_id")["play_pattern"].shift(-1) if "play_pattern" in df.columns else pd.NA
    df["next_index"] = df.groupby("match_id")["index"].shift(-1)
    df["next_team"] = df.groupby("match_id")["team"].shift(-1)
    df["next_type"] = df.groupby("match_id")["type"].shift(-1)

    # sólo eventos del equipo en posesión (evita atribuir pérdidas a acciones defensivas del rival)
    in_poss = df["team"].notna() & (df["team"] == df["possession_team"])

    # turnover = cambia possession_team al siguiente evento
    change_poss = df["next_possession_team"].notna() & (df["possession_team"] != df["next_possession_team"])

    mask = in_poss & change_poss

    # open play
    if open_play_only and "play_pattern" in df.columns:
        mask &= df["play_pattern"].isin(OPEN_PLAY_PATTERN)

    # excluir reinicios en la posesión siguiente (opcional)
    if exclude_restart_patterns and "play_pattern" in df.columns:
        mask &= ~df["next_play_pattern"].isin(RESTART_PATTERNS)

    # excluir tipos (opcional)
    if exclude_types is None:
        exclude_types = set(DEFAULT_EXCLUDE_TYPES)
    if exclude_types:
        mask &= ~df["type"].isin(exclude_types)

    # requiere jugador (si existe la col)
    if "player" in df.columns:
        mask &= df["player"].notna()

    out = df.loc[mask].copy()

    # outcomes opcionales (si no existen, las crea para que el script no rompa)
    for c in ["player", "player_id", "timestamp", "minute", "second", "play_pattern",
              "pass_outcome", "dribble_outcome", "duel_outcome", "ball_receipt_outcome", "out", "id"]:
        if c not in out.columns:
            out[c] = pd.NA

    out["turnover_team_lost"] = out["possession_team"]
    out["turnover_team_won"] = out["next_possession_team"]
    out["turnover_how"] = out.apply(classify_turnover, axis=1)

    keep = [
        "match_id", "period", "index", "timestamp", "minute", "second",
        "turnover_team_lost", "turnover_team_won",
        "team", "possession_team",
        "player", "player_id",
        "type", "turnover_how",
        "pass_outcome", "dribble_outcome", "duel_outcome", "ball_receipt_outcome", "out",
        "play_pattern",
        "next_index", "next_team", "next_type",
        "id",
    ]
    keep = [c for c in keep if c in out.columns]
    return out[keep].reset_index(drop=True)