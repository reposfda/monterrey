# core/build_score_cost.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process
from unidecode import unidecode

import config

# =============================================================================
# CONFIG LOCAL
# =============================================================================

METHOD_FOLDER_MAP = {
    "promedio_ponderado": "promedio_ponderado",
    "valor_presente": "valor_presente",
    "media_bayesiana": "media_bayesiana",
    "ponderacion_dinamica": "ponderacion_dinamica",
    "momentum": "momentum",
}

METHOD_SUFFIX_MAP = {
    "promedio_ponderado": "pp",
    "valor_presente": "vp",
    "media_bayesiana": "mb",
    "ponderacion_dinamica": "pd",
    "momentum": "mo",
}

POSITIONS = ["arq", "def", "lat", "vol", "int", "ext", "del"]

SCORES_BASE_DIR = config.DATA_DIR / "scores" / "score_consolidado"
PLAYERS_ANNUAL_COST_FILE = config.DATA_DIR / "salarios" / "players_annual_cost.csv"
OUTPUT_DIR = config.DATA_DIR / "scores" / "score_cost"
OUTPUT_DIR_SIN_DATA = config.DATA_DIR / "scores" / "sin_data_economica"

# =============================================================================
# HELPERS
# =============================================================================

def normalize_name(name: str) -> str:
    if pd.isna(name):
        return ""

    s = str(name).lower()
    s = unidecode(s)
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"#?\d+$", "", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s


def get_scores_input_dir(method: str) -> Path:
    if method not in METHOD_FOLDER_MAP:
        raise ValueError(f"Método inválido: {method}")

    return SCORES_BASE_DIR


def get_scores_file_path(position: str, method: str) -> Path:
    input_dir = get_scores_input_dir(method)
    suffix = METHOD_SUFFIX_MAP[method]
    return input_dir / f"score_{position}_final_{suffix}.csv"


def load_score_files(method: str) -> dict[str, pd.DataFrame]:
    score_dfs: dict[str, pd.DataFrame] = {}

    for position in POSITIONS:
        file_path = get_scores_file_path(position, method)

        if not file_path.exists():
            print(f"⚠️ Archivo no encontrado para {position}: {file_path}")
            continue

        df = pd.read_csv(file_path)
        score_dfs[position] = df
        print(f"✅ Loaded {file_path.name} for position: {position}")

    if not score_dfs:
        raise ValueError("No se pudo cargar ningún archivo de scores.")

    return score_dfs


def load_players_annual_cost(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {file_path}")

    players_annual_cost = pd.read_csv(file_path)

    required_cols = [
        "club_name",
        "player_name",
        "player_annual_cost",
        "player_annual_cost_percentage",
        "team_total_cost",
    ]
    missing = [col for col in required_cols if col not in players_annual_cost.columns]
    if missing:
        raise ValueError(f"Faltan columnas en players_annual_cost: {missing}")

    players_annual_cost = players_annual_cost[required_cols].rename(
        columns={"club_name": "team_name"}
    )

    players_annual_cost = players_annual_cost[
        players_annual_cost["player_annual_cost_percentage"] > 0
    ].copy()

    return players_annual_cost


def build_choices(players_annual_cost: pd.DataFrame) -> dict:
    return players_annual_cost["name_key"].to_dict()


def find_best_match_for_scoring(row, choices: dict, score_cutoff: int = 90) -> pd.Series:
    """
    Para una fila de score_{pos}, encuentra el mejor jugador en players_annual_cost.
    Devuelve:
      players_idx, match_score
    """
    key = row["name_key"]
    if not key:
        return pd.Series([pd.NA, 0])

    result = process.extractOne(
        key,
        choices,
        scorer=fuzz.token_set_ratio,
        score_cutoff=score_cutoff,
    )

    if result is None:
        return pd.Series([pd.NA, 0])

    best_value, score, players_idx = result
    return pd.Series([players_idx, score])


def get_dynamic_team_name_cols(df: pd.DataFrame) -> list[str]:
    cols = [col for col in df.columns if col.startswith("team_name_")]
    return sorted(cols)


def get_dynamic_score_cols(df: pd.DataFrame) -> list[str]:
    """
    Devuelve dinámicamente:
    - todas las columnas Score_*
    - Overall_Score_Final, si existe
    """
    score_cols = [col for col in df.columns if col.startswith("Score_")]

    if "Overall_Score_Final" in df.columns and "Overall_Score_Final" not in score_cols:
        score_cols.append("Overall_Score_Final")

    def sort_key(col_name: str):
        # primero las métricas, luego el sufijo temporal si existe
        m = re.search(r"_(\d{2})$", col_name)
        suffix = int(m.group(1)) if m else 999
        return (col_name.replace(f"_{m.group(1)}", "") if m else col_name, suffix)

    return sorted(score_cols, key=sort_key)


def get_output_columns(df: pd.DataFrame) -> list[str]:
    """
    Arma dinámicamente las columnas finales del output score_cost.
    """
    output_cols = ["team_name"]

    if "player_id" in df.columns:
        output_cols.append("player_id")

    output_cols.append("player_name")
    output_cols.extend(get_dynamic_score_cols(df))

    if "player_annual_cost_percentage" in df.columns:
        output_cols.append("player_annual_cost_percentage")

    return [col for col in output_cols if col in df.columns]


def build_missing_economic_columns(df: pd.DataFrame) -> list[str]:
    cols = ["player_name"]
    cols.extend(get_dynamic_team_name_cols(df))
    return [col for col in cols if col in df.columns]

# =============================================================================
# CORE
# =============================================================================

def merge_one_position_with_cost(
    score_df: pd.DataFrame,
    players_annual_cost: pd.DataFrame,
    choices: dict,
    score_cutoff: int = 90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = score_df.copy()
    cost_df = players_annual_cost.copy()

    if "player_name" not in df.columns:
        raise ValueError("El dataframe de score no tiene columna 'player_name'.")

    df["name_key"] = df["player_name"].map(normalize_name)
    cost_df["name_key"] = cost_df["player_name"].map(normalize_name)

    df[["players_idx", "match_score"]] = df.apply(
        lambda row: find_best_match_for_scoring(row, choices=choices, score_cutoff=score_cutoff),
        axis=1,
    )

    df["players_idx"] = pd.to_numeric(df["players_idx"], errors="coerce").astype("Int64")

    merged_scoring = df.merge(
        cost_df,
        how="left",
        left_on="players_idx",
        right_index=True,
        suffixes=("_scoring", "_cost"),
    )

    if "player_name_scoring" in merged_scoring.columns:
        merged_scoring = merged_scoring.rename(columns={"player_name_scoring": "player_name"})

    missing_cols = build_missing_economic_columns(merged_scoring)
    sin_data_economica = merged_scoring[
        merged_scoring["name_key_cost"].isna()
    ][missing_cols].copy()

    filtered = merged_scoring[merged_scoring["player_annual_cost"].notna()].copy()

    output_cols = get_output_columns(filtered)
    score_cost = filtered[output_cols].copy()

    if "player_annual_cost_percentage" in score_cost.columns:
        score_cost = score_cost.rename(
            columns={"player_annual_cost_percentage": "Cost_Share"}
        )

    return score_cost, sin_data_economica


def run_score_cost_build(
    method: str,
    score_cutoff: int = 90,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    score_dfs = load_score_files(method=method)
    players_annual_cost = load_players_annual_cost(PLAYERS_ANNUAL_COST_FILE)

    players_annual_cost["name_key"] = players_annual_cost["player_name"].map(normalize_name)
    choices = build_choices(players_annual_cost)

    score_cost_results: dict[str, pd.DataFrame] = {}
    missing_results: dict[str, pd.DataFrame] = {}

    for position, score_df in score_dfs.items():
        print(f"Procesando posición: {position}")

        score_cost, sin_data_economica = merge_one_position_with_cost(
            score_df=score_df,
            players_annual_cost=players_annual_cost,
            choices=choices,
            score_cutoff=score_cutoff,
        )

        score_cost_results[position] = score_cost
        missing_results[position] = sin_data_economica

    return score_cost_results, missing_results

# =============================================================================
# SAVE
# =============================================================================

def save_outputs(
    score_cost_results: dict[str, pd.DataFrame],
    missing_results: dict[str, pd.DataFrame],
    output_dir: Path = OUTPUT_DIR,
    output_dir_sin_data: Path = OUTPUT_DIR_SIN_DATA
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_sin_data.mkdir(parents=True, exist_ok=True)

    for position, df in score_cost_results.items():
        output_file = output_dir / f"{position}_score_cost.csv"
        df.to_csv(output_file, index=False)
        print(f"💾 Guardado: {output_file}")

    for position, df in missing_results.items():
        output_file = output_dir_sin_data / f"{position}_sin_data_economica.csv"
        df.to_csv(output_file, index=False)
        print(f"💾 Guardado: {output_file}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    method = config.SCORE_CONSOLIDATION_METHOD

    print(f"Método de consolidación leído desde config: {method}")
    print(f"Leyendo scores desde: {get_scores_input_dir(method)}")
    print(f"Leyendo costos desde: {PLAYERS_ANNUAL_COST_FILE}")

    score_cost_results, missing_results = run_score_cost_build(
        method=method,
        score_cutoff=90,
    )

    save_outputs(score_cost_results, missing_results, output_dir=OUTPUT_DIR)

    print("\nResumen final:")
    for position, df in score_cost_results.items():
        print(f"{position}: score_cost={df.shape}, sin_data={missing_results[position].shape}")

    return score_cost_results, missing_results


if __name__ == "__main__":
    main()