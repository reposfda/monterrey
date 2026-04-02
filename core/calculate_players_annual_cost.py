# core/calculate_players_annual_cost.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from unidecode import unidecode

import config

# =============================================================================
# CONFIG LOCAL
# =============================================================================

SALARIES_FILE = config.DATA_DIR / "salarios" / "ligamx_salarios.csv"
TRANSFERS_DIR = config.DATA_DIR / "transfers"
OUTPUT_FILE = config.DATA_DIR / "salarios" / "players_annual_cost.csv"

CLUB_NAME_MAPPING_SALARIES_TRANSFERS = {
    "Atlas": "Atlas Guadalajara",
    "Tigres UANL": "Tigres UANL",
    "Pachuca": "CF Pachuca",
    "Queretaro": "Querétaro FC",
    "Monterrey": "CF Monterrey",
    "Necaxa": "Club Necaxa",
    "Cruz Azul": "CD Cruz Azul",
    "Juarez": "FC Juárez",
    "Guadalajara": "CD Guadalajara",
    "Tijuana": "Club Tijuana",
    "Pumas UNAM": "Pumas UNAM",
    "Atletico San Luis": "Atlético de San Luis",
    "America": "CF América",
    "Toluca": "Deportivo Toluca FC",
    "Leon": "Club León FC",
    "Puebla": "Club Puebla",
    "Santos Laguna": "Santos Laguna",
    "Mazatlan": "Mazatlán FC",
}

# =============================================================================
# LOADERS
# =============================================================================

def load_salaries(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"No existe el archivo de salarios: {file_path}")

    salaries_raw = pd.read_csv(file_path)
    salaries_raw = salaries_raw[
        ["club_name", "player_name", "total_gross_salary", "signed", "contract_expiration"]
    ]

    return salaries_raw


def load_transfers(folder_path: Path) -> tuple[pd.DataFrame, list[pd.DataFrame], dict[str, pd.DataFrame]]:
    if not folder_path.exists():
        raise FileNotFoundError(f"No existe la carpeta de transferencias: {folder_path}")

    ligamx_transfers_list = []
    ligamx_transfers_dict = {}

    for file_path in folder_path.glob("*.csv"):
        df_name = file_path.stem
        df = pd.read_csv(file_path)

        ligamx_transfers_list.append(df)
        ligamx_transfers_dict[df_name] = df

        print(f"✅ Loaded {file_path.name} into DataFrame: {df_name}")

    if not ligamx_transfers_list:
        raise ValueError(f"No se encontraron CSVs en {folder_path}")

    transfers = pd.concat(ligamx_transfers_list, ignore_index=True)
    transfers = transfers.drop_duplicates()

    return transfers, ligamx_transfers_list, ligamx_transfers_dict

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


def find_best_match_blocked(row, compras: pd.DataFrame, score_cutoff: int = 87) -> pd.Series:
    """
    For a row in salaries, find the best matching player in compras
    within the mapped club.
    Returns:
      compras_idx (actual compras index), match_score
    """
    name_key = row["name_key"]
    club_comp = row["club_name_compras"]

    if pd.isna(club_comp) or not name_key:
        return pd.Series([pd.NA, 0])

    subset = compras[compras["club_name"] == club_comp]
    if subset.empty:
        return pd.Series([pd.NA, 0])

    choices_local = subset["name_key"].to_dict()

    result = process.extractOne(
        name_key,
        choices_local,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=score_cutoff,
    )

    if result is None:
        return pd.Series([pd.NA, 0])

    best_value, score, compras_idx = result
    return pd.Series([compras_idx, score])


# =============================================================================
# CORE LOGIC
# =============================================================================

def prepare_transfers_for_matching(transfers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    transfers = transfers[transfers["transfer_type"] == "Transferencia"].copy()

    compras_raw = transfers[transfers["direction"] == "In"].copy()
    compras_raw = compras_raw[
        ["season", "transfer_window", "club_id", "club_name", "player_id", "player_name", "transfer_price"]
    ]

    ventas_raw = transfers[transfers["direction"] == "Out"].copy()
    ventas_raw = ventas_raw[
        ["season", "transfer_window", "club_id", "club_name", "player_id", "player_name", "transfer_price"]
    ]

    return compras_raw, ventas_raw


def build_salary_purchase_match(
    salaries_raw: pd.DataFrame,
    compras_raw: pd.DataFrame,
    score_cutoff: int = 87,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    salaries = salaries_raw.copy()
    compras = compras_raw.copy()

    salaries["name_key"] = salaries["player_name"].map(normalize_name)
    compras["name_key"] = compras["player_name"].map(normalize_name)

    salaries["club_name_compras"] = salaries["club_name"].map(CLUB_NAME_MAPPING_SALARIES_TRANSFERS)

    salaries[["compras_idx", "match_score"]] = salaries.apply(
        lambda row: find_best_match_blocked(row, compras=compras, score_cutoff=score_cutoff),
        axis=1,
    )

    suspicious = salaries[(salaries["match_score"] > 0) & (salaries["match_score"] < 90)].copy()

    salaries["compras_idx"] = pd.to_numeric(salaries["compras_idx"], errors="coerce").astype("Int64")

    salarios_compras = salaries.merge(
        compras,
        how="left",
        left_on="compras_idx",
        right_index=True,
        suffixes=("_sal", "_comp"),
    )

    salarios_compras = salarios_compras[
        [
            "club_name_sal",
            "player_name_sal",
            "total_gross_salary",
            "signed",
            "contract_expiration",
            "match_score",
            "season",
            "transfer_window",
            "club_id",
            "player_id",
            "transfer_price",
        ]
    ].rename(
        columns={
            "club_name_sal": "club_name",
            "player_name_sal": "player_name",
            "match_score": "match_score_salaries_compras",
            "club_id": "club_id_tmkt",
            "player_id": "player_id_tmkt",
        }
    )

    return salarios_compras, suspicious


def build_players_annual_cost(salarios_compras: pd.DataFrame) -> pd.DataFrame:
    players_annual_cost = salarios_compras.copy()

    players_annual_cost["signed_dt"] = pd.to_datetime(
        players_annual_cost["signed"],
        format="%d-%m-%Y",
        errors="coerce",
    )
    players_annual_cost["expiration_dt"] = pd.to_datetime(
        players_annual_cost["contract_expiration"],
        format="%d-%m-%Y",
        errors="coerce",
    )

    players_annual_cost["contract_length_years"] = (
        players_annual_cost["expiration_dt"].dt.year
        - players_annual_cost["signed_dt"].dt.year
    )

    players_annual_cost["player_annual_fee"] = np.nan

    mask_valid = (
        players_annual_cost["contract_length_years"].notna()
        & (players_annual_cost["contract_length_years"] > 0)
        & players_annual_cost["transfer_price"].notna()
    )

    players_annual_cost.loc[mask_valid, "player_annual_fee"] = (
        players_annual_cost.loc[mask_valid, "transfer_price"]
        / players_annual_cost.loc[mask_valid, "contract_length_years"]
    )

    players_annual_cost["player_annual_fee"] = (
        players_annual_cost["player_annual_fee"]
        .replace([np.inf, -np.inf], np.nan)
        .round()
        .astype("Int64")
    )

    players_annual_cost = players_annual_cost.drop(columns=["signed_dt", "expiration_dt"])

    players_annual_cost["player_annual_cost"] = players_annual_cost[
        ["total_gross_salary", "player_annual_fee"]
    ].sum(axis=1, skipna=True)

    players_annual_cost["player_annual_cost"] = (
        players_annual_cost["player_annual_cost"]
        .round()
        .astype("Int64")
    )

    annual_salary_cost_per_team = (
        players_annual_cost.groupby("club_name")["total_gross_salary"]
        .sum()
        .reset_index()
        .rename(columns={"total_gross_salary": "team_total_salary"})
    )

    players_annual_cost = players_annual_cost.merge(
        annual_salary_cost_per_team,
        on="club_name",
        how="left",
    )

    annual_transfers_fees_per_team = (
        players_annual_cost.groupby("club_name")["player_annual_fee"]
        .sum()
        .reset_index()
        .rename(columns={"player_annual_fee": "team_total_fees"})
    )

    players_annual_cost = players_annual_cost.merge(
        annual_transfers_fees_per_team,
        on="club_name",
        how="left",
    )

    players_annual_cost["team_total_cost"] = (
        players_annual_cost["team_total_salary"]
        + players_annual_cost["team_total_fees"]
    )

    players_annual_cost["player_annual_cost_percentage"] = round(
        players_annual_cost["player_annual_cost"] / players_annual_cost["team_total_cost"],
        3,
    )

    return players_annual_cost


def save_players_annual_cost(df: pd.DataFrame, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"💾 Archivo guardado en: {output_file}")


def main() -> pd.DataFrame:
    print(f"Leyendo salarios desde: {SALARIES_FILE}")
    print(f"Leyendo transfers desde: {TRANSFERS_DIR}")

    salaries_raw = load_salaries(SALARIES_FILE)
    transfers, _, _ = load_transfers(TRANSFERS_DIR)

    compras_raw, ventas_raw = prepare_transfers_for_matching(transfers)

    salarios_compras, suspicious = build_salary_purchase_match(
        salaries_raw=salaries_raw,
        compras_raw=compras_raw,
        score_cutoff=87,
    )

    if not suspicious.empty:
        print(f"⚠️ Matches sospechosos encontrados: {len(suspicious)}")

    players_annual_cost = build_players_annual_cost(salarios_compras)
    save_players_annual_cost(players_annual_cost, OUTPUT_FILE)

    print(f"✅ Proceso finalizado. Shape final: {players_annual_cost.shape}")
    return players_annual_cost


if __name__ == "__main__":
    main()