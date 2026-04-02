# core/consolidar_salarios.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import pandas as pd

import config


# =============================================================================
# CONFIG LOCAL
# =============================================================================

SALARIES_DIR = config.DATA_DIR / "salarios" / "equipos" / config.SALARIES_SEASON
OUTPUT_FILE = config.DATA_DIR / "salarios" / "ligamx_salarios.csv"

MONETARY_COLS = [
    "gross_salary_per_week",
    "gross_salary_per_year",
    "gross_bonus_per_year",
    "total_gross_salary",
    "gross_salary_remaining",
]

DATE_COLS = [
    "signed",
    "contract_expiration",
]


# =============================================================================
# HELPERS
# =============================================================================

def load_salary_files(folder_path: Path) -> tuple[list[pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Carga todos los CSV del primer nivel de la carpeta indicada.
    Devuelve:
    - lista de DataFrames
    - diccionario {nombre_archivo_sin_extension: dataframe}
    """
    if not folder_path.exists():
        raise FileNotFoundError(f"No existe la carpeta de salarios: {folder_path}")

    ligamx_salaries_list = []
    ligamx_salaries_dict = {}

    for file_path in folder_path.glob("*.csv"):
        df_name = file_path.stem
        df = pd.read_csv(file_path)

        ligamx_salaries_list.append(df)
        ligamx_salaries_dict[df_name] = df

        print(f"✅ Loaded {file_path.name} into DataFrame: {df_name}")

    if not ligamx_salaries_list:
        raise ValueError(f"No se encontraron archivos CSV en {folder_path}")

    return ligamx_salaries_list, ligamx_salaries_dict


def clean_monetary_columns(df: pd.DataFrame, monetary_cols: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in monetary_cols:
        if col not in df.columns:
            print(f"⚠️ Columna monetaria no encontrada, se omite: {col}")
            continue

        df[col] = (
            df[col]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        )

        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df


def clean_date_columns(df: pd.DataFrame, date_cols: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in date_cols:
        if col not in df.columns:
            print(f"⚠️ Columna de fecha no encontrada, se omite: {col}")
            continue

        df[col] = pd.to_datetime(df[col], format="%b %d, %Y", errors="coerce").dt.strftime("%d-%m-%Y")

    return df


def consolidate_salaries(folder_path: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Carga, concatena y limpia los salarios.
    """
    ligamx_salaries_list, ligamx_salaries_dict = load_salary_files(folder_path)

    ligamx_salaries = pd.concat(ligamx_salaries_list, ignore_index=True)

    ligamx_salaries = clean_monetary_columns(ligamx_salaries, MONETARY_COLS)
    ligamx_salaries = clean_date_columns(ligamx_salaries, DATE_COLS)

    return ligamx_salaries, ligamx_salaries_dict


def save_salaries(df: pd.DataFrame, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"💾 Archivo guardado en: {output_file}")


def main() -> pd.DataFrame:
    print(f"Temporada seleccionada desde config: {config.SALARIES_SEASON}")
    print(f"Leyendo salarios desde: {SALARIES_DIR}")

    ligamx_salaries, _ = consolidate_salaries(SALARIES_DIR)
    save_salaries(ligamx_salaries, OUTPUT_FILE)

    print(f"✅ Consolidación finalizada. Shape: {ligamx_salaries.shape}")
    return ligamx_salaries


if __name__ == "__main__":
    main()