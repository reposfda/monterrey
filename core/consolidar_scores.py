# core/consolidar_scores.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import config


# =============================================================================
# CONFIG LOCAL DEL MÓDULO
# =============================================================================

SCORES_DIR = config.DATA_DIR / "scores"
OUTPUT_DIR = config.DATA_DIR / "scores" / "score_consolidado"

POSITION_MAP = {
    "golero": "arq",
    "zaguero": "def",
    "lateral": "lat",
    "volante": "vol",
    "interior": "int",
    "extremo": "ext",
    "delantero": "del",
}

MERGE_KEY_MAP = {
    "arq": "player_id",
    "def": "player_name",   # respetando tu lógica original
    "lat": "player_id",
    "vol": "player_id",
    "int": "player_id",
    "ext": "player_id",
    "del": "player_id",
}

PROTECTED_COLS = ["player_id", "player_name"]

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

KNOWN_FIXES = {
    "Uroš ?ur?evi?": "Uroš Đurđević",
}


# =============================================================================
# HELPERS DE CARGA
# =============================================================================

def extract_score_file_info(file_path: Path) -> dict | None:
    """
    Extrae metadata desde nombres como:
    golero_scores_2024_2025.csv
    """
    pattern = r"^(?P<position>[a-z]+)_scores_(?P<start>\d{4})_(?P<end>\d{4})\.csv$"
    match = re.match(pattern, file_path.name)

    if not match:
        return None

    position_long = match.group("position")
    if position_long not in POSITION_MAP:
        return None

    start_year = match.group("start")
    end_year = match.group("end")

    return {
        "position_long": position_long,
        "position_short": POSITION_MAP[position_long],
        "start_year": start_year,
        "end_year": end_year,
        "season_suffix": end_year[-2:],
    }


def load_and_rename_score(file_path: Path, suffix: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    rename_dict = {
        col: f"{col}_{suffix}"
        for col in df.columns
        if col not in PROTECTED_COLS
    }

    return df.rename(columns=rename_dict)


def build_score_datasets() -> dict[str, pd.DataFrame]:
    """
    Lee solo los CSV del primer nivel de data/scores,
    renombra columnas según la temporada y unifica por posición.
    """
    grouped = {pos: [] for pos in POSITION_MAP.values()}

    for file_path in SCORES_DIR.glob("*.csv"):
        info = extract_score_file_info(file_path)
        if info is None:
            print(f"⚠️ Archivo ignorado: {file_path.name}")
            continue

        df = load_and_rename_score(file_path, info["season_suffix"])

        grouped[info["position_short"]].append({
            "start_year": info["start_year"],
            "end_year": info["end_year"],
            "file_name": file_path.name,
            "df": df,
        })

        print(
            f"✅ Loaded {file_path.name} -> "
            f"{info['position_short']} | sufijo {info['season_suffix']}"
        )

    dfs_posiciones = {}

    for position, items in grouped.items():
        if not items:
            continue

        items = sorted(items, key=lambda x: (x["start_year"], x["end_year"]))
        merge_key = MERGE_KEY_MAP[position]

        merged_df = items[0]["df"]
        for item in items[1:]:
            merged_df = merged_df.merge(item["df"], on=merge_key, how="outer")

        dfs_posiciones[position] = merged_df

    return dfs_posiciones


# =============================================================================
# MÉTODOS DE CONSOLIDACIÓN
# =============================================================================

def promedio_ponderado_dinamico(
    df: pd.DataFrame,
    base_col: str = "Score_Overall",
    out_col: str = "Overall_Score_Final",
    w_latest: float = 0.80,
) -> pd.DataFrame:
    """
    Consolida dinámicamente todas las columnas tipo Score_Overall_XX.

    - Si hay 1 temporada: usa esa.
    - Si hay 2 temporadas: mantiene la lógica clásica (80% última, 20% previa).
    - Si hay 3+ temporadas: asigna 80% a la última y reparte el 20% restante
      entre las anteriores dando más peso a las más recientes.
    """
    df = df.copy()

    pattern = re.compile(rf"^{re.escape(base_col)}_(\d{{2}})$")
    score_cols = []

    for col in df.columns:
        match = pattern.match(col)
        if match:
            season_suffix = int(match.group(1))
            score_cols.append((season_suffix, col))

    if not score_cols:
        raise ValueError(f"No se encontraron columnas tipo {base_col}_XX")

    score_cols = sorted(score_cols, key=lambda x: x[0])
    ordered_cols = [col for _, col in score_cols]

    if len(ordered_cols) == 1:
        df[out_col] = df[ordered_cols[0]]
        return df

    if len(ordered_cols) == 2:
        old_col, latest_col = ordered_cols[0], ordered_cols[1]

        old_s = pd.to_numeric(df[old_col], errors="coerce")
        latest_s = pd.to_numeric(df[latest_col], errors="coerce")

        both = old_s.notna() & latest_s.notna()
        df.loc[both, out_col] = w_latest * latest_s[both] + (1 - w_latest) * old_s[both]

        only_latest = latest_s.notna() & old_s.isna()
        df.loc[only_latest, out_col] = latest_s[only_latest]

        only_old = old_s.notna() & latest_s.isna()
        df.loc[only_old, out_col] = old_s[only_old]

        return df

    previous_cols = ordered_cols[:-1]
    latest_col = ordered_cols[-1]

    if len(previous_cols) == 1:
        prev_weights = np.array([1.0])
    else:
        prev_weights = np.arange(1, len(previous_cols) + 1, dtype=float)
        prev_weights = prev_weights / prev_weights.sum()

    prev_total_weight = 1 - w_latest
    prev_weights = prev_weights * prev_total_weight
    latest_weight = w_latest

    def row_weighted_score(row):
        values = []
        weights = []

        for col, w in zip(previous_cols, prev_weights):
            val = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if pd.notna(val):
                values.append(float(val))
                weights.append(w)

        latest_val = pd.to_numeric(pd.Series([row[latest_col]]), errors="coerce").iloc[0]
        if pd.notna(latest_val):
            values.append(float(latest_val))
            weights.append(latest_weight)

        if not values:
            return np.nan

        weights = np.array(weights, dtype=float)
        weights = weights / weights.sum()

        return np.dot(values, weights)

    df[out_col] = df.apply(row_weighted_score, axis=1)

    return df


def valor_presente_descontado(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    raise NotImplementedError("Falta migrar esta función desde la notebook.")


def media_bayesiana(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    raise NotImplementedError("Falta migrar esta función desde la notebook.")


def ponderacion_dinamica(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    raise NotImplementedError("Falta migrar esta función desde la notebook.")


def momentum_2_temporadas(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    raise NotImplementedError("Falta migrar esta función desde la notebook.")


def consolidar_scores(
    dfs_posiciones: dict[str, pd.DataFrame],
    method: str = "promedio_ponderado",
) -> dict[str, pd.DataFrame]:
    method_map = {
        "promedio_ponderado": promedio_ponderado_dinamico,
        "valor_presente": valor_presente_descontado,
        "media_bayesiana": media_bayesiana,
        "ponderacion_dinamica": ponderacion_dinamica,
        "momentum": momentum_2_temporadas,
    }

    if method not in method_map:
        raise ValueError(
            "method must be one of: "
            "'promedio_ponderado', 'valor_presente', 'media_bayesiana', "
            "'ponderacion_dinamica', 'momentum'"
        )

    funcion_metodo = method_map[method]

    resultados = {}
    for posicion, df_pos in dfs_posiciones.items():
        resultados[posicion] = funcion_metodo(df_pos)

    return resultados


# =============================================================================
# LIMPIEZA DE TEXTO
# =============================================================================

def fix_mojibake(s: str):
    if pd.isna(s):
        return s
    try:
        return s.encode("latin1").decode("utf-8")
    except UnicodeError:
        try:
            return s.encode("cp1252").decode("utf-8")
        except UnicodeError:
            return s


def fix_mojibake_and_known_replacements(s: str, mapping: dict[str, str]):
    if pd.isna(s):
        return s

    try:
        s2 = s.encode("latin1").decode("utf-8")
    except UnicodeError:
        try:
            s2 = s.encode("cp1252").decode("utf-8")
        except UnicodeError:
            s2 = s

    return mapping.get(s2, s2)


def limpiar_texto_columna(serie: pd.Series, use_known_fixes: bool = False) -> pd.Series:
    if use_known_fixes:
        return serie.apply(lambda x: fix_mojibake_and_known_replacements(x, KNOWN_FIXES))
    return serie.apply(fix_mojibake)


# =============================================================================
# HELPERS PARA RECONSTRUIR NOMBRES
# =============================================================================

def get_suffix_from_col(col_name: str) -> int:
    match = re.search(r"_(\d{2})$", col_name)
    if not match:
        return -1
    return int(match.group(1))


def add_player_name_column_dynamic(
    df: pd.DataFrame,
    position: str | None = None,
    prefix: str = "player_name_",
    new_col: str = "player_name",
    insert_pos: int = 1,
    raise_on_mismatch: bool = True,
) -> pd.DataFrame:
    """
    Reconstruye una sola columna player_name a partir de todas las columnas
    player_name_XX que existan dinámicamente.
    """
    df = df.copy()

    player_name_cols = [col for col in df.columns if col.startswith(prefix)]
    if not player_name_cols:
        return df

    player_name_cols = sorted(player_name_cols, key=get_suffix_from_col)

    # Limpiar antes de comparar
    for col in player_name_cols:
        df[col] = limpiar_texto_columna(
            df[col],
            use_known_fixes=(position == "del"),
        )

    if raise_on_mismatch and len(player_name_cols) > 1:
        base_col = player_name_cols[0]

        for other_col in player_name_cols[1:]:
            mismatch_mask = (
                df[base_col].notna()
                & df[other_col].notna()
                & (df[base_col] != df[other_col])
            )

            if mismatch_mask.any():
                mismatches = df.loc[mismatch_mask, [base_col, other_col]].head(10)
                raise ValueError(
                    f"Se encontraron diferencias entre {base_col} y {other_col}.\n"
                    f"Ejemplos:\n{mismatches}"
                )

    player_name = pd.Series(pd.NA, index=df.index, dtype="object")
    for col in player_name_cols:
        player_name = player_name.combine_first(df[col])

    if new_col in df.columns:
        df.drop(columns=[new_col], inplace=True)

    safe_insert_pos = min(insert_pos, len(df.columns))
    df.insert(safe_insert_pos, new_col, player_name)

    df.drop(columns=player_name_cols, inplace=True, errors="ignore")

    return df


def limpiar_columnas_team_name_dinamicas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia dinámicamente todas las columnas team_name_XX que existan.
    """
    df = df.copy()

    team_name_cols = [col for col in df.columns if col.startswith("team_name_")]
    for col in team_name_cols:
        df[col] = limpiar_texto_columna(df[col], use_known_fixes=False)

    return df


def limpiar_score_resultante(
    df: pd.DataFrame,
    posicion: str | None = None,
    raise_on_mismatch: bool = True,
) -> pd.DataFrame:
    """
    Limpia columnas de nombres/equipos dinámicamente:
    - limpia player_name_XX y team_name_XX
    - crea player_name
    - elimina player_name_XX
    """
    df = df.copy()

    df = limpiar_columnas_team_name_dinamicas(df)

    df = add_player_name_column_dynamic(
        df=df,
        position=posicion,
        prefix="player_name_",
        new_col="player_name",
        insert_pos=1,
        raise_on_mismatch=raise_on_mismatch,
    )

    return df


def limpiar_nombres(
    resultados_por_posicion: dict[str, pd.DataFrame],
    raise_on_mismatch: bool = True,
) -> dict[str, pd.DataFrame]:
    resultados_limpios = {}

    for posicion, df in resultados_por_posicion.items():
        df_limpio = limpiar_score_resultante(
            df=df,
            posicion=posicion,
            raise_on_mismatch=raise_on_mismatch,
        )
        resultados_limpios[posicion] = df_limpio

    return resultados_limpios


# =============================================================================
# GUARDADO
# =============================================================================

def guardar_scores_consolidados_limpios(
    resultados_por_posicion: dict[str, pd.DataFrame],
    method: str,
    base_dir: Path = OUTPUT_DIR,
) -> None:
    if method not in METHOD_FOLDER_MAP:
        raise ValueError(
            "method must be one of: "
            "'promedio_ponderado', 'valor_presente', 'media_bayesiana', "
            "'ponderacion_dinamica', 'momentum'"
        )

    output_folder = base_dir
    output_folder.mkdir(parents=True, exist_ok=True)

    suffix = METHOD_SUFFIX_MAP[method]

    for posicion, df in resultados_por_posicion.items():
        output_file = output_folder / f"score_{posicion}_final_{suffix}.csv"
        df.to_csv(output_file, index=False)
        print(f"💾 Guardado: {output_file}")


# =============================================================================
# PIPELINE COMPLETO
# =============================================================================

def run_score_consolidation(method: str = "promedio_ponderado") -> dict[str, pd.DataFrame]:
    dfs_posiciones = build_score_datasets()

    scores_consolidados = consolidar_scores(
        dfs_posiciones=dfs_posiciones,
        method=method,
    )

    scores_consolidados_limpios = limpiar_nombres(
        scores_consolidados,
        raise_on_mismatch=True,
    )

    guardar_scores_consolidados_limpios(
        resultados_por_posicion=scores_consolidados_limpios,
        method=method,
    )

    return scores_consolidados_limpios


if __name__ == "__main__":
    metodo_seleccionado = config.SCORE_CONSOLIDATION_METHOD

    metodos_validos = {
        "promedio_ponderado",
        "valor_presente",
        "media_bayesiana",
        "ponderacion_dinamica",
        "momentum",
    }

    if metodo_seleccionado not in metodos_validos:
        raise ValueError(
            f"Método inválido en config.SCORE_CONSOLIDATION_METHOD: {metodo_seleccionado}"
        )

    resultados = run_score_consolidation(method=metodo_seleccionado)

    print(f"\nMétodo seleccionado: {metodo_seleccionado}")
    print("Resumen final:")
    for posicion, df in resultados.items():
        print(f"{posicion}: {df.shape}")