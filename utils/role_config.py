# utils/role_config.py
from __future__ import annotations

from pathlib import Path
import ast
from typing import Dict, List

# =========================
# MACRO: categorías del rol (Score_*)
# =========================
ROLE_MACRO = {
    "Golero": [
        ("Score_Effectiveness", "Efectividad"),
        ("Score_Area_Domination", "Dominio del área"),
        ("Score_Foot_Play", "Juego con los pies"),
        ("Score_Outside_Box", "Acciones fuera del área"),
    ],

    "Zaguero": [
        ("Score_ControlDefensivo", "Control Defensivo"),
        ("Score_AccionDefensiva", "Acción Defensiva"),
        ("Score_Progresion", "Progresión"),
        ("Score_ImpactoOfensivo", "Impacto Ofensivo"),
    ],
    "Lateral": [
        ("Score_Defensivo", "Defensivo"),
        ("Score_Presion", "Presión"),
        ("Score_Profundidad", "Profundidad"),
        ("Score_Calidad", "Calidad"),
    ],
    "Volante": [
        ("Score_Posesion", "Posesión"),
        ("Score_Progresion", "Progresión"),
        ("Score_Territoriales", "Territoriales"),
        ("Score_Contencion", "Contención"),
    ],
    "Interior/Mediapunta": [
        ("Score_Organizacion", "Organización"),
        ("Score_BoxToBox", "Box to Box"),
        ("Score_Desequilibrio", "Desequilibrio"),
        ("Score_ContencionPresion", "Contención/Presión"),
    ],
    "Extremo": [
        ("Score_ZonaInfluencia", "Zona de influencia"),
        ("Score_Desequilibrio", "Desequilibrio"),
        ("Score_Finalizacion", "Finalización"),
        ("Score_CompromisoDef", "Compromiso defensivo"),
    ],
    "Delantero": [
        ("Score_Finalizacion", "Finalización"),
        ("Score_Presionante", "Presionante"),
        ("Score_Conector", "Conector"),
        ("Score_Disruptivo", "Disruptivo"),
    ],
}

# =========================
# DETALLADO: nombres de listas dentro de cada script
# (se parsean desde el .py, así evitás duplicar las listas)
# =========================
ROLE_DETAIL_SPECS = {
    "Golero": {
        "file": "position_scoring_golero.py",
        "cats": {
            "Efectividad": "EFFECTIVENESS",
            "Dominio del área": "AREA_DOMINATION",
            "Juego con los pies": "FOOT_PLAY",
            "Acciones fuera del área": "OUTSIDE_BOX",
        },
    },

    "Zaguero": {
        "file": "position_scoring_defensor_central.py",
        "cats": {
            "Control Defensivo": "CONTROL_DEF",
            "Acción Defensiva": "ACCION_DEF",
            "Progresión": "PROGRESION",
            "Impacto Ofensivo": "OFENSIVO",
        },
    },
    "Lateral": {
        "file": "position_scoring_lateral.py",
        "cats": {
            "Defensivo (Exec)": "DEF_EXEC",
            "Defensivo (OBV)": "DEF_OBV",
            "Presión": "PRESS",
            "Profundidad": "DEPTH",
            "Calidad": "QUALITY",
        },
    },
    "Volante": {
        "file": "position_scoring_volante.py",
        "cats": {
            "Posesión": "POSESION",
            "Progresión": "PROGRESION",
            "Territoriales": "TERRITORIALES",
            "Contención": "CONTENCION",
        },
    },
    "Interior/Mediapunta": {
        "file": "position_scoring_interior.py",
        "cats": {
            "Organización": "ORGANIZACION",
            "Box to Box": "BOX_TO_BOX",
            "Desequilibrio": "DESEQUILIBRIO",
            "Contención/Presión": "CONTENCION_PRESION",
        },
    },
    "Extremo": {
        "file": "position_scoring_extremos.py",
        "cats": {
            "Zona de influencia": "ZONA_INFLUENCIA",
            "Desequilibrio": "DESEQUILIBRIO",
            "Finalización": "FINALIZACION",
            "Compromiso defensivo": "COMPROMISO_DEF",
        },
    },
    "Delantero": {
        "file": "position_scoring_delantero.py",
        "cats": {
            "Finalización": "FINALIZACION",
            "Presionante": "PRESIONANTE",
            "Conector": "CONECTOR",
            "Disruptivo": "DISRUPTIVO",
        },
    },
}


def _extract_list_literal(py_text: str, varname: str):
    """
    Busca 'VARNAME = [ ... ]' y devuelve la lista (via ast.literal_eval).
    """
    key = f"{varname} = ["
    idx = py_text.find(key)
    if idx == -1:
        return None

    start = py_text.find("[", idx)
    depth = 0
    end = None
    for i in range(start, len(py_text)):
        ch = py_text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end is None:
        return None

    literal = py_text[start:end]
    return ast.literal_eval(literal)


def get_macro_config(position_key: str):
    """
    Devuelve lista de (col_score, label) para el radar macro.
    """
    return ROLE_MACRO.get(position_key, [])


def get_detail_categories(position_key: str):
    """
    Devuelve labels disponibles para el radar detallado (según scripts).
    """
    spec = ROLE_DETAIL_SPECS.get(position_key)
    if not spec:
        return []
    return list(spec["cats"].keys())


def get_detail_metric_list(position_key: str, detail_label: str, base_dir: Path):
    """
    Lee el script de scoring correspondiente y devuelve la lista de métricas:
    [(metric, weight, invert), ...]
    """
    spec = ROLE_DETAIL_SPECS.get(position_key)
    if not spec:
        return None

    varname = spec["cats"].get(detail_label)
    if not varname:
        return None

    script_path = base_dir / spec["file"]
    if not script_path.exists():
        return None

    txt = script_path.read_text(encoding="utf-8", errors="ignore")
    return _extract_list_literal(txt, varname)

# --------------------
# Pesos de categorías (exactos)
# --------------------
CATEGORY_WEIGHTS_BY_POSITION: Dict[str, Dict[str, float]] = {

    "Golero": {
        "Score_Effectiveness": 0.50,
        "Score_Area_Domination": 0.20,
        "Score_Foot_Play": 0.15,
        "Score_Outside_Box": 0.15,
    },

    # Zaguero  -> position_scoring_defensor_central.py (CAT_W)
    "Zaguero": {
        "Score_AccionDefensiva": 0.25,
        "Score_ControlDefensivo": 0.45,
        "Score_Progresion": 0.20,
        "Score_ImpactoOfensivo": 0.10,
    },

    # Lateral -> position_scoring_lateral.py (CAT_W)
    "Lateral": {
        "Score_Profundidad": 0.30,
        "Score_Calidad": 0.30,
        "Score_Presion": 0.20,
        "Score_Defensivo": 0.20,
    },

    # Volante -> position_scoring_volante.py (CAT_W)
    "Volante": {
        "Score_Posesion": 0.25,
        "Score_Progresion": 0.30,
        "Score_Territoriales": 0.25,
        "Score_Contencion": 0.20,
    },

    # Interior/Mediapunta -> position_scoring_interior.py (CAT_W)
    "Interior/Mediapunta": {
        "Score_BoxToBox": 0.25,
        "Score_Desequilibrio": 0.30,
        "Score_Organizacion": 0.25,
        "Score_ContencionPresion": 0.20,
    },

    # Extremo -> position_scoring_extremos.py (CAT_W)
    "Extremo": {
        "Score_CompromisoDef": 0.20,
        "Score_Desequilibrio": 0.35,
        "Score_Finalizacion": 0.30,
        "Score_ZonaInfluencia": 0.15,
    },

    # Delantero -> position_scoring_delantero.py (CAT_W)
    "Delantero": {
        "Score_Finalizacion": 0.40,
        "Score_Presionante": 0.10,
        "Score_Conector": 0.25,
        "Score_Disruptivo": 0.25,
    },
}


# --------------------
# Splits internos (solo si aplica)
# --------------------
# Lateral: Defensivo = combinación interna Exec + OBV
LATERAL_DEF_SPLIT: Dict[str, float] = {
    "DEF_EXEC": 0.60,  # def_exec_w
    "DEF_OBV": 0.40,   # def_obv_w
}


# --------------------
# Helpers
# --------------------
def get_category_weights(position: str) -> Dict[str, float]:
    """Devuelve dict de pesos por categoría (Score_*) para la posición."""
    return CATEGORY_WEIGHTS_BY_POSITION.get(position, {}).copy()


def get_categories(position: str) -> List[str]:
    """Devuelve lista ordenada de categorías (Score_*) para la posición."""
    return list(get_category_weights(position).keys())


def get_lateral_def_split() -> Dict[str, float]:
    """Devuelve split interno para laterales (Exec/OBV)."""
    return LATERAL_DEF_SPLIT.copy()


def strip_score_prefix(cat: str) -> str:
    """Convierte 'Score_Progresion' -> 'Progresion'."""
    return cat.replace("Score_", "", 1)

