# utils/role_config.py
from __future__ import annotations

from pathlib import Path
import ast

# =========================
# MACRO: categorías del rol (Score_*)
# =========================
ROLE_MACRO = {
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
