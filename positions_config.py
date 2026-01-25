# -*- coding: utf-8 -*-
"""
Configuración de posiciones para análisis de jugadores
Mapea posiciones StatsBomb a grupos amplios
"""

from __future__ import annotations

# Mapeo de posiciones StatsBomb a grupos amplios
POSITION_MAP = {
    # GOLERO
    "Goalkeeper": "Golero",

    # ZAGUEROS
    "Center Back": "Zaguero",
    "Left Center Back": "Zaguero",
    "Right Center Back": "Zaguero",

    # LATERALES
    "Left Back": "Lateral",
    "Right Back": "Lateral",
    "Left Wing Back": "Lateral",
    "Right Wing Back": "Lateral",

    # VOLANTES (base / mixtos)
    "Center Defensive Midfield": "Volante",
    "Left Defensive Midfield": "Volante",
    "Right Defensive Midfield": "Volante",

    # INTERIORES / MEDIAPUNTAS
    "Left Center Midfield": "Interior",
    "Right Center Midfield": "Interior",
    "Center Attacking Midfield": "Interior",

    # EXTREMOS
    "Left Wing": "Extremo",
    "Right Wing": "Extremo",
    "Left Midfield": "Extremo",
    "Right Midfield": "Extremo",

    # DELANTEROS
    "Center Forward": "Delantero",
    "Left Center Forward": "Delantero",
    "Right Center Forward": "Delantero",
}

# Grupos organizados (invertido de POSITION_MAP)
POSITION_GROUPS = {}
for sb_pos, group in POSITION_MAP.items():
    POSITION_GROUPS.setdefault(group, set()).add(sb_pos)

# Grupos válidos
VALID_GROUPS = set(POSITION_GROUPS.keys())


def normalize_group(group: str) -> str:
    """
    Valida y normaliza el nombre del grupo.
    
    Args:
        group: Nombre del grupo (ej: "Lateral", "Zaguero")
        
    Returns:
        Nombre del grupo validado
        
    Raises:
        ValueError: Si el grupo no es válido
    """
    g = (group or "").strip()
    if g not in VALID_GROUPS:
        raise ValueError(f"Grupo inválido: '{group}'. Válidos: {sorted(VALID_GROUPS)}")
    return g


def sb_positions_for(group: str) -> set[str]:
    """
    Retorna todas las posiciones StatsBomb que pertenecen a un grupo.
    
    Args:
        group: Nombre del grupo (ej: "Lateral")
        
    Returns:
        Set de posiciones StatsBomb del grupo
        
    Example:
        >>> sb_positions_for("Lateral")
        {'Left Back', 'Right Back', 'Left Wing Back', 'Right Wing Back'}
    """
    g = normalize_group(group)
    return POSITION_GROUPS[g]


def get_group_for_position(position: str) -> str | None:
    """
    Obtiene el grupo al que pertenece una posición StatsBomb.
    
    Args:
        position: Posición StatsBomb (ej: "Left Back")
        
    Returns:
        Grupo al que pertenece o None si no existe
        
    Example:
        >>> get_group_for_position("Left Back")
        'Lateral'
    """
    return POSITION_MAP.get(position)


def list_all_groups() -> list[str]:
    """
    Lista todos los grupos disponibles ordenados alfabéticamente.
    
    Returns:
        Lista de nombres de grupos
    """
    return sorted(VALID_GROUPS)


if __name__ == "__main__":
    # Testing
    print("="*70)
    print("GRUPOS DE POSICIONES DISPONIBLES")
    print("="*70)
    
    for group in list_all_groups():
        positions = sb_positions_for(group)
        print(f"\n{group}:")
        for pos in sorted(positions):
            print(f"  - {pos}")
