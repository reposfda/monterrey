# scoring/__init__.py
# -*- coding: utf-8 -*-
"""
Sistema de Scoring por Posición.

Este módulo proporciona scorers para todas las posiciones del fútbol,
con arquitectura OOP y funciones de compatibilidad legacy.

Uso básico:
-----------
>>> from scoring import DelanteroScorer
>>> scorer = DelanteroScorer(min_minutes=450)
>>> df_scored = scorer.score(df=my_dataframe)

Uso legacy (compatible con código viejo):
------------------------------------------
>>> from scoring import run_delantero_scoring
>>> df_scored = run_delantero_scoring(df=my_dataframe, min_minutes=450)

Posiciones disponibles:
-----------------------
- Delantero (striker)
- Extremo (winger)
- Interior/Mediapunta (attacking midfielder)
- Volante (defensive midfielder)
- Lateral (fullback)
- Zaguero/Defensor Central (center back)
- Golero (goalkeeper)
"""

from .base import PositionScorer
from .io import read_input

# Forwards
from .forwards import (
    DelanteroScorer,
    ExtremoScorer,
    run_delantero_scoring,
    run_extremo_scoring,
)

# Midfielders
from .midfielders import (
    VolanteScorer,
    InteriorScorer,
    run_volante_scoring,
    score_volante_df,  # Alternative name
    run_interior_scoring,
)

# Defenders
from .defenders import (
    DefensorCentralScorer,
    LateralScorer,
    run_cb_scoring,
    run_position_scoring,  # Lateral legacy name
    score_lateral_df,  # Alternative name
)

# Goalkeeper
from .goalkeeper import (
    GoalkeeperScorer,
    run_goalkeeper_scoring,
)

# Public API
__all__ = [
    # Base classes
    "PositionScorer",
    "read_input",
    
    # Scorer classes (OOP interface)
    "DelanteroScorer",
    "ExtremoScorer",
    "VolanteScorer",
    "InteriorScorer",
    "DefensorCentralScorer",
    "LateralScorer",
    "GoalkeeperScorer",
    
    # Legacy functions (backward compatibility)
    "run_delantero_scoring",
    "run_extremo_scoring",
    "run_volante_scoring",
    "score_volante_df",
    "run_interior_scoring",
    "run_cb_scoring",
    "run_position_scoring",
    "score_lateral_df",
    "run_goalkeeper_scoring",
]

__version__ = "2.0.0"
