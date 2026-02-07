# 🏆 Monterrey Scoring App

Sistema de análisis y evaluación de rendimiento de jugadores de fútbol profesional, diseñado para asistir en la toma de decisiones estratégicas sobre contratos de futbolistas.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.54+-red.svg)
![License](https://img.shields.io/badge/License-Proprietary-gray.svg)

---

## 📋 Tabla de Contenidos

- [Descripción General](#-descripción-general)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación](#-instalación)
- [Uso de la Aplicación](#-uso-de-la-aplicación)
- [Pipeline de Datos](#-pipeline-de-datos)
- [Sistema de Scoring](#-sistema-de-scoring)
- [Configuración](#-configuración)
- [Agregar Nueva Temporada](#-agregar-nueva-temporada)

---

## 📖 Descripción General

La **Monterrey Scoring App** es una herramienta de análisis de datos deportivos que permite:

- **Evaluar jugadores** mediante un sistema de scoring por posición
- **Comparar rendimiento** entre jugadores de la liga
- **Analizar costo-beneficio** relacionando performance con inversión salarial
- **Filtrar por temporada** para análisis históricos

### Casos de uso principales

1. Decisiones de renovación de contratos
2. Evaluación de fichajes potenciales
3. Benchmarking de jugadores propios vs. liga
4. Análisis de eficiencia de inversión en plantilla

---

## 📁 Estructura del Proyecto

```
fda-mty-main/
│
├── Inicio.py                    # Página principal de la app
├── config.py                    # Configuración centralizada
├── requirements.txt             # Dependencias Python
│
├── pages/                       # Páginas de Streamlit
│   ├── 2_Scoring_Liga.py        # Ranking de jugadores por posición
│   ├── 3_Tablero_Jugadores.py   # Análisis individual de jugadores
│   └── 4_Cost_Performance.py    # Análisis costo-rendimiento
│
├── core/                        # Procesamiento de datos
│   ├── event_processor.py       # Pipeline principal de eventos
│   ├── goalkeeper_metrics.py    # Métricas específicas de arqueros
│   ├── obv_lanes.py             # Análisis de carriles OBV
│   ├── obv_zones.py             # Análisis de zonas defensivas
│   └── turnover_calculator.py   # Cálculo de turnovers
│
├── scoring/                     # Sistema de scoring por posición
│   ├── __init__.py              # Exports públicos
│   ├── base.py                  # Clase base PositionScorer
│   ├── forwards.py              # Delantero, Extremo
│   ├── midfielders.py           # Volante, Interior
│   ├── defenders.py             # Zaguero, Lateral
│   └── goalkeeper.py            # Golero
│
├── utils/                       # Utilidades
│   ├── season_manager.py        # Gestión de temporadas
│   ├── loaders.py               # Carga de datos
│   ├── filters.py               # Filtros del sidebar
│   ├── role_config.py           # Configuración de métricas por rol
│   ├── scoring_wrappers.py      # Wrappers para scoring
│   ├── radar_chart.py           # Gráficos de radar
│   ├── lollipop_chart.py        # Gráficos lollipop
│   └── metrics_labels.py        # Etiquetas de métricas
│
├── data/                        # Datos de entrada
│   ├── per90/                   # CSVs de estadísticas por temporada
│   │   ├── all_players_complete_2024_2025.csv
│   │   └── all_players_complete_2025_2026.csv
│   ├── scores/                  # Scores precalculados por posición
│   └── economica/               # Datos de costos de jugadores
│
├── outputs/                     # Archivos generados
│   └── player_minutes_by_match_YYYY_YYYY.csv
│
├── assets/                      # Recursos estáticos
│   └── monterrey_logo.png
│
└── test/                        # Scripts de testing
    ├── check_calc_main_csv.py
    └── multi_team.py
```

---

## 🚀 Instalación

### Requisitos previos

- Python 3.10 o superior
- pip (gestor de paquetes)

### Paso 1: Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd fda-mty-main
```

### Paso 2: Crear entorno virtual

**En Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**En Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Paso 3: Instalar dependencias

```bash
pip install -r requirements.txt
```

### Paso 4: Verificar instalación

```bash
streamlit --version
# Debería mostrar: Streamlit, version 1.54.0 o superior
```

---

## 🖥️ Uso de la Aplicación

### Iniciar la aplicación

```bash
streamlit run Inicio.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

### Páginas disponibles

#### 1. **Inicio** (`Inicio.py`)
Página principal con navegación y selector de temporada.

#### 2. **Scoring Liga** (`pages/2_Scoring_Liga.py`)
Ranking de jugadores filtrado por:
- **Posición**: Golero, Zaguero, Lateral, Volante, Interior, Extremo, Delantero
- **Minutos mínimos**: Filtrar jugadores con suficiente tiempo de juego
- **Equipos**: Seleccionar uno o varios equipos
- **Pesos de categorías**: Ajustar importancia de cada categoría de scoring

**Funcionalidades:**
- Top 10 por posición
- Ranking completo expandible
- Boxplot de distribución por categoría
- Highlight de jugador específico

#### 3. **Tablero Jugadores** (`pages/3_Tablero_Jugadores.py`)
Análisis individual detallado:
- Ficha técnica del jugador
- Radar chart de categorías
- Lollipop chart de métricas detalladas
- Comparación con promedio de la liga o jugador específico

#### 4. **Cost Performance** (`pages/4_Cost_Performance.py`)
Análisis de relación costo-rendimiento:
- Curva de mercado por posición
- Estimación de "precio justo" basado en performance
- Escala de rendimiento requerido por nivel salarial

### Selector de Temporada

En el sidebar de cada página aparece el selector de temporada:
- Detecta automáticamente las temporadas disponibles
- Afecta todos los datos mostrados en la aplicación
- La selección persiste al navegar entre páginas

---

## 🔄 Pipeline de Datos

### Flujo general

```
events_YYYY_YYYY.csv  →  event_processor.py  →  all_players_complete_YYYY_YYYY.csv
        ↓                                                    ↓
   (datos crudos)                                    (estadísticas per90)
                                                             ↓
                                                    Aplicación Streamlit
```

### Ejecutar el procesador de eventos

El script `core/event_processor.py` transforma datos crudos de eventos en estadísticas per90.

#### Configurar el archivo de entrada

1. Editar `config.py`:

```python
# Cambiar la ruta al archivo de eventos de la temporada deseada
EVENTS_CSV = DATA_DIR / "events_2025_2026.csv"
```

2. Asegurarse de que el archivo de eventos existe en `data/`

#### Ejecutar el procesamiento

```bash
cd core
python event_processor.py
```

#### Salidas generadas

El script genera dos archivos en `outputs/`:

| Archivo | Descripción |
|---------|-------------|
| `all_players_complete_YYYY_YYYY.csv` | Estadísticas completas per90 de todos los jugadores |
| `player_minutes_by_match_YYYY_YYYY.csv` | Minutos jugados por partido por jugador |

**Nota:** El nombre de la temporada (`YYYY_YYYY`) se extrae automáticamente del nombre del archivo de eventos.

#### Mover archivo para la app

Para que la aplicación detecte la nueva temporada, mover el CSV a `data/per90/`:

```bash
mv outputs/all_players_complete_2025_2026.csv data/per90/
```

---

## 📊 Sistema de Scoring

El sistema de scoring evalúa a cada jugador mediante un proceso de **normalización por percentiles** dentro de su posición, permitiendo comparaciones justas entre jugadores con diferentes volúmenes de participación.

### Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────────┐
│                         scoring/                                     │
├─────────────────────────────────────────────────────────────────────┤
│  base.py          │  Clase abstracta PositionScorer                 │
│                   │  - Template Method pattern                       │
│                   │  - Lógica común de cálculo                       │
├───────────────────┼─────────────────────────────────────────────────┤
│  forwards.py      │  DelanteroScorer, ExtremoScorer                 │
│  midfielders.py   │  VolanteScorer, InteriorScorer                  │
│  defenders.py     │  DefensorCentralScorer, LateralScorer           │
│  goalkeeper.py    │  GoalkeeperScorer                               │
└───────────────────┴─────────────────────────────────────────────────┘
```

### Flujo de Cálculo de Scores

```
1. FILTRADO
   └─> Jugadores de la posición con minutos >= min_minutes

2. MÉTRICAS → PERCENTILES
   └─> Cada métrica se convierte a percentil 0-100
   └─> Métricas invertidas (ej: turnovers): percentil = 100 - percentil

3. CATEGORÍAS
   └─> Promedio ponderado de percentiles dentro de cada categoría
   └─> Resultado: Score_Categoria (0-100)

4. SCORE TOTAL
   └─> Promedio ponderado de todas las categorías
   └─> Resultado: Score_Total (0-100)

5. FLAGS
   └─> Jugadores en el top 25% de cada categoría reciben flag = 1
   └─> Se genera perfil descriptivo basado en flags activos
```

### Detalle del Cálculo

#### Paso 1: Conversión a Percentiles

Cada métrica se normaliza a un rango 0-100 usando ranking percentil:

```python
# Ejemplo: goals_per90
Jugador A: 0.8 goles/90 → percentil 95 (top 5%)
Jugador B: 0.3 goles/90 → percentil 50 (mediana)
Jugador C: 0.1 goles/90 → percentil 15 (bajo)
```

Para **métricas invertidas** (donde menor es mejor, ej: turnovers):
```python
# total_turnovers_per90 (invertida)
Jugador A: 2.5 turnovers/90 → percentil bruto 80 → 100-80 = 20 (malo)
Jugador B: 0.5 turnovers/90 → percentil bruto 10 → 100-10 = 90 (bueno)
```

#### Paso 2: Cálculo de Categorías

Cada categoría es un promedio ponderado de sus métricas:

```python
# Ejemplo: Score_Finalizacion para Delantero
Score_Finalizacion = (
    xg_per_shot_pct * 0.20 +
    shot_statsbomb_xg_per90_pct * 0.18 +
    obv_total_net_type_shot_per90_pct * 0.15 +
    goals_per90_pct * 0.05 +
    touches_in_opp_box_per90_pct * 0.15 +
    ...
)
```

#### Paso 3: Score Total

El score total combina todas las categorías con sus pesos:

```python
# Ejemplo: Delantero
Score_Total = (
    Score_Finalizacion * 0.40 +
    Score_Presionante * 0.10 +
    Score_Conector * 0.25 +
    Score_Disruptivo * 0.25
)
```

### Posiciones y Categorías

#### ⚽ Delantero (Striker)

| Categoría | Peso | Métricas Principales |
|-----------|------|---------------------|
| **Finalización** | 40% | xG por disparo, xG total, toques en área rival, goles |
| **Presionante** | 10% | Presiones, contrapresión, recuperaciones ofensivas |
| **Conector** | 25% | Pases completados, pases al tercio final, asistencias |
| **Disruptivo** | 25% | OBV en regates, OBV en conducciones, progresiones |

**Perfiles posibles:**
- `Killer` - Elite en finalización
- `Presionante` - Alto trabajo defensivo
- `Falso 9` - Excelente conexión con mediocampo
- `Disruptivo` - Genera peligro con regate y conducción

---

#### 🏃 Extremo (Winger)

| Categoría | Peso | Métricas Principales |
|-----------|------|---------------------|
| **Compromiso Def** | 20% | Presiones en tercio ofensivo, contrapresión, recuperaciones |
| **Desequilibrio** | 35% | OBV en regates, conducciones, pases clave, xA |
| **Finalización** | 30% | xG, OBV en disparos, toques en área |
| **Zona Influencia** | 15% | OBV desde carril exterior/interior |

**Perfiles posibles:**
- `Compromiso Def` - Alto trabajo sin balón
- `Desequilibrio` - Crea superioridad 1v1
- `Finalización` - Perfil goleador
- `Zona Influencia` - Domina su carril

---

#### 🎯 Interior/Mediapunta

| Categoría | Peso | Métricas Principales |
|-----------|------|---------------------|
| **Box to Box** | 25% | Acciones defensivas + ofensivas combinadas |
| **Desequilibrio** | 30% | Regates, conducciones, pases al área |
| **Organización** | 25% | Pases progresivos, cambios de juego, control |
| **Contención/Presión** | 20% | Duelos, intercepciones, presión |

**Perfiles posibles:**
- `Box to Box` - Cobertura completa del campo
- `Desequilibrantes` - Creadores de ocasiones
- `Organizadores` - Directores de juego
- `Contención/Presión` - Equilibrio defensivo

---

#### 🛡️ Volante (Defensive Midfielder)

| Categoría | Peso | Métricas Principales |
|-----------|------|---------------------|
| **Posesión** | 25% | Pases completados, retención, pases bajo presión |
| **Progresión** | 25% | Pases progresivos, conducciones, cambios de juego |
| **Territoriales** | 25% | Presión, contrapresión, recuperaciones |
| **Contención** | 25% | Duelos, tackles, intercepciones, bloqueos |

**Perfiles posibles:**
- `Posesión` - Metrónomos del equipo
- `Progresión` - Generan transiciones
- `Territoriales` - Dominan espacios
- `Contención` - Escudos defensivos

---

#### 🧱 Zaguero (Center Back)

| Categoría | Peso | Métricas Principales |
|-----------|------|---------------------|
| **Acción Defensiva** | 30% | Duelos ganados, tackles, recuperaciones, despejes |
| **Control Defensivo** | 25% | OBV defensivo, presión, disparos concedidos en zona |
| **Progresión** | 25% | Pases al tercio final, cambios de juego, OBV en pases |
| **Impacto Ofensivo** | 20% | xG en jugada/pelota parada, OBV ofensivo |

**Perfiles posibles:**
- `Acción Def` - Ganadores de duelos
- `Control Def` - Lectores del juego
- `Progresión` - Salida limpia desde atrás
- `Ofensivo` - Peligrosos en área rival

---

#### 🏃‍♂️ Lateral (Fullback)

| Categoría | Peso | Métricas Principales |
|-----------|------|---------------------|
| **Profundidad** | 25% | Centros, pases al área, OBV en centros |
| **Calidad** | 25% | Precisión de pases, pases progresivos, OBV en pases |
| **Presión** | 25% | Presiones, recuperaciones ofensivas, contrapresión |
| **Defensivo** | 25% | Duelos, tackles, intercepciones, OBV defensivo |

**Perfiles posibles:**
- `Profundos` - Carrileros ofensivos
- `Técnicos` - Alta calidad con balón
- `Presionantes` - Agresivos sin balón
- `Protectores` - Sólidos defensivamente

---

#### 🧤 Golero (Goalkeeper)

| Categoría | Peso | Métricas Principales |
|-----------|------|---------------------|
| **Efectividad** | 35% | Goles prevenidos, % de atajadas, errores |
| **Dominio de Área** | 25% | Salidas aéreas, disparos en área concedidos |
| **Juego de Pies** | 25% | OBV en pases, pases largos, pases bajo presión |
| **Fuera del Área** | 15% | Acciones fuera del área, distancia agresiva |

**Perfiles posibles:**
- `Atajador` - Elite bajo los palos
- `Dominante` - Control del área
- `Juego de Pies` - Salida con balón
- `Libero` - Activo fuera del área

---

### Sistema de Flags y Perfiles

#### ¿Qué son los Flags?

Los **flags** son indicadores binarios (0 o 1) que marcan si un jugador está en el **top 25%** de su posición en una categoría específica.

```python
# Configuración por defecto
flag_q = 0.75  # Percentil 75 = top 25%

# Cálculo
threshold = df["Score_Finalizacion"].quantile(0.75)
df["flag_Score_Finalizacion"] = (df["Score_Finalizacion"] >= threshold).astype(int)
```

#### ¿Cómo se genera el Perfil?

El perfil es una cadena descriptiva que combina las etiquetas de todas las categorías donde el jugador tiene `flag = 1`:

```python
# Ejemplo para un Delantero
flag_Score_Finalizacion = 1  → "Killer"
flag_Score_Presionante = 0   → (no se incluye)
flag_Score_Conector = 1      → "Falso 9"
flag_Score_Disruptivo = 0    → (no se incluye)

# Resultado
Perfil = "Killer | Falso 9"
```

Si un jugador no tiene ningún flag activo, su perfil es `"Balanceado"`.

#### Mapeo de Categorías a Etiquetas

```python
PROFILE_LABELS = {
    "Delantero": {
        "Score_Finalizacion": "Killer",
        "Score_Presionante": "Presionante",
        "Score_Conector": "Falso 9",
        "Score_Disruptivo": "Disruptivo",
    },
    "Extremo": {
        "Score_CompromisoDef": "Compromiso Def",
        "Score_Desequilibrio": "Desequilibrio",
        "Score_Finalizacion": "Finalización",
        "Score_ZonaInfluencia": "Zona Influencia",
    },
    # ... etc para cada posición
}
```

---

### Uso Programático

#### Ejemplo básico con clase

```python
from scoring import DelanteroScorer

# Crear scorer con parámetros
scorer = DelanteroScorer(
    min_minutes=450,    # Mínimo ~5 partidos completos
    min_matches=5,      # Mínimo 5 partidos
    flag_q=0.75,        # Top 25% para flags
    verbose=True        # Mostrar progreso
)

# Aplicar scoring
df_scored = scorer.score(df=my_dataframe)

# Columnas generadas:
# - Score_Finalizacion, Score_Presionante, Score_Conector, Score_Disruptivo
# - Score_Total
# - flag_Score_Finalizacion, flag_Score_Presionante, etc.
# - flag_Total
```

#### Ejemplo con función legacy

```python
from scoring import run_delantero_scoring

df_scored = run_delantero_scoring(
    df=my_dataframe,
    min_minutes=450,
    min_matches=5,
    flag_q=0.75
)
```

#### Ejemplo generando perfiles

```python
from utils.scoring_wrappers import compute_scoring_from_df

df_scored = compute_scoring_from_df(
    df_base=my_dataframe,
    position_key="Delantero",
    min_minutes=450,
    min_matches=5,
    selected_teams=["Monterrey", "Tigres"]  # Opcional
)

# Incluye columna "Flags" con perfil descriptivo
print(df_scored[["player_name", "Score_Total", "Flags"]])
```

---

### Personalización de Métricas

Para modificar las métricas de una posición, editar el archivo correspondiente en `scoring/`:

```python
# scoring/forwards.py

class DelanteroScorer(PositionScorer):
    
    @property
    def categories(self) -> dict:
        return {
            "Score_Finalizacion": [
                # (nombre_metrica, peso, invertida)
                ("xg_per_shot", 0.20, False),
                ("goals_per90", 0.15, False),
                ("turnovers_per90", 0.10, True),  # Invertida: menos es mejor
                # Agregar nuevas métricas aquí...
            ],
            # ... otras categorías
        }
    
    @property
    def category_weights(self) -> dict:
        return {
            "Score_Finalizacion": 0.40,  # Ajustar pesos aquí
            "Score_Presionante": 0.10,
            "Score_Conector": 0.25,
            "Score_Disruptivo": 0.25,
            # Los pesos deben sumar 1.0
        }
```

Para agregar perfiles personalizados, editar `utils/scoring_wrappers.py`:

```python
PROFILE_LABELS = {
    "Delantero": {
        "Score_Finalizacion": "Killer",      # Cambiar etiqueta
        "Score_NuevaCategoria": "Mi Perfil", # Agregar nueva
    },
}
```

---

## ⚙️ Configuración

### Archivo `config.py`

Configuración centralizada del proyecto:

```python
# Paths principales
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Archivo de eventos para procesar
EVENTS_CSV = DATA_DIR / "events_2025_2026.csv"

# Colores corporativos
class Colors:
    PRIMARY_BG = "#0B1F38"   # Azul oscuro
    ACCENT = "#6CA0DC"       # Azul claro
    GOLD = "#c49308"         # Dorado

# Valores por defecto
class Defaults:
    MIN_MINUTES = 450        # Minutos mínimos
    MIN_MATCHES = 5          # Partidos mínimos
    TOP_N_RANKING = 10       # Top N en ranking
```

### Personalización de estilos

Los estilos CSS se definen en `config.py` dentro de `get_global_css()`. Para modificar la apariencia:

1. Editar los valores en la clase `Colors`
2. Modificar el CSS en `get_global_css()`
3. Reiniciar la aplicación

---

## 📅 Agregar Nueva Temporada

### Paso 1: Preparar datos de eventos

Colocar el archivo de eventos en `data/`:
```
data/events_2026_2027.csv
```

### Paso 2: Configurar el procesador

Editar `config.py`:
```python
EVENTS_CSV = DATA_DIR / "events_2026_2027.csv"
```

### Paso 3: Ejecutar procesamiento

```bash
cd core
python event_processor.py
```

### Paso 4: Mover archivo generado

```bash
mv outputs/all_players_complete_2026_2027.csv data/per90/
```

### Paso 5: Verificar en la app

La nueva temporada aparecerá automáticamente en el selector del sidebar.

---

## 🔧 Solución de Problemas

### Error: "No se encontró el archivo base per90"

**Causa:** No existe el CSV de la temporada seleccionada.

**Solución:**
1. Verificar que existe `data/per90/all_players_complete_YYYY_YYYY.csv`
2. Seleccionar otra temporada disponible
3. Ejecutar `event_processor.py` para generar el archivo

### Error: "ModuleNotFoundError"

**Causa:** Dependencias no instaladas o entorno virtual no activado.

**Solución:**
```bash
source venv/bin/activate  # Activar entorno
pip install -r requirements.txt  # Reinstalar dependencias
```

### La aplicación no detecta nuevas temporadas

**Causa:** El archivo CSV no está en la ubicación correcta.

**Solución:**
- Verificar que el archivo está en `data/per90/`
- Verificar el nombre: `all_players_complete_YYYY_YYYY.csv`
- Refrescar la página (F5)

---
