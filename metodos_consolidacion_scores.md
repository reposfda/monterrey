## Métodos para unificar scores entre temporadas

Cuando un jugador tiene score en más de una temporada, el objetivo no es solo “promediar”, sino estimar **su nivel actual** usando información histórica. Cada método resuelve eso con una lógica distinta, y la elección depende de qué se quiera priorizar: actualidad, estabilidad, robustez o sensibilidad a la trayectoria. La comparación siguiente resume sus principales ventajas, limitaciones y casos de uso. :contentReference[oaicite:0]{index=0}

### 1. Promedio ponderado simple
Combina los scores de distintas temporadas asignándole mayor peso a la más reciente.

**Ventajas**
- Es el método más simple, transparente e intuitivo.
- Fácil de explicar a perfiles no técnicos.
- Funciona bien cuando el objetivo principal es representar el **nivel actual** del jugador.
- Mantiene buena interpretabilidad futbolística: “lo último que hizo importa más, pero el año anterior todavía aporta contexto”.

**Desventajas**
- Usa el mismo criterio para todos los jugadores, aunque algunos sean más estables y otros más irregulares.
- No distingue entre un score construido con muchos minutos y otro con poca muestra.
- Puede subestimar trayectorias muy marcadas de mejora o declive.

**Interpretación futbolística**
Es ideal cuando se quiere una mirada práctica y ejecutiva del jugador: el último año manda, pero el anterior evita sobre-reaccionar a un solo rendimiento.

**Cuándo usarlo**
- Informes para toma de decisión rápida.
- Modelos donde la interpretabilidad importa mucho.
- Contextos con pocas temporadas disponibles.
- Cuando se busca un equilibrio razonable entre actualidad y consistencia.

---

### 2. Valor presente / descuento exponencial
Aplica una lógica similar a finanzas: el pasado sigue importando, pero pierde valor a medida que se aleja en el tiempo.

**Ventajas**
- Tiene una justificación matemática muy sólida.
- Refuerza la idea de que el rendimiento reciente es más valioso que el antiguo.
- Es elegante conceptualmente para presentar el score como una estimación del “valor presente” del jugador.

**Desventajas**
- En la práctica, si el hiperparámetro se calibra parecido al promedio ponderado, los resultados pueden ser muy similares.
- Sigue sin corregir por tamaño de muestra si no se incorporan minutos o partidos.
- Puede ser menos intuitivo para usuarios no acostumbrados a conceptos de descuento temporal.

**Interpretación futbolística**
Representa bien una lógica de scouting o mercado: lo que el jugador hizo hace dos años sirve, pero pesa menos que lo que mostró recientemente.

**Cuándo usarlo**
- Cuando se quiere una narrativa más formal o financiera del modelo.
- Cuando el cliente valora una lógica explícita de depreciación de la información pasada.
- Cuando interesa enfatizar que el score es una estimación del estado actual, no un promedio histórico.

---

### 3. Media bayesiana / shrinkage
Ajusta el score observado hacia un valor de referencia del rol o posición, especialmente cuando hay poca evidencia.

**Ventajas**
- Es el método más robusto cuando hay jugadores con pocos minutos o una sola temporada.
- Reduce el riesgo de sobrevalorar outliers construidos con poca muestra.
- Permite incorporar la calidad de la evidencia usando minutos jugados.
- Da rankings más estables y menos sensibles al ruido.

**Desventajas**
- Es menos intuitivo que los métodos anteriores.
- Requiere definir un promedio de referencia del rol (`MU_POSITION`) y una fuerza de prior (`K_PRIOR`).
- Si el prior es demasiado fuerte, puede “aplanar” demasiado las diferencias entre jugadores.

**Interpretación futbolística**
Dice, en el fondo: “si todavía vi poco de este jugador, no tomo su score al pie de la letra; lo acerco al nivel típico de su posición hasta tener más evidencia”.

**Cuándo usarlo**
- Cuando hay mucha heterogeneidad en minutos jugados.
- Cuando abundan jugadores con una sola temporada.
- Cuando se quiere un modelo más prudente y robusto.
- Muy útil en análisis de proyección, scouting o mercados con muestras desbalanceadas.

---

### 4. Ponderación dinámica según estabilidad
Hace que el peso de la última temporada cambie según cuán estable o volátil haya sido el jugador.

**Ventajas**
- Reconoce que no todos los jugadores deben tratarse igual.
- Permite que jugadores muy irregulares sean evaluados más por su versión reciente.
- Es una forma más “inteligente” de ponderar sin abandonar la interpretabilidad.

**Desventajas**
- Exige definir una medida de estabilidad o volatilidad.
- Puede ser más difícil de explicar y auditar.
- Con pocas temporadas, la estimación de estabilidad puede ser frágil.

**Interpretación futbolística**
Si un jugador fue consistente a lo largo del tiempo, tiene sentido confiar más en su tendencia general. Si fue muy cambiante, conviene mirar sobre todo su forma más reciente.

**Cuándo usarlo**
- Cuando se quiere personalizar la ponderación por jugador.
- Cuando hay suficiente historial para medir estabilidad.
- En contextos donde la irregularidad es una dimensión relevante del análisis.

---

### 5. Método de tendencia / momentum
No solo combina niveles pasados, sino que intenta capturar la dirección del rendimiento: mejora, estancamiento o caída.

**Ventajas**
- Incorpora la trayectoria del jugador, no solo su promedio.
- Puede detectar perfiles en crecimiento antes de que exploten plenamente.
- Aporta valor cuando se dispone de varias temporadas.

**Desventajas**
- Con solo dos temporadas, la tendencia puede ser demasiado sensible.
- Puede sobre-premiar rachas cortas o castigar caídas transitorias.
- Requiere más cautela para no confundir forma reciente con nivel estructural.

**Interpretación futbolística**
Es el enfoque más útil cuando interesa responder no solo “qué nivel tiene hoy”, sino también “hacia dónde va”. Un jugador joven en clara progresión y uno veterano en declive pueden tener promedios similares, pero lecturas futbolísticas muy distintas.

**Cuándo usarlo**
- Cuando hay al menos 3 a 5 temporadas.
- En evaluaciones de proyección, reventa o planificación deportiva.
- Cuando la dirección de la curva de rendimiento importa tanto como el nivel actual.

---

## Recomendación práctica de uso

- **Promedio ponderado simple**: mejor opción si se busca claridad, estabilidad e interpretabilidad.
- **Valor presente / descuento exponencial**: alternativa conceptualmente fuerte cuando se quiere presentar el score como una estimación del valor actual.
- **Media bayesiana**: mejor opción cuando hay diferencias grandes en cantidad de minutos o evidencia disponible.
- **Ponderación dinámica**: útil si se quiere adaptar la importancia de la última temporada según la estabilidad del jugador.
- **Tendencia / momentum**: recomendable cuando hay varias temporadas y se quiere capturar evolución, no solo nivel.

## Criterio general de decisión

En términos futbolísticos, la elección depende de qué pregunta se quiera contestar:

- Si la pregunta es **“qué tan bueno es hoy este jugador”**, conviene priorizar métodos con más peso en la última temporada.
- Si la pregunta es **“qué tan confiable es ese nivel observado”**, conviene usar enfoques bayesianos o ajustados por minutos.
- Si la pregunta es **“hacia dónde está evolucionando”**, conviene incorporar tendencia.
- Si la prioridad es **explicar fácilmente el modelo al cliente**, el promedio ponderado simple sigue siendo la opción más fuerte.

## Conclusión

No existe un único método “correcto” para todos los contextos. Cada alternativa refleja una forma distinta de entender el rendimiento: como nivel actual, como valor presente, como evidencia sujeta a incertidumbre, o como trayectoria. Por eso, la elección debe alinearse con el objetivo analítico y con la lectura futbolística que se quiera privilegiar. En este trabajo, el promedio ponderado simple fue elegido como método principal por su equilibrio entre solidez matemática, interpretabilidad y facilidad de implementación. :contentReference[oaicite:1]{index=1}