# Scout-AI-Training — FTC Match Predictor

## Problemas a Corregir

### 1. Accuracy estancada en ~76-77%

| Configuración | Accuracy |
|---|---|
| Stats básicas (726 samples) | 67.12% |
| + 5057 samples | 75.00% |
| + ELO | 75.00% |
| + OPR/DPR | 76.68% |
| + Rolling averages | 76.48% |
| + Match level (qual/finals) | ~76% |

Para romper el techo se necesita:
- Data de capacidades específicas de robots (puede colgarse? velocidad? consistencia?)
- Historial de penalidades por equipo
- Contexto del evento (rankings actuales, composición de alianzas)

### 2. Rolling averages sin mejora real

Los promedios móviles no mejoraron la accuracy porque los partidos no están ordenados cronológicamente entre eventos (solos por match_id dentro de cada evento). Habría que agrupar por evento y ordenar por match_num para que tengan sentido temporal.

### 3. Features individuales por equipo empeoraron el modelo

Separar features por cada uno de los 4 equipos (en vez de sumar por alianza) bajó la accuracy de 76.68% → 74.90%. El orden de los equipos es arbitrario (sorted por número de equipo), por lo que el modelo no aprende patrones significativos por posición.

### 4. Regresor con MAE alto (~77 pts)

El score prediction tiene un error absoluto medio de ~77 puntos en partidos que promedian 150-250 pts. No es útil para predicción de scores exactos porque no sabemos qué robot específico contribuye qué en cada alianza.

### 5. Limitaciones del API de FTCScout

- Solo 200 eventos recolectados de 1900 disponibles
- `quickStats` son promedios de temporada, no rendimiento reciente
- No expone data de penalidades por equipo
- Campos `endgamePoints` y `penaltyPoints` no están disponibles en `MatchScores2025`

### 6. NPU no utilizable

XGBoost y scikit-learn corren solo en CPU. Para usar la NPU se necesitaría migrar a PyTorch/TensorFlow con modelos de deep learning, lo cual no es ideal para datos tabulares.

### 7. Sin validación temporal (data leakage)

Los datos se dividen con train_test_split aleatorio. Para una evaluación más realista, habría que separar por tiempo (entrenar con eventos pasados, predecir eventos futuros) y no mezclar partidos del mismo evento en train y test.
