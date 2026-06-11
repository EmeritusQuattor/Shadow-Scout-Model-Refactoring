# Scout-AI-Training — FTC Match Predictor

## Problemas Encontrados y Soluciones

---

### 1. Dependencias de JavaFX inexistentes en pom.xml

```
spring-boot-h2console       → NO EXISTE, borrar
spring-boot-starter-webmvc  → NO EXISTE, usar spring-boot-starter-web
spring-boot-starter-*-test  → NO EXISTEN, usar spring-boot-starter-test
maven-compiler-plugin extra → innecesario (lo maneja spring-boot-starter-parent)
```

### 2. Java version mismatch

- pom.xml pedía Java 21 pero el JDK instalado era 20.
- **Fix**: cambiar `<java.version>` en pom.xml o instalar JDK correcto.

### 3. Recursión infinita en `deposit()`

```java
// MAL
public double deposit(double amount) {
    if (amount > 0) return deposit(amount);  // se llama a sí mismo
    return amount;
}
```
- **Fix**: `this.balance += amount`

### 4. `this.balance += balance` en vez de `this.balance += amount`

Error clásico de confundir el parámetro `amount` con el field `balance`. Repetido en `withdraw()`.

### 5. `withdraw()` sin validación de saldo

Permitía retirar más del balance disponible.
- **Fix**: agregar `balance >= amount`

### 6. `transfer()` sin parámetro destino

```java
public double transfer(double amount)  // ¿a dónde?
```
- **Fix**: `public void transfer(BankAccount destination, double amount)`

### 7. `getText()` con argumentos incorrectos

```java
textField.getText(150, 200)  // NO existe ese método así
```
- **Fix**: `textField.getText()`

### 8. `setOnAction()` encadenado dentro de `add()`

```java
vBox.getChildren().add(new Button("Deposit").setOnAction(...));
// setOnAction devuelve void, no Button
```
- **Fix**: crear el Button en variable, setear acción, después agregar.

### 9. `try-catch` fuera del `setOnAction`

El catch rodeaba el `setOnAction()` en vez de estar **dentro** de la lambda.
La excepción ocurre al clickear, no al registrar el evento.

### 10. `catch` con `throw new RuntimeException()`

Anulaba el propósito del catch — la app igual crasheaba.
- **Fix**: mostrar mensaje de error o simplemente no hacer nada.

### 11. GraphQL field names incorrectos

```graphql
endgamePoints  → NO existe en MatchScores2025
penaltyPoints  → NO existe, usar penaltyPointsByOpp
```
Campos disponibles: `totalPoints`, `autoPoints`, `dcPoints`, `totalPointsNp`

### 12. Rolling averages sin mejora significativa

Agregar promedios móviles de últimos 5 partidos no subió la accuracy (~0% mejora).
Los partidos están ordenados por match_id (no cronológico entre eventos), lo que limita la utilidad.

### 13. Features individuales por equipo empeoraron

Separar features por equipo (team1, team2, team3, team4) en vez de sumadas por alianza **bajó** la accuracy de 76.68% → 74.90%.
El orden de teams es arbitrario (sorted por número), el modelo no aprende patrones significativos por "slot".

### 14. Límite del API de FTCScout

- 1900 eventos disponibles, solo recolectamos 200 (~35 min)
- Los `quickStats` son promedios de temporada, no rendimiento reciente
- No hay data de penalidades por equipo en el schema GraphQL

### 15. Accuracy estancada en ~76-77%

| Config | Accuracy |
|---|---|
| Solo stats sumadas (726 samples) | 67.12% |
| + 5057 samples | 75.00% |
| + ELO features | 75.00% |
| + OPR/DPR | 76.68% |
| + Rolling averages | 76.48% |
| + Match level (quals/finals) | ~76% |
| Features individuales (por team) | 74.90% |

**Causa raíz**: FTC tiene alta varianza (penalties, fallos mecánicos, errores de driver).
Para pasar de 77% se necesita:
- Data de capacidades específicas de robots
- Historial de penalidades por equipo
- Modelo que capture contexto del evento (rankings, alianzas específicas)

### 16. Regresor (score prediction) con MAE alto

- MAE: ~77 puntos (en partidos que promedian ~150-250 pts)
- El modelo no puede predecir scores exactos porque no sabe **qué robot específico** hace qué en el campo.

### 17. NPU no usable para XGBoost

XGBoost/sklearn corren solo en CPU. NPU (Neural Processing Unit) solo sirve para deep learning (PyTorch/TensorFlow via DirectML), no para modelos tabulares.

---

## Stack final

- **Python 3.12** + XGBoost + scikit-learn
- Flask API con endpoints `/predict` y `/health`
- Text generation de estrategias (template-based)
- ELO rating + OPR/DPR como features
- ~5057 matches de 200 eventos FTC 2025
