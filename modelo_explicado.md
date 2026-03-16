# SMARTUR v2 — Modelo explicado (técnico y matemático)

Este documento describe de forma completa el funcionamiento técnico y matemático del sistema de recomendación híbrido SMARTUR v2.

---

## 1. Visión general del sistema

SMARTUR es un **sistema de recomendación híbrido** que combina:

1. **Filtrado colaborativo (CF)** basado en **correlación de Pearson** y **K-Nearest Neighbors (KNN)** para explotar similitud entre usuarios.
2. **Modelo contextual (RF)** basado en **Random Forest** que predice el rating a partir de características del negocio (ubicación, categorías, popularidad).
3. **Fusión lineal** de ambas predicciones con un peso \(\alpha\) configurable.
4. **Filtrado por contexto** opcional (tipos de turismo, preferencia outdoor) antes del re-ranking.

El flujo de una recomendación es:

```
Usuario u + (opcional) contexto del formulario
    → Pool de candidatos (KNN sobre usuarios similares)
    → Filtrado por contexto (categorías Yelp ↔ tipos de turismo)
    → Para cada candidato: score_CF(u, i), score_RF(i)
    → Score final = α·score_CF + (1−α)·score_RF
    → Ordenar por score final → Top-N
```

---

## 2. Datos de entrada y preprocesamiento

### 2.1 Fuente de datos

- **Reviews**: archivo de reseñas de Yelp (`data_reviews_limpio.csv`) con columnas como `user_id`, `business_id`, `stars` (1–5).
- **Negocios**: archivo de negocios (`data_negocios_limpio.csv`) con `business_id`, `name`, `latitude`, `longitude`, `stars`, `review_count`, `is_open`, `categories`, etc.

El script `pre_procesamiento.py` filtra los JSON originales de Yelp para quedarse solo con negocios cuya columna `categories` contiene al menos uno de: `Tourism`, `Hotels`, `Restaurants`, `Local Services`. Las reseñas se filtran para incluir solo las que corresponden a esos `business_id`, con un límite configurable de filas para controlar tamaño y memoria.

### 2.2 Partición entrenamiento / prueba

Se hace un **split aleatorio 80 % / 20 %** (semilla fija `random_state=42`) sobre el dataframe de reviews:

- **Train**: ~80 % de las filas → se usa para construir la matriz usuario–ítem, entrenar KNN y entrenar el Random Forest.
- **Test**: ~20 % → se usa solo para evaluar métricas (RMSE, MAE), nunca para entrenar.

Toda la lógica de recomendación y de predicción utiliza únicamente `train_data`; `test_data` solo interviene en el script de evaluación.

---

## 3. Matriz de utilidad y centrado (Motor Pearson)

### 3.1 Matriz usuario–ítem

A partir de `train_data` se construye la **matriz de utilidad** \(R\):

- **Filas**: usuarios (indexados por `user_id`).
- **Columnas**: negocios/ítems (indexados por `business_id`).
- **Valor** \(R_{ui}\): estrellas que el usuario \(u\) dio al ítem \(i\) (1–5). Si \(u\) no calificó \(i\), la celda es **missing** (no se rellena con 0 en esta etapa).

En código se obtiene con un `pivot_table` sobre `user_id`, `business_id` y `values='stars'`, de modo que \(R\) es una matriz dispersa (muchos NaN).

### 3.2 Media por usuario

Para cada usuario \(u\) se define la **media de sus ratings** (solo sobre los ítems que sí calificó):

\[
\bar{r}_u = \frac{1}{|\mathcal{I}_u|} \sum_{i \in \mathcal{I}_u} R_{ui}
\]

donde \(\mathcal{I}_u\) es el conjunto de ítems calificados por \(u\). En implementación: `raw.mean(axis=1)` sobre la matriz cruda (los NaN se ignoran en el cálculo de la media).

### 3.3 Matriz centrada (para Pearson)

La **correlación de Pearson** entre dos usuarios se basa en las **desviaciones respecto a su media**, no en los ratings crudos. Por eso se construye la matriz centrada:

\[
\tilde{R}_{ui} = R_{ui} - \bar{r}_u \quad \text{(solo donde } R_{ui} \text{ existe)}
\]

En el código:

- Se guarda una máscara booleana `has_rating` donde hay valor real (no NaN).
- Se rellena temporalmente los NaN con 0 en una copia para poder restar: `centered = R - bar_r_u`.
- Luego se fuerza a 0 todas las celdas que originalmente eran missing: `centered[~has_rating] = 0`.

Así, en la matriz centrada:

- **Celda con rating**: \(\tilde{R}_{ui} = R_{ui} - \bar{r}_u\) (puede ser positiva o negativa).
- **Celda sin rating**: \(\tilde{R}_{ui} = 0\) (“sin señal”), en lugar de \(- \bar{r}_u\), que sería una señal falsa y distorsionaría la correlación.

Esta matriz centrada \(\tilde{R}\) es la que se usa para entrenar el KNN y para calcular similitudes entre usuarios.

### 3.4 Matriz con ceros (para lookup)

Además se mantiene una matriz **user–item con ceros en los missing** (`user_item_matrix = raw.fillna(0)`), en float32, solo para consultas rápidas: “¿qué valor puso el usuario \(v\) en el ítem \(i\)?” (0 significa “no lo calificó”). No se usa para el cálculo de medias ni para el KNN.

---

## 4. K-Nearest Neighbors (KNN) y métrica de correlación

### 4.1 Objetivo

Dado un usuario \(u\), queremos encontrar los **K usuarios más similares** a \(u\) para luego usar sus ratings (o sus desviaciones) y predecir cómo valoraría \(u\) ítems que aún no ha calificado.

### 4.2 Métrica: correlación de Pearson como distancia

Scikit-learn no tiene una “métrica de Pearson” directa; usa **distancia** (a menor distancia, más similares). En el módulo se usa:

- **`metric='correlation'`**: para dos vectores \(\mathbf{x}, \mathbf{y}\), la “distancia” es
  \[
  d_{\text{corr}}(\mathbf{x}, \mathbf{y}) = 1 - \rho(\mathbf{x}, \mathbf{y})
  \]
  donde \(\rho\) es el **coeficiente de correlación de Pearson** entre \(\mathbf{x}\) e \(\mathbf{y}\).

Por tanto:

- **Pearson = 1** → distancia 0 (máxima similitud).
- **Pearson = 0** → distancia 1 (sin correlación).
- **Pearson = -1** → distancia 2 (correlación negativa).

Las filas que se pasan al KNN son los **vectores fila de la matriz centrada** \(\tilde{R}\): cada usuario \(u\) está representado por el vector de desviaciones \(\tilde{R}_{u\cdot}\).

### 4.3 Entrenamiento del KNN

- **Algoritmo**: `brute` (fuerza bruta), adecuado para comparar cada usuario con todos los demás.
- **Entrada**: matriz \(\tilde{R}\) (usuarios × ítems), tipo float32.
- **Salida**: un modelo que, dado el vector de un usuario, devuelve los índices y las distancias de sus K vecinos más cercanos.

No se hace predicción en esta etapa; el KNN solo sirve para obtener vecinos. La predicción se hace en el módulo CF con la fórmula de Pearson ponderado (ver siguiente sección).

---

## 5. Predicción por filtrado colaborativo (Pearson ponderado)

### 5.1 Fórmula matemática

Para predecir el rating del usuario \(u\) sobre el ítem \(i\):

1. Se obtienen los **K vecinos más cercanos** de \(u\) en la matriz centrada (excluyendo al propio \(u\) si aparece).
2. Se convierten distancias a **similitud**: \(\text{sim}(u,v) = 1 - d_{\text{corr}}(u,v)\). Así, similitud alta ⇔ Pearson alto.
3. Solo se usan vecinos \(v\) que **sí hayan calificado** el ítem \(i\) (es decir, \(R_{vi} > 0\) en la matriz de lookup).

La **predicción** es:

\[
\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in \mathcal{N}_u^{(i)}} \text{sim}(u,v) \cdot \tilde{R}_{vi}}{\sum_{v \in \mathcal{N}_u^{(i)}} |\text{sim}(u,v)|}
\]

donde:

- \(\mathcal{N}_u^{(i)}\): conjunto de vecinos de \(u\) que tienen rating para el ítem \(i\).
- \(\tilde{R}_{vi} = R_{vi} - \bar{r}_v\): desviación del vecino \(v\) en el ítem \(i\) (ya está en `matrix_centered`).
- El denominador con \(|\text{sim}(u,v)|\) permite usar similitudes negativas (Pearson negativo) sin invertir la fórmula.

Interpretación: se parte de la media del usuario \(\bar{r}_u\) y se corrige con un promedio ponderado por similitud de las desviaciones de los vecinos en ese ítem. Si no hay ningún vecino que haya calificado \(i\), se devuelve \(\bar{r}_u\). El resultado se **recorta al intervalo [1, 5]**.

### 5.2 Casos especiales en código

- Si **\(u\) o \(i\) no están** en la matriz (usuario o ítem nuevo): se devuelve la **media global** de los ratings de entrenamiento.
- Si **\(\bar{r}_u\) es NaN** (no debería ocurrir si \(u\) está en train): se usa la media global.
- **Similitud**: `similarities = 1 - distances[0]` (el KNN devuelve distancias de correlación).
- Solo se suman términos donde `rating = user_item_matrix.loc[neighbor_id, item_id] > 0`.

---

## 6. Modelo contextual: Random Forest (RF)

### 6.1 Objetivo

Predecir el **rating que un usuario típico daría a un negocio** a partir **solo de características del negocio** (y de la interacción usuario–negocio en entrenamiento). No se usa el perfil del usuario como input del RF; el “contexto” viene de ubicación, categorías y popularidad del ítem.

### 6.2 Variable objetivo (target)

En el entrenamiento, cada fila es una **pareja (usuario, negocio)** con su rating real:

- **\(y\)**: `stars_user` = estrellas que ese usuario puso a ese negocio en las reviews (1–5). Es el target a predecir.

### 6.3 Variables de entrada (features)

Se usan **solo atributos del negocio** (y de la review, si se añaden en el merge), nunca el `user_id` como feature. Tras el merge con la tabla de negocios, las features son:

**Numéricas:**

- **review_count**: número de reseñas del negocio (popularidad).
- **latitude**, **longitude**: ubicación geográfica.
- **is_open**: 0/1 (abierto o no).

**Categóricas binarias (one-hot implícito):**

- Se toma la columna `categories` (texto con categorías separadas por comas, estilo Yelp).
- Se extraen las **15 categorías más frecuentes** en el conjunto de entrenamiento (p. ej. Restaurants, Food, Nightlife, Bars, …).
- Para cada negocio y cada una de esas 15 categorías se crea una feature binaria: **cat_&lt;nombre&gt; = 1** si el negocio tiene esa categoría en su string, **0** si no.

En total hay **4 + 15 = 19 features**. No se usa `stars` del negocio como feature para evitar circularidad (predecir rating usando el rating agregado del negocio).

### 6.4 Entrenamiento

- **Modelo**: `RandomForestRegressor` (scikit-learn): 200 árboles, `max_depth=12`, `min_samples_leaf=5`, `n_jobs=-1`.
- **Entrada**: matriz \(X\) (filas = pares usuario–negocio de train, columnas = 19 features), vector \(y\) (rating usuario).
- **Salida**: modelo RF que, dado un vector de 19 features de un negocio, predice un valor continuo en [1, 5] (luego se hace clip en predicción).

El modelo se guarda en `models/rf_context_yelp.joblib` junto con la lista de categorías y nombres de features para que en inferencia se construya el mismo vector de 19 dimensiones.

### 6.5 Predicción y alineación

Para una lista de `business_ids`, hay que predecir un score por negocio **en el mismo orden** que la lista:

- Se indexa la tabla de negocios por `business_id` y se hace **reindex(business_ids)** para obtener una fila por ID en el orden solicitado (si un ID no existe, la fila será NaN y se rellena después).
- Se construyen las 19 features (numéricas + binarias de categorías) para cada fila.
- Se llama `model.predict(X)`; el resultado es un array en el mismo orden que `business_ids`. Se aplica **clip a [1, 5]**.

Así se evita el bug de desalineación (que el score del ítem \(i\) se asignara a otro ítem \(j\)).

---

## 7. Fusión híbrida y peso \(\alpha\)

### 7.1 Fórmula del score final

Para un usuario \(u\) y un ítem \(i\):

\[
\text{score}_{\text{final}}(u, i) = \alpha \cdot \hat{r}_{ui}^{(\text{CF})} + (1 - \alpha) \cdot \hat{r}_{i}^{(\text{RF})}
\]

donde:

- \(\hat{r}_{ui}^{(\text{CF})}\): predicción del modelo CF (Pearson) para el par \((u, i)\).
- \(\hat{r}_{i}^{(\text{RF})}\): predicción del RF para el negocio \(i\) (solo depende del ítem).
- \(\alpha \in [0, 1]\): peso del CF. Con \(\alpha = 0.1\) (valor por defecto actual), el 90 % del score viene del RF y el 10 % del CF.

El valor óptimo de \(\alpha\) se obtuvo por **grid search** (p. ej. 0.0, 0.1, …, 1.0) minimizando RMSE en un conjunto de validación (muestra del test set), resultando en **\(\alpha = 0.1\)** para este dataset.

### 7.2 Orden final

Los ítems del pool final se ordenan por **score final** de mayor a menor y se devuelve el **Top-N** (por defecto 5).

---

## 8. Flujo completo de recomendación (paso a paso)

1. **Entrada**: `user_id` \(u\), opcionalmente `context` (p. ej. tipos de turismo, preferencia outdoor), `alpha`, `top_n`.

2. **Pool de candidatos (KNN)**  
   - Si \(u\) está en la matriz de usuarios: se buscan sus K vecinos más cercanos en la matriz centrada; se agregan los negocios que esos vecinos han valorado, se promedian sus ratings por negocio y se toma el **top 100** por rating medio.  
   - Si \(u\) no está (cold start): se devuelven los 100 negocios más “populares” por número de ratings en train.

3. **Filtrado por contexto**  
   - Si se recibe `context` (p. ej. desde el formulario React):  
     - `tiposTurismo` se mapea a categorías Yelp (p. ej. “naturaleza” → Parks, Hiking, …); se filtran los candidatos que tengan al menos una de esas categorías.  
     - Si `pref_outdoor` es true, se filtran por categorías como Parks, Outdoor, Hiking, Nature, Lakes.  
   - Si tras filtrar no queda ninguno, se usa el pool sin filtrar.

4. **Scores RF**  
   - Para la lista ordenada de `business_id` del pool (refinado o no), se llama a `predict_context(business_ids)` y se obtiene un score RF por ítem, alineado por índice.

5. **Scores CF y fusión**  
   - Para cada ítem en el pool se calcula \(\hat{r}_{ui}^{(\text{CF})}\) con la fórmula de Pearson ponderado.  
   - Para cada ítem: \(\text{score}_{\text{final}} = \alpha \cdot \hat{r}_{ui}^{(\text{CF})} + (1-\alpha) \cdot \hat{r}_{i}^{(\text{RF})}\).

6. **Salida**  
   - Se ordenan los ítems por \(\text{score}_{\text{final}}\) descendente y se devuelven los primeros `top_n`, con campos como `item_id`, `title`, `score`, `pred_cf`, `pred_rf`.

---

## 9. Métricas de evaluación

### 9.1 Definiciones

Sobre un conjunto de pares \((u, i)\) con rating real \(r_{ui}\) y predicción \(\hat{r}_{ui}\):

- **RMSE (Root Mean Squared Error)**:
  \[
  \text{RMSE} = \sqrt{\frac{1}{n} \sum_{(u,i)} (r_{ui} - \hat{r}_{ui})^2}
  \]

- **MAE (Mean Absolute Error)**:
  \[
  \text{MAE} = \frac{1}{n} \sum_{(u,i)} |r_{ui} - \hat{r}_{ui}|
  \]

Se evalúa por separado: solo CF, solo RF, solo baseline (predecir siempre la media global) e híbrido con \(\alpha\) fijo (p. ej. 0.1), sobre una muestra del test set (p. ej. 1000 filas) para tener resultados representativos sin tardar demasiado.

### 9.2 Optimización de \(\alpha\)

En `optimize_alpha.py` se hace un barrido en \(\alpha \in \{0.0, 0.1, \ldots, 1.0\}\). Para cada \(\alpha\) se forma el score híbrido y se calcula RMSE (y opcionalmente MAE) sobre la misma muestra de test. El \(\alpha\) con menor RMSE se toma como óptimo (en el dataset actual, 0.1).

---

## 10. Cold start y casos borde

- **Usuario nuevo** (no está en la matriz de train): no se puede usar KNN; el pool de candidatos son los negocios más populares por conteo de ratings. Las predicciones CF para ese usuario se sustituyen por la media global cuando no hay vecinos o no hay datos.
- **Ítem nuevo** (negocio no visto en train): puede no estar en las columnas de la matriz; en ese caso la predicción CF devuelve la media global. El RF puede seguir prediciendo si el negocio está en la tabla de negocios con sus features (ubicación, categorías, etc.).
- **NaN**: si en algún paso sale NaN (p. ej. media de usuario inexistente), en evaluación se descarta esa muestra o se usa media global para no contaminar RMSE/MAE.

---

## 11. Resumen de símbolos y notación

| Símbolo | Significado |
|--------|-------------|
| \(u, v\) | Usuarios (user_id) |
| \(i\) | Ítem / negocio (business_id) |
| \(R_{ui}\) | Rating real de \(u\) a \(i\) (1–5) o missing |
| \(\bar{r}_u\) | Media de ratings del usuario \(u\) |
| \(\tilde{R}_{ui}\) | Rating centrado: \(R_{ui} - \bar{r}_u\) (0 si no hay rating) |
| \(\mathcal{N}_u\) | Vecinos de \(u\) (KNN) |
| \(\mathcal{N}_u^{(i)}\) | Vecinos de \(u\) que han calificado \(i\) |
| \(\text{sim}(u,v)\) | Similitud Pearson: \(1 - d_{\text{corr}}(u,v)\) |
| \(\hat{r}_{ui}^{(\text{CF})}\) | Predicción CF (Pearson ponderado) para \((u,i)\) |
| \(\hat{r}_{i}^{(\text{RF})}\) | Predicción RF para el ítem \(i\) |
| \(\alpha\) | Peso del CF en la fusión; \(1-\alpha\) peso del RF |
| \(\text{score}_{\text{final}}\) | \(\alpha \hat{r}^{(\text{CF})} + (1-\alpha) \hat{r}^{(\text{RF})}\) |

---

## 12. Archivos del modelo y su rol

| Archivo | Rol |
|---------|-----|
| `engine.py` | Carga datos, split 80/20, construcción de \(R\), \(\bar{r}_u\), \(\tilde{R}\), KNN, generación del pool de candidatos. |
| `cf.py` | Predicción \(\hat{r}_{ui}^{(\text{CF})}\) con Pearson ponderado por vecinos. |
| `rf_model.py` | Features de negocio (19), entrenamiento del RF, predicción \(\hat{r}_{i}^{(\text{RF})}\) con alineación por `business_id`. |
| `fusion.py` | Mapeo contexto → categorías Yelp, filtrado de candidatos, fusión \(\alpha \text{CF} + (1-\alpha)\text{RF}\), ordenación y Top-N. |
| `evaluate.py` | Cálculo de RMSE/MAE por componente (baseline, CF, RF, híbrido). |
| `optimize_alpha.py` | Grid search de \(\alpha\) y reporte del óptimo. |
| `pre_procesamiento.py` | Filtrado de JSON Yelp a CSVs de negocios y reviews. |

Con esto queda descrito el comportamiento técnico y matemático completo del modelo SMARTUR v2.
