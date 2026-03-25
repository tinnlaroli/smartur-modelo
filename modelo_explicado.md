# SMARTUR v4 — Modelo explicado (True ML Contextual)

Este documento describe de forma completa el funcionamiento técnico, arquitectónico y matemático del sistema de recomendación híbrido SMARTUR v4, enfocado en el aprendizaje intrínseco (Machine Learning Contextual).

---

## 1. Visión general del sistema

SMARTUR es un **sistema de recomendación híbrido de nivel industrial** que combina:

1. **Filtrado colaborativo (CF)** basado en **correlación de Pearson** y **K-Nearest Neighbors (KNN)** para explotar similitud entre turistas.
2. **Modelo contextual True ML (Random Forest)** que no solo predice a partir del negocio local, sino del cruce bidimensional dinámico entre *[Features del Negocio] × [Features del Turista]*.
3. **Fusión lineal** de ambas vías algorítmicas con peso \(\alpha\) configurable.
4. **Filtros Duros (Poda)** que desechan agresivamente locales que violen reglas de "No-Negociables" (Ej. Accesibilidad obligatoria, Necesidad de Hotel, Cero comida).

El flujo de una recomendación es:

```
Turista u + Ingesta completa de su Formulario React (Contexto)
    → Pool de candidatos inicial (KNN sobre usuarios similares o Populares)
    → Filtro Duro (Descarte por Accesibilidad, Comida y Hoteles)
    → Filtro Suave (Afinación por Categorías Yelp vs Tipos de Turismo)
    → Para cada candidato: score_CF(u, i)
    → Para cada candidato: score_RF( [Array de 30 variables combinadas Item+User+Match] )
    → Score final = α·score_CF + (1−α)·score_RF
    → Ordenar por score final → Top-N
```

---

## 2. Datos de entrada y Procesamiento de Lenguaje Natural (NLP)

### 2.1 Fuente de datos oficial

El motor consume el inmenso archivo original de Yelp (`yelp_academic_dataset_business.json`). Mediante el script `pre_procesamiento.py` extraemos campos ricos y enterrados en el JSON:
- Atributos base: `price_level`, `is_accessible`, `outdoor`.
- Atributos enriquecidos: `is_good_for_kids` (Familias) y parseamos el diccionario interno de `Ambience` buscando claves `romantic: true` o `intimate: true` (`is_romantic`).

Las reseñas se filtran (`data_reviews_limpio.csv`) para contener solo las interacciones correspondientes al radar Turístico/Gastronómico.

---

## 3. Matriz de utilidad y centrado (Motor Pearson)

*(Esta fase se mantiene intacta desde la arquitectura base)*

La **correlación de Pearson** mide a los usuarios a base de las desviaciones de sus gustos respecto a su propia media \(\bar{r}_u\). Solo se entrena y se rellenan las distancias de donde hubo señal, sin rellenar con ceros artificiales. \(\tilde{R}_{ui} = R_{ui} - \bar{r}_u\).

---

## 4. K-Nearest Neighbors (KNN)

El KNN de fuerza bruta mapea a los usuarios dentro del clúster euclidiano.
La métrica es distancia de correlación: \(d_{\text{corr}}(\mathbf{x}, \mathbf{y}) = 1 - \rho(\mathbf{x}, \mathbf{y})\).

No se hace predicción aquí; el KNN solo obtiene a los K vecinos más cercanos de ese turista u.

---

## 5. Predicción CF (Pearson ponderado)

Para predecir el rating del usuario \(u\) sobre el negocio \(i\):

\[
\hat{r}_{ui}^{(\text{CF})} = \bar{r}_u + \frac{\sum_{v \in \mathcal{N}_u^{(i)}} \text{sim}(u,v) \cdot \tilde{R}_{vi}}{\sum_{v \in \mathcal{N}_u^{(i)}} |\text{sim}(u,v)|}
\]

Se parte de la costumbre promedio de calificaciones del turista y se corrige sumando cómo calificaron sus vecinos ese lugar, ponderado en función de qué tan parecidos son a él.

---

## 6. Machine Learning Contextual: El Nuevo Random Forest (RF v4)

### 6.1 Objetivo True ML

Antiguamente (v2), el Random Forest solo veía el Restaurante y predecía una nota base. Luego, se sumaban/restaban puntajes ("reglas harcodeadas" o Sistemas Expertos) si el presupuesto del usuario era distinto.
En **SMARTUR v4**, el Random Forest debe inferir dinámicamente y por sí solo las reglas psicológicas de la recomendación aprendiendo de datos bi-direccionales.

### 6.2 Data Synthesis (Entrenamiento por Simulación de Turistas)

Para que el modelo aprenda a predecir cruces, necesitamos turistas que tengan "Presupuesto", "Rango de Edad", "Comitiva (Familia/Pareja)", etc. La data de Yelp solo tiene `user_id` y su rating.
La función mágica `_simulate_user_contexts` (dentro de `.train()`) inyecta **Contexto Sintético** por ingeniería inversa a las reseñas del set de Entrenamiento:

- *Presupuesto inferido*: Si un usuario otorgó grandes notas (≥ 4 estrellas) a puros lugares "Price Level 4", le asume un `user_budget` de 4.
- *Atributos estocásticos*: Asigna demográficas (Edades, Familas, Parejas) con distribuciones lógicas a los 80,000 registros.
- *Match Features Vectoriales*: Cruza los datos sintéticos con las particularidades del Negocio para calcular penalizaciones implícitas pre-entrenamiento (Ej. `budget_delta`, `kids_match`, `romantic_match`).

### 6.3 Súper Matriz X Ampliada (Features Cruzados)

La matriz final \(\mathbf{X}\) con la que el modelo invoca `model.fit(X, y)` posee más de **35 Dimensiones**:
1. **Item Features**: Ubicación, Popularidad, `price_level`, Categorías One-Hot, `is_good_for_kids`, `is_romantic`.
2. **User Features**: Presupuesto y Edades (Ordinales), Preferencias Multi-Hot, `group_type` (solo, pareja, familia).
3. **Match (Interacción)**: Distancia alzacostos (`budget_delta`), Puntos de Overlap de turismo, Choques `romantic_match` y `kids_match`.

### 6.4 Predicción Pura ("No-Expert")

Dado que las colisiones están mapeadas, el RF `predict_with_context` recibe las mismas 35 características del nuevo turista real que llena el formulario y extrae la nota. **Ya no existen multiplicadores fijos de +0.15 o -0.20**. Es el algoritmo de árboles profundos quien determina cuánto dañar la valoración de un hotel si se sale de presupuesto, condicionado si esa persona viene en familia o no.

---

## 7. Retrieve & Rank (Fusión Híbrida y Noción de Poda)

### 7.1 Poda Radical (Filtro Duro Estricto)

Antes de hacer los extenuantes cálculos del RF o del KNN, el pool de hasta 200 candidatos cruza los "No Negociables" en `fusion.py`:
- `needs_hotel == true`: La matriz descarta violenta e inmediatamente cualquier local no-hotel.
- `pref_food == false`: Elimina explícitamente negocios de comida.
- `requiere_accesibilidad`: Barre con lugares sin rampas informadas.

### 7.2 Fusión Lineal

Para los escasos sobrevivientes que el filtro permitió llegar a la final, se combinan:

\[
\text{score}_{\text{final}}(u, i) = \alpha \cdot \hat{r}_{ui}^{(\text{CF})} + (1 - \alpha) \cdot \hat{r}_{i}^{(\text{RF})}
\]

Dado el masivo incremento de inteligencia algorítmica y paramétrica del nuevo Random Forest v4, usamos un \(\alpha = 0.2\). Esto significa que **el 80% de la fuerza de decisión de SMARTUR recae sobre la inferencia de Machine Learning profundo**, y el 20% es un bonus colaborativo por similitud gregaria.
