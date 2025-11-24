### Descripción del Workflow de Datos

El procedimiento técnico se articula en cuatro fases secuenciales, integrando puntos de decisión crítica (*Checkpoints*) que condicionan el avance del flujo de datos.

#### 1. Fase de Inicialización y ETL (Extract, Transform, Load)
Se establece la base de datos inicial mediante la ejecución secuencial de los scripts de ingestión y limpieza.
* **Paso 1.1:** Ejecutar **`01_extract_raw_data.py`** para obtener los datos crudos de la API.
* **Paso 1.2:** Ejecutar **`02_process_raw_data.py`** para unificar tablas y limpiar ruido estructural.
* **Salida ($O_1$):** Archivo base **`02_processed_data.parquet`**.

#### 2. Fase de Filtrado de Relevancia (Optimización de Costes)
El objetivo es reducir la dimensionalidad del *dataset* antes del análisis costoso.
* **Paso 2.1 (Cálculo):** Se ejecuta **`03_content_relevance_feature_engineering.py`** sobre $O_1$. Se genera la variable `content_relevance_score`.
* **Paso 2.2 (Validación I):** Se ejecuta **`feature_engineering_validation_03.py`**, generando una muestra aleatoria en JSON para revisión humana.
* **Decisión Lógica 1 (Punto de Control):**
    * **SI [Validación < Umbral de Calidad]:** Se detectan inconsistencias en la detección de temas. (**TO DO:** definir el Umbral de Calidad)
        * *Acción:* Refinar el *prompt* en `src/feature_engineering_utils.py`.
        * *Retorno:* Volver al **Paso 2.1** (Recalcular la variable).
    * **SI [Validación $\ge$ Umbral de Calidad]:** La clasificación es correcta.
        * *Acción A:* Ejecutar **`04_join_content_relevance_feature.py`** para integrar los *scores*.
        * *Acción B:* Ejecutar **`05_filter_relevant_content.py`** aplicando el filtro ($\text{score} \ge 3$).
* **Salida ($O_2$):** *Dataset* depurado **`05_processed_data.parquet`** (solo contenido relevante).

#### 3. Fase de Análisis Semántico Profundo (Feature Engineering Complejo)
Esta fase se ejecuta exclusivamente sobre la salida depurada $O_2$.
* **Paso 3.1 (Cálculo):** Se ejecuta **`06_remaining_feature_engineering.py`** para inferir variables latentes (Postura Política, Sentimiento, Tono).
* **Paso 3.2 (Validación II):** Se ejecuta **`feature_engineering_validation_06.py`** para auditar la coherencia ideológica y sentimental.
* **Decisión Lógica 2 (Punto de Control):**
    * **SI [Validación < Umbral de Calidad]:** Se detectan alucinaciones o sesgos en la clasificación política.
        * *Acción:* Ajustar las muestras *Few-Shot* en `data/expert_samples/` o calibrar la temperatura del modelo.
        * *Retorno:* Volver al **Paso 3.1** (Recalcular variables afectadas).
    * **SI [Validación $\ge$ Umbral de Calidad]:** Las inferencias son teóricamente válidas.
        * *Acción:* Proceder a la fase de consolidación.

#### 4. Fase de Consolidación Final
Se ejecuta tras superar satisfactoriamente todos los puntos de control.
* **Paso 4.1:** Ejecutar **`07_join_remaining_features.py`**. Este script ensambla las variables validadas en la Fase 3 con los metadatos originales preservados desde la Fase 1.
* **Salida Final ($O_{final}$):** Matriz de datos **`07_processed_data.parquet`**, lista para el modelado de *clustering*.