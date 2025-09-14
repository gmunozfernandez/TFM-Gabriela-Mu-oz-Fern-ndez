# TFM – Gabriela Muñoz Fernández

## 🎬 Título
**Análisis y modelado de un sistema para la clasificación de películas en “exitosas” y “no exitosas” en función del revenue.**

Este trabajo desarrolla un modelo predictivo que, a partir de información disponible **antes del estreno** de cada película, estima su probabilidad de éxito.  
Los datos utilizados provienen de la plataforma **Kaggle**.

---

## 📂 Estructura del proyecto

> ⚠️ La carpeta `data/` no se incluye en el repositorio porque contiene información de **más de un millón de películas** y su peso es demasiado elevado.  
> Sin embargo, **todos los notebooks ya están ejecutados**, por lo que se pueden revisar resultados y gráficas sin necesidad de volver a correrlos.

```bash
TFM-GABRIELA-MU-OZ-FERNANDEZ_v4
├── images/                          # Recursos gráficos y visualizaciones
├── models/                          # Modelos entrenados (checkpoints, serializaciones, etc.)
├── notebooks/                       # Jupyter Notebooks usados en el análisis y modelado
│   ├── descriptive_analysis/        
│   │   └── (1)_descriptive.ipynb          # Análisis exploratorio inicial
│   ├── modeling/                    
│   │   └── (2)_modeling_with_inflation_correction.ipynb   # Modelado con correcciones
│   └── preparing_data/              
│       └── embeddings_&_imputations.ipynb # Preparación, embeddings e imputaciones
├── production/                      # Scripts y artefactos de despliegue
│   └── movie_classification_embeddings_inflation/
│       ├── calibrated_model.pkl            # Modelo calibrado
│       ├── experiment_config.json          # Configuración del experimento
│       ├── inflation_analysis.json         # Resultados de análisis con inflación
│       ├── model_comparison_metrics.csv    # Métricas comparativas entre modelos
│       ├── preprocessor.pkl                # Preprocesador de datos
│       └── sample_predictions.csv          # Predicciones de ejemplo
├── test/                           # Scripts de pruebas
│   ├── main.py
│   ├── open_ai.py
│   └── venv_activate.ps1
├── utils/                          # Funciones auxiliares
│   ├── embedding.py
│   ├── gpu_check.py
│   ├── open_ai.py
│   └── project_structure.txt
├── README.md                       # Documentación del proyecto
