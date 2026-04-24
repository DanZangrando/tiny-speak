# 🧪 experiments/
Este directorio centraliza los resultados, logs e informes de los experimentos realizados con la arquitectura TinySpeak.

## Contenido
- **`logs/`**: Registros detallados de PyTorch Lightning y logs de entrenamiento (tensorboard, csv).
- **`metrics/`**: Archivos JSON que contienen las métricas de rendimiento brutas para su posterior análisis.
- **`informe_final.md`**: El compendio de hallazgos del experimento de transparencia y otros estudios.
- **`*.json`**: Archivos de resultados de experimentos automatizados (como el de Transparencia de Streamlit).

## Uso en el Experimento
Aquí es donde se "cosecha" el conocimiento. Una vez que un modelo termina su entrenamiento (en `data/checkpoints/`), sus métricas de éxito y curvas se analizan desde este directorio para validar la hipótesis de transparencia ortográfica.
