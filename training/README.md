# 🏃‍♂️ training/
Lógica de bajo nivel para el entrenamiento de modelos y gestión de datos.

## Contenido
- **`runner.py`**: El orquestador automatizado. Permite ejecutar entrenamientos completos desde scripts o el experimento de transparencia.
- **`audio_dataset.py` / `visual_dataset.py`**: Clases `Dataset` y `DataLoader` de PyTorch para alimentar los modelos.
- **`audio_module.py` / `visual_module.py` / `reader_module.py`**: Envoltorios de PyTorch Lightning que definen los pasos de entrenamiento, validación y optimizadores.
- **`callbacks.py`**: Utilidades para visualización en tiempo real y logs personalizados durante el entrenamiento.
- **`config.py`**: Manejador centralizado para cargar y validar la configuración global del proyecto.

## Uso en el Experimento
Es el motor "bajo el capó". Proporciona la infraestructura necesaria para que las páginas de Streamlit y los scripts de automatización realicen entrenamientos reproducibles y eficientes.
