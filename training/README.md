# 🏋️ Módulos de Entrenamiento

Este directorio contiene la lógica central de entrenamiento implementada con **PyTorch Lightning**. Aquí se definen los sistemas de datos (DataModules/Datasets) y los sistemas de modelos (LightningModules).

## 🧠 Lightning Modules
Estos archivos encapsulan la lógica de entrenamiento, validación, optimizadores y cálculo de pérdidas.

*   **`audio_module.py`**:
    *   `TinyListenerLightning`: Entrena el modelo de audición.
    *   Maneja Wav2Vec 2.0 (congelado o fine-tuned) y el decodificador LSTM.
    *   Calcula Top-1 y Top-5 Accuracy.

*   **`visual_module.py`**:
    *   `TinyRecognizerLightning`: Entrena el modelo de visión.
    *   Implementa CORnet-Z para reconocimiento de caracteres.
    *   Gestiona aumentación de datos en tiempo real.

*   **`reader_module.py`**:
    *   `TinyReaderLightning`: Entrena el modelo de imaginación.
    *   **Lógica Multimodal**:
        1.  **Bottom-Up**: Extrae embeddings reales de audio usando un Listener congelado.
        2.  **Top-Down**: Genera embeddings desde el concepto visual.
        3.  **Pérdidas**: Combina MSE (reconstrucción), Coseno (similitud) y **Perceptual Loss** (feedback del Listener).

## 💾 Datasets
Definiciones de clases `torch.utils.data.Dataset` personalizadas.

*   **`audio_dataset.py`**:
    *   Carga y procesa archivos de audio.
    *   Maneja el padding y la tokenización de palabras.
    *   Soporta carga en memoria para alta velocidad.

*   **`visual_dataset.py`**:
    *   Genera imágenes sintéticas de letras "al vuelo" o carga desde disco.
    *   Aplica transformaciones visuales (ruido, rotación, desenfoque) para robustez.

## ⚙️ Configuración
*   **`config.py`**:
    *   Utilidades para cargar y validar la configuración maestra (`master_dataset_config.json`).
