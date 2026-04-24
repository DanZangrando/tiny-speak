# 💾 Checkpoints: Pesos y Estado de Modelos

Este directorio almacena los pesos entrenados (`.ckpt`) y sus metadatos asociados (`.meta.json`). Está organizado por arquitectura para facilitar la carga y evaluación.

## Organización
*   **`tiny_ears_phonemes/`**: Pesos de la vía auditiva experta en fonemas.
*   **`tiny_ears_words/`**: Pesos de la vía auditiva experta en palabras completas.
*   **`tiny_eyes/`**: Pesos del reconocedor visual de grafemas.
*   **`tiny_speller/`**: Pesos del Stage 1 de lectura (G2P).
*   **`tiny_reader/`**: Pesos del Stage 2 de lectura (P2W) y modelos End-to-End.

> [!NOTE]
> Cada archivo `.ckpt` tiene un archivo `.meta.json` hermano que describe el vocabulario, hiperparámetros e idioma con el que fue entrenado.
