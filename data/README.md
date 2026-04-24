# 📁 data/
Este directorio contiene todos los activos estáticos, datasets y configuraciones maestras necesarios para el funcionamiento de TinySpeak.

## Contenido
- **`master_dataset_config.json`**: El archivo de configuración de la "verdad absoluta" para todos los idiomas y datasets.
- **`audios/`**: Almacenamiento organizado por idioma para fonemas y palabras sintetizadas.
- **`visual/`**: Almacenamiento de imágenes de letras/grafemas generadas sintéticamente.
- **`checkpoints/`**: Los pesos de los modelos entrenados (`.ckpt`) y sus metadatos asociados.

## Uso en el Experimento
Toda la canalización de procesamiento comienza aquí. Los modelos cargan sus datasets desde estas subcarpetas y guardan los resultados definitivos en `checkpoints/`. Es crucial mantener la estructura sincronizada con lo definido en `master_dataset_config.json`.
