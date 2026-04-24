# 🖼️ pages/
Contiene la interfaz de usuario modular construida con Streamlit.

## Contenido (Pipeline del Experimento)
- **`01_Audio_Dataset.py`**: Gestión de síntesis de voz y configuración fonológica.
- **`02_Visual_Dataset.py`**: Generador de estímulos visuales (letras en distintas fuentes).
- **`03_TinyEars_Phonemes.py`**: Laboratorio de fonemas. Entrenamiento del modelo auditivo de bajo nivel.
- **`04_TinyEars_Words.py`**: Laboratorio léxico auditivo. Reconocimiento de palabras habladas.
- **`05_TinyEyes.py`**: Laboratorio visual. Entrenamiento del reconocedor de grafemas.
- **`06_TinySpeller.py`**: Stage 1 de lectura. Asociación grafema -> fonema (G2P).
- **`07_TinyReader.py`**: Stage 2 de lectura. Integración fonológica para acceso léxico (P2W).
- **`08_🔬_Transparency_Experiment.py`**: El experimento final automatizado para medir la opacidad entre idiomas.

## Uso en el Experimento
Cada página representa un paso lógico en la jerarquía de aprendizaje. Se debe seguir el orden numérico para un entrenamiento coherente desde lo sensorial a lo cognitivo.