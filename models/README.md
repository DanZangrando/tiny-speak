# 🧠 models/
Definiciones de las arquitecturas neuronales personalizadas (Tiny Architectures) de TinySpeak.

## Contenido
- **`tiny_ears.py`**: El modelo `TinyEars` (vía fonológica). Basado en capas convolucionales temporales y de audio para reconocimiento de patrones en espectrogramas.
- **`tiny_eyes.py`**: El modelo `TinyEyes` (vía visual). Una jerarquía CNN para el reconocimiento de grafemas.
- **`tiny_reader.py`**: La suite `TinyReader` (G2P y P2W). Modelos Seq2Seq y mecanismos de atención para integrar ambas vías.

## Uso en el Experimento
Representa el cerebro del sistema. Todos los modelos aquí se construyen utilizando la misma filosofía: 100% personalizados, modulares y diseñados para imitar funciones cognitivas humanas sin depender de black-boxes externas.
