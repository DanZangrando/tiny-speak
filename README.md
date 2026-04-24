# 🚀 TinySpeak: Proyecto de Refactorización Cognitiva

TinySpeak es un experimento de modelado cognitivo que explora la adquisición del lenguaje y la lectura desde una perspectiva 100% personalizada y modular. Basado en la teoría de la **Codificación Predictiva**, el sistema integra vías auditivas (TinyEars) y visuales (TinyEyes) para simular el aprendizaje de la lectura.

## 🏗️ Estructura del Proyecto (Refactorizada)

La arquitectura ha sido simplificada y centralizada para máxima prolijidad y modularidad:

```text
.
├── 📁 components/              # Componentes UI de Streamlit (Premium Aesthetics)
├── 📁 data/                    # Activos estáticos, Datasets y Configuración
│   ├── 📁 audios/              # Dataset auditivo (fonemas/palabras)
│   ├── 📁 checkpoints/         # Pesos de modelos y metadatos (Gros de la IA)
│   ├── 📁 visual/              # Dataset visual (grafemas)
│   └── 📄 master_dataset_config.json  # Configuración maestra (Verdad Absoluta)
├── 📁 experiments/             # Resultados, Métricas y Logs
│   ├── 📁 logs/                # Logs de entrenamiento (Lightning)
│   ├── 📁 metrics/             # Reportes JSON de rendimiento
│   └── 📄 informe_final.md     # Hallazgos y conclusiones
├── 📁 models/                  # Arquitecturas 100% Custom (No Black-Boxes)
│   ├── 📄 tiny_ears.py         # Vía Auditiva (V1, V2, Transformer)
│   ├── 📄 tiny_eyes.py         # Vía Visual (V1, V2, V4, IT)
│   └── 📄 tiny_reader.py       # Interfaz G2P y P2W (Seq2Seq)
├── 📁 pages/                   # Módulos de la aplicación Streamlit (01-08)
├── 📁 training/                # Motor de entrenamiento y lógica de datos
├── 📁 utils/                   # Herramientas de soporte y procesamiento
├── 📄 app.py                   # Dashboard Principal (TinyDashboard)
└── 📄 requirements.txt         # Dependencias del proyecto
```

## 🧠 Arquitecturas Tiny

- **TinyEars**: Procesa audio bruto y espectrogramas para identificar fonemas y palabras.
- **TinyEyes**: Procesa imágenes de grafemas simulando el flujo ventral de la corteza visual humana.
- **TinySpeller (Stage 1 G2P)**: Aprende a asociar grafemas visuales con sus correspondientes "imágenes fonémicas".
- **TinyReader (Stage 2 P2W)**: Acceso léxico completo. De sonidos imaginados a significados de palabras.

## 🚀 Cómo Empezar

1. Instala las dependencias: `pip install -r requirements.txt`
2. Ejecuta el dashboard: `streamlit run app.py`
3. Sigue el orden de entrenamiento sugerido en el menú lateral para construir el sistema cognitivo paso a paso.

---
*Este proyecto ya no depende de modelos pre-entrenados como Wav2Vec o CorNet. Todo el aprendizaje ocurre desde cero dentro del ecosistema TinySpeak.*