"""
utils/ — Utilidades de TinyLearner.

Re-exports públicos del paquete para mantener compatibilidad con
imports existentes del tipo ``from utils import AUDIO_SR, load_waveform``.

Submódulos:
    audio          → carga y síntesis de audio
    checkpoints    → gestión de checkpoints
    device         → detección de hardware
    graphemes      → tokenización y fonemas
    plotting       → visualizaciones y SoftDTW
    serialization  → carga/guardado de JSON
"""

# Constantes
from utils.audio import AUDIO_SR, BATCH_SIZE

# Device
from utils.device import encontrar_device

# Audio
from utils.audio import (
    load_waveform,
    synthesize_word,
    save_waveform_to_audio_file,
    generar_audio_gtts,
    generar_audio_espeak,
    generar_audio_segun_metodo,
    aplicar_variaciones_audio,
    generar_variaciones_completas,
    save_audio_file,
    change_speed,
)

# Graphemes / vocabulary
from utils.graphemes import (
    tokenize_graphemes,
    get_language_letters,
    get_phoneme_inventory,
    get_phonemes_from_word,
    get_default_words,
    LETTERS,
)

# Checkpoints
from utils.checkpoints import (
    list_checkpoints,
    save_model_metadata,
    delete_run_artifacts,
    save_training_metrics,
    load_training_metrics,
    list_metrics_files,
)

# Plotting / losses
from utils.plotting import (
    plot_waveform_native,
    plot_logits_native,
    SoftDTW,
)

# Serialization
from utils.serialization import (
    convert_numpy_types,
    load_dataset_config,
    save_dataset_config,
)

# ---------------------------------------------------------------------------
# Callbacks de entrenamiento (importados desde training para compatibilidad)
# ---------------------------------------------------------------------------
# Nota: se importan con try para que utils sea importable sin training instalado.
try:
    from training.callbacks import RealTimePlotCallback, ReaderPredictionCallback
except ImportError:
    pass

__all__ = [
    # Constantes
    "AUDIO_SR", "BATCH_SIZE", "LETTERS",
    # Device
    "encontrar_device",
    # Audio
    "load_waveform", "synthesize_word", "save_waveform_to_audio_file",
    "generar_audio_gtts", "generar_audio_espeak",
    "generar_audio_segun_metodo", "aplicar_variaciones_audio",
    "generar_variaciones_completas", "save_audio_file", "change_speed",
    # Graphemes
    "tokenize_graphemes", "get_language_letters", "get_phoneme_inventory",
    "get_phonemes_from_word", "get_default_words",
    # Checkpoints
    "list_checkpoints", "save_model_metadata", "delete_run_artifacts",
    "save_training_metrics", "load_training_metrics", "list_metrics_files",
    # Plotting
    "plot_waveform_native", "plot_logits_native", "SoftDTW",
    # Serialization
    "convert_numpy_types", "load_dataset_config", "save_dataset_config",
    # Callbacks
    "RealTimePlotCallback", "ReaderPredictionCallback",
]
