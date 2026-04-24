"""
utils/audio.py — Síntesis, carga y augmentación de audio.

Funciones:
    load_waveform               Carga un archivo de audio como tensor PyTorch.
    synthesize_word             Síntesis con eSpeak.
    save_waveform_to_audio_file Guarda un tensor como WAV.
    generar_audio_gtts          Síntesis con Google TTS.
    generar_audio_espeak        Síntesis con eSpeak (bytes).
    generar_audio_segun_metodo  Dispatcher según método.
    aplicar_variaciones_audio   Aplica augmentaciones de audio.
    generar_variaciones_completas Genera audio base + N variaciones.
    save_audio_file             Guarda bytes de audio en disco.
    change_speed                Cambia la velocidad de un AudioSegment.
    load_wav2vec_model          Carga el modelo Wav2Vec2 (para comparación).

    BATCH_SIZE   Batch size por defecto para entrenamiento.
"""

from __future__ import annotations

import io
import os
import random
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

AUDIO_SR: int = 16_000
BATCH_SIZE: int = 32
AUDIO_EMBED_DIM: int = 256  # Dimensión estándar de embeddings auditivos custom


# ---------------------------------------------------------------------------
# Carga de audio
# ---------------------------------------------------------------------------

def load_waveform(audio_path, target_sr: int = AUDIO_SR) -> torch.Tensor | None:
    """Carga un archivo de audio y lo retorna como tensor (T,) a ``target_sr``.

    Acepta rutas de archivo (str / Path) o buffers de bytes (BytesIO).
    Usa torchaudio con fallback a librosa/soundfile.
    """
    try:
        import torchaudio  # local import to keep module importable without torchaudio

        def _load_torchaudio(path):
            waveform, sample_rate = torchaudio.load(path)
            return waveform, sample_rate

        if hasattr(audio_path, "read"):
            # BytesIO — guardar temporalmente
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_path.read())
                tmp_path = tmp.name
            try:
                waveform, sample_rate = _load_torchaudio(tmp_path)
            finally:
                os.remove(tmp_path)
        else:
            waveform, sample_rate = _load_torchaudio(audio_path)

        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform, dtype=torch.float32)

    except Exception:
        # Fallback a librosa
        try:
            import librosa

            if hasattr(audio_path, "read"):
                audio_path.seek(0)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_path.read())
                    tmp_path = tmp.name
                try:
                    audio_np, sample_rate = librosa.load(tmp_path, sr=target_sr, mono=True)
                finally:
                    os.remove(tmp_path)
            else:
                audio_np, sample_rate = librosa.load(audio_path, sr=target_sr, mono=True)

            waveform = torch.tensor(audio_np, dtype=torch.float32)
        except Exception as exc:
            print(f"Error cargando audio desde {audio_path}: {exc}")
            return None

    # Convertir a mono
    if waveform.dim() > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze()

    # Remuestrear si es necesario
    if sample_rate != target_sr:
        try:
            import torchaudio
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)
        except Exception:
            pass

    return waveform


def synthesize_word(
    word: str,
    voice: str = "es",
    rate: int = 80,
    pitch: int = 70,
    amplitude: int = 120,
) -> torch.Tensor | None:
    """Sintetiza una palabra usando eSpeak y la retorna como tensor."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_path = tmp.name

        cmd = ["espeak", "-v", voice, "-s", str(rate), "-p", str(pitch),
               "-a", str(amplitude), "-w", tmp_path, word]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        proc.wait()

        if os.path.exists(tmp_path):
            waveform = load_waveform(tmp_path)
            os.remove(tmp_path)
            return waveform
    except Exception as exc:
        print(f"Error sintetizando '{word}': {exc}")
    return None


def save_waveform_to_audio_file(
    waveform: torch.Tensor,
    file_path: str | Path,
    sample_rate: int = AUDIO_SR,
) -> bool:
    """Guarda un waveform tensor como WAV. Retorna True si tuvo éxito."""
    try:
        import torchaudio
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        torchaudio.save(str(file_path), waveform, sample_rate)
        return True
    except Exception:
        try:
            import soundfile as sf
            w_np = waveform.cpu().numpy() if hasattr(waveform, "cpu") else waveform
            if w_np.ndim > 1:
                w_np = w_np.squeeze()
            sf.write(str(file_path), w_np, sample_rate)
            return True
        except Exception as exc:
            print(f"Error guardando audio: {exc}")
    return False


# ---------------------------------------------------------------------------
# Síntesis TTS
# ---------------------------------------------------------------------------

def change_speed(audio_segment, speed: float = 1.0):
    """Cambia la velocidad de un pydub AudioSegment."""
    new_sr = int(audio_segment.frame_rate * speed)
    return audio_segment._spawn(
        audio_segment.raw_data, overrides={"frame_rate": new_sr}
    ).set_frame_rate(audio_segment.frame_rate)


def generar_audio_gtts(
    texto: str, idioma: str = "es", velocidad: float = 1.0
) -> bytes | None:
    """Genera audio WAV usando Google Text-to-Speech."""
    from gtts import gTTS
    from pydub import AudioSegment
    from pydub.effects import normalize

    phoneme_map = {"r": "rrr", "s": "sss", "f": "fff", "m": "mmm",
                   "n": "nnn", "l": "lll", "z": "zzz"}
    texto_a_sintetizar = phoneme_map.get(texto.lower(), texto) if len(texto) == 1 else texto

    try:
        tts = gTTS(text=texto_a_sintetizar, lang=idioma, slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tts.save(tmp.name)
            audio = AudioSegment.from_mp3(tmp.name)
            if velocidad != 1.0:
                audio = change_speed(audio, velocidad)
            audio = normalize(audio)
            buf = io.BytesIO()
            audio.export(buf, format="wav")
            os.unlink(tmp.name)
            return buf.getvalue()
    except Exception as exc:
        print(f"Error gTTS: {exc}")
        return None


def generar_audio_espeak(
    texto: str,
    idioma: str = "es",
    rate: int = 80,
    pitch: int = 70,
    amplitude: int = 120,
) -> bytes | None:
    """Genera audio WAV usando eSpeak, retorna bytes."""
    try:
        cmd = ["espeak", "-v", idioma, "-s", str(rate), "-p", str(pitch),
               "-a", str(amplitude), "-w", "/dev/stdout", texto]
        result = subprocess.run(cmd, capture_output=True, check=True)
        return result.stdout
    except Exception as exc:
        print(f"Error eSpeak: {exc}")
        return None


def generar_audio_segun_metodo(
    texto: str,
    metodo: str = "gtts",
    idioma: str = "es",
    **kwargs,
) -> bytes | None:
    """Dispatcher: genera audio según el método indicado ('gtts' o 'espeak')."""
    if metodo == "gtts":
        return generar_audio_gtts(texto, idioma, kwargs.get("velocidad", 1.0))
    if metodo == "espeak":
        return generar_audio_espeak(
            texto, idioma,
            kwargs.get("rate", 80),
            kwargs.get("pitch", 70),
            kwargs.get("amplitude", 120),
        )
    return None


# ---------------------------------------------------------------------------
# Variaciones / Augmentaciones
# ---------------------------------------------------------------------------

def aplicar_variaciones_audio(
    audio_bytes: bytes | None,
    variacion_tipo: str,
    config_rangos: dict | None = None,
) -> tuple[bytes | None, dict]:
    """Aplica una variación de augmentación al audio y retorna los nuevos bytes y params."""
    from pydub import AudioSegment
    from pydub.effects import normalize

    if not audio_bytes:
        return None, {}
    if config_rangos is None:
        config_rangos = {"pitch": [0.8, 1.3], "speed": [0.7, 1.4], "volume": [0.8, 1.2]}

    try:
        audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
        params = {
            "pitch_factor": 1.0,
            "speed_factor": 1.0,
            "volume_factor": 1.0,
            "tipo": variacion_tipo,
        }

        # Aplicar aumentación multifactor (Combinación de todos los factores)
        if variacion_tipo == "multifactor":
            params["pitch_factor"] = random.uniform(config_rangos["pitch"][0], config_rangos["pitch"][1])
            params["speed_factor"] = random.uniform(config_rangos["speed"][0], config_rangos["speed"][1])
            params["volume_factor"] = random.uniform(config_rangos["volume"][0], config_rangos["volume"][1])
            
            # 1. Pitch
            if params["pitch_factor"] != 1.0:
                new_rate = int(audio.frame_rate * params["pitch_factor"])
                audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_rate}).set_frame_rate(22050)
            
            # 2. Speed
            if params["speed_factor"] != 1.0:
                audio = change_speed(audio, params["speed_factor"])
                
            # 3. Volume
            if params["volume_factor"] != 1.0:
                audio = audio + (20 * np.log10(params["volume_factor"]))
        
        # Lógica Legacy (para mantener compatibilidad si se llama individualmente)
        elif variacion_tipo == "pitch_alto":
            params["pitch_factor"] = random.uniform(1.1, config_rangos["pitch"][1])
            new_rate = int(audio.frame_rate * params["pitch_factor"])
            audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_rate}).set_frame_rate(22050)
        elif variacion_tipo == "pitch_bajo":
            params["pitch_factor"] = random.uniform(config_rangos["pitch"][0], 0.9)
            new_rate = int(audio.frame_rate * params["pitch_factor"])
            audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_rate}).set_frame_rate(22050)
        elif variacion_tipo == "rapido":
            params["speed_factor"] = random.uniform(1.1, config_rangos["speed"][1])
            audio = change_speed(audio, params["speed_factor"])
        elif variacion_tipo == "lento":
            params["speed_factor"] = random.uniform(config_rangos["speed"][0], 0.9)
            audio = change_speed(audio, params["speed_factor"])
        elif variacion_tipo == "fuerte":
            params["volume_factor"] = random.uniform(1.1, config_rangos["volume"][1])
            audio = audio + (20 * np.log10(params["volume_factor"]))
        elif variacion_tipo == "suave":
            params["volume_factor"] = random.uniform(config_rangos["volume"][0], 0.9)
            audio = audio + (20 * np.log10(params["volume_factor"]))

        audio = normalize(audio)
        buf = io.BytesIO()
        audio.export(buf, format="wav")
        return buf.getvalue(), params
    except Exception as exc:
        print(f"Error aplicando variación {variacion_tipo}: {exc}")
        return None, {}


def save_audio_file(
    audio_bytes: bytes,
    dataset_name: str,
    word: str,
    filename: str,
    base_folder: str = "audios",
) -> str | None:
    """Guarda bytes de audio en data/<base_folder>/<dataset_name>/<word>/<filename>."""
    try:
        repo_root = Path(__file__).parent.parent
        base_dir = repo_root / "data" / base_folder / dataset_name / word
        base_dir.mkdir(parents=True, exist_ok=True)
        file_path = base_dir / filename
        with file_path.open("wb") as f:
            f.write(audio_bytes)
        return str(file_path.relative_to(repo_root))
    except Exception as exc:
        print(f"Error guardando audio {filename}: {exc}")
        return None


def generar_variaciones_completas(
    texto: str,
    idioma: str,
    num_variaciones: int,
    metodo_sintesis: str = "gtts",
    dataset_name: str = "custom_dataset",
    rangos: dict | None = None,
    base_folder: str = "audios",
) -> list[dict]:
    """Genera audio base + ``num_variaciones`` variaciones augmentadas.

    Retorna lista de dicts con metadata de cada muestra generada.
    """
    from pydub import AudioSegment

    if rangos is None:
        from training.config import load_master_dataset_config
        config = load_master_dataset_config()
        rangos = config.get("configuracion_audio", {}).get(
            "rangos", {"pitch": [0.8, 1.3], "speed": [0.7, 1.4], "volume": [0.8, 1.2]}
        )

    audio_base = generar_audio_segun_metodo(texto, metodo_sintesis, idioma)
    if not audio_base:
        return []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_original = f"{texto}_original_{timestamp}.wav"
    file_path_original = save_audio_file(audio_base, dataset_name, texto, filename_original, base_folder)
    if not file_path_original:
        return []

    duracion_ms = len(AudioSegment.from_wav(io.BytesIO(audio_base)))
    resultados = [
        {
            "file_path": file_path_original,
            "duracion_ms": duracion_ms,
            "timestamp": datetime.now().isoformat(),
            "tipo": "original",
            "metodo_sintesis": metodo_sintesis,
            "pitch_factor": 1.0,
            "speed_factor": 1.0,
            "volume_factor": 1.0,
        }
    ]

    for i in range(num_variaciones):
        tipo = "multifactor"
        audio_var, params = aplicar_variaciones_audio(audio_base, tipo, rangos)
        if audio_var:
            fn = f"{texto}_{tipo}_{i}_{timestamp}.wav"
            fp = save_audio_file(audio_var, dataset_name, texto, fn, base_folder)
            dur = len(AudioSegment.from_wav(io.BytesIO(audio_var)))
            resultados.append(
                {
                    "file_path": fp,
                    "duracion_ms": dur,
                    "timestamp": datetime.now().isoformat(),
                    "tipo": tipo,
                    "metodo_sintesis": metodo_sintesis,
                    **params,
                }
            )

    return resultados



