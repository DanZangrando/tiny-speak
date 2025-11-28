"""
Utilidades para TinySpeak
"""
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import io
import subprocess
from transformers import Wav2Vec2Model, Wav2Vec2Config
import os
from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
import streamlit as st
import tempfile
import librosa
import soundfile as sf
from gtts import gTTS
from pydub import AudioSegment
from pydub.effects import normalize
import random
from datetime import datetime
from training.config import load_master_dataset_config

def encontrar_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def load_wav2vec_model(model_name="facebook/wav2vec2-base-es-voxpopuli-v2", device="cpu"):
    """Carga el modelo Wav2Vec2 con configuración para extraer estados ocultos"""
    config = Wav2Vec2Config.from_pretrained(model_name)
    config.output_hidden_states = True
    model = Wav2Vec2Model.from_pretrained(model_name, config=config)
    model = model.to(device)
    return model

def synthesize_word(word, voice="es", rate=80, pitch=70, amplitude=120):
    """Sintetiza una palabra usando espeak"""
    try:
        import tempfile
        
        # Crear archivo temporal para el audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_path = tmp_file.name
        
        cmd = [
            "espeak",
            "-v", voice,
            "-s", str(rate),
            "-p", str(pitch),
            "-a", str(amplitude),
            "-w", tmp_path,  # Escribir a archivo en lugar de stdout
            word
        ]
        
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        proc.wait()
        
        if os.path.exists(tmp_path):
            waveform = load_waveform(tmp_path)
            os.remove(tmp_path)  # Limpiar archivo temporal
            return waveform
        else:
            return None
            
    except Exception as e:
        print(f"Error synthesizing word '{word}': {e}")
        return None

def load_waveform(audio_path, target_sr=16000):
    """Carga un archivo de audio y lo convierte al formato requerido"""
    try:
        # Intentar primero con torchaudio
        try:
            # Si es un BytesIO, guardarlo temporalmente
            if hasattr(audio_path, 'read'):
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(audio_path.read())
                    tmp_path = tmp_file.name
                
                # Cargar desde el archivo temporal
                waveform, sample_rate = torchaudio.load(tmp_path)
                os.remove(tmp_path)  # Limpiar archivo temporal
            else:
                # Cargar directamente si es una ruta
                waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convertir a tensor de PyTorch si no lo es
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.tensor(waveform, dtype=torch.float32)
        
        except Exception:
            # Fallback a librosa si torchaudio falla
            import librosa
            
            if hasattr(audio_path, 'read'):
                import tempfile
                audio_path.seek(0)  # Asegurar que estamos al inicio
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(audio_path.read())
                    tmp_path = tmp_file.name
                
                waveform, sample_rate = librosa.load(tmp_path, sr=target_sr, mono=True)
                os.remove(tmp_path)  # Limpiar archivo temporal
            else:
                waveform, sample_rate = librosa.load(audio_path, sr=target_sr, mono=True)
            
            # Convertir a tensor de PyTorch
            waveform = torch.tensor(waveform, dtype=torch.float32)
        
        # Convertir a mono si es estéreo
        if len(waveform.shape) > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze()
        
        # Remuestrear si es necesario (solo si no usamos librosa que ya remuestrea)
        if sample_rate != target_sr and 'librosa' not in str(type(sample_rate)):
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)
            
        return waveform
    except Exception as e:
        print(f"Error loading waveform from {audio_path}: {e}")
        return None

def plot_waveform(waveform, title="Waveform", sample_rate=16000):
    """Grafica una forma de onda"""
    if len(waveform.shape) > 1:
        waveform = waveform.squeeze()
    
    time_axis = torch.arange(0, len(waveform)) / sample_rate
    
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(time_axis.numpy(), waveform.numpy())
    ax.set_title(title)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud")
    ax.grid(True)
    
    return fig

def plot_logits(logits, words, title="Predicciones"):
    """Grafica los logits de predicción"""
    if isinstance(logits, torch.Tensor):
        values = logits.squeeze().cpu().numpy()
    else:
        values = np.array(logits)
    
    # Ordenar por valor descendente
    order = np.argsort(values)[::-1]
    sorted_words = [words[i] for i in order[:10]]  # Top 10
    sorted_values = values[order[:10]]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(sorted_words)), sorted_values)
    ax.set_xlabel('Palabras')
    ax.set_ylabel('Logits')
    ax.set_title(title)
    ax.set_xticks(range(len(sorted_words)))
    ax.set_xticklabels(sorted_words, rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Colorear la barra más alta
    bars[0].set_color('red')
    
    plt.tight_layout()
    return fig

def ensure_data_downloaded():
    """Asegura que los datos estén descargados"""
    from pathlib import Path
    import gdown
    import tarfile
    
    data_path = Path("data")
    data_path.mkdir(exist_ok=True)
    
    # Archivos a descargar
    files_to_download = {
        "tiny-kalulu-200.tar.xz": "1ItYeR5WdJVtXYbwUhbeKqvaKXGFAmN_4",
        "tiny-phones-200.tar.xz": "1V81aQxnww5nWNcDQJHbGmiijT9Lz-mgF",
        "tiny-emnist-26.tar.xz": "1fRUtxB05-77koJ9ZpamWk-LWu_NP0aR8",
    }
    
    for filename, file_id in files_to_download.items():
        tar_path = Path(filename)
        extracted_path = data_path / filename.replace(".tar.xz", "")
        
        # Descargar si no existe el archivo comprimido
        if not tar_path.exists():
            print(f"Descargando {filename}...")
            try:
                gdown.download(id=file_id, output=str(tar_path))
            except Exception as e:
                print(f"Error descargando {filename}: {e}")
                continue
        
        # Extraer si no existe la carpeta
        if not extracted_path.exists():
            print(f"Extrayendo {filename}...")
            try:
                with tarfile.open(tar_path, mode="r:xz") as tar:
                    tar.extractall(path=data_path)
            except Exception as e:
                print(f"Error extrayendo {filename}: {e}")
    
    return data_path

def save_waveform_to_audio_file(waveform, file_path, sample_rate=16000):
    """
    Guarda un waveform como archivo de audio con fallback robusto
    """
    try:
        # Intentar con torchaudio primero
        import torchaudio
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        torchaudio.save(file_path, waveform, sample_rate)
        return True
    except Exception:
        try:
            # Fallback con soundfile
            import soundfile as sf
            waveform_np = waveform.cpu().numpy() if hasattr(waveform, 'cpu') else waveform
            if len(waveform_np.shape) > 1:
                waveform_np = waveform_np.squeeze()
            sf.write(file_path, waveform_np, sample_rate)
            return True
        except Exception as e:
            print(f"Error guardando audio: {e}")
            return False

def plot_waveform_native(waveform, title="Waveform", sample_rate=16000):
    """
    Crea un gráfico nativo de Streamlit para el waveform usando plotly
    """
    import streamlit as st
    import plotly.graph_objects as go
    import numpy as np
    
    # Convertir a numpy si es tensor
    if hasattr(waveform, 'cpu'):
        waveform_np = waveform.cpu().numpy()
    else:
        waveform_np = np.array(waveform)
    
    if len(waveform_np.shape) > 1:
        waveform_np = waveform_np.squeeze()
    
    # Crear eje de tiempo
    time_axis = np.linspace(0, len(waveform_np) / sample_rate, len(waveform_np))
    
    # Crear gráfico
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=waveform_np,
        mode='lines',
        name='Waveform',
        line=dict(color='#FF6B6B', width=1)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Tiempo (s)',
        yaxis_title='Amplitud',
        height=300,
        template='plotly_dark',
        showlegend=False
    )
    
    return fig

def plot_logits_native(logits, words, title="Predicciones del Modelo"):
    """
    Crea un gráfico nativo de Streamlit para logits usando plotly
    """
    import streamlit as st
    import plotly.graph_objects as go
    import torch
    import numpy as np
    
    # Convertir logits a probabilidades
    if isinstance(logits, torch.Tensor):
        probs = torch.softmax(logits, dim=-1)
        probs_np = probs.cpu().numpy().squeeze()
    else:
        probs_np = np.array(logits)
    
    # Obtener top 10 predicciones
    top_indices = np.argsort(probs_np)[-10:][::-1]
    top_words = [words[i] for i in top_indices]
    top_probs = probs_np[top_indices]
    
    # Crear gráfico de barras
    fig = go.Figure(data=[
        go.Bar(
            y=top_words,
            x=top_probs,
            orientation='h',
            marker=dict(
                color=top_probs,
                colorscale='Viridis',
                colorbar=dict(title="Probabilidad")
            )
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title='Probabilidad',
        yaxis_title='Palabras',
        height=400,
        template='plotly_dark'
    )
    
    return fig

def get_default_words():
    """Retorna la lista de palabras por defecto para el modelo"""
    try:
        data_path = ensure_data_downloaded()
        kalulu_path = data_path / "tiny-kalulu-200" / "val"
        
        if kalulu_path.exists():
            words = [
                d for d in sorted(os.listdir(kalulu_path))
                if not d.startswith(".") and os.path.isdir(kalulu_path / d)
            ]
            return words
    except:
        pass
    
    # Palabras por defecto si no se pueden cargar
    return [
        'agua', 'amor', 'azul', 'bailar', 'barco', 'blanco', 'bosque', 'cama', 'campo',
        'cantar', 'casa', 'cielo', 'color', 'correr', 'dormir', 'escuela', 'estrella',
        'familia', 'feliz', 'flor', 'fuego', 'grande', 'hermano', 'historia', 'hombre',
        'jardín', 'juego', 'leche', 'libro', 'lluvia', 'lugar', 'luna', 'madre', 'mar',
        'mesa', 'montaña', 'música', 'niño', 'noche', 'número', 'ojo', 'papel', 'palabra',
        'parecer', 'parte', 'pequeño', 'perro', 'persona', 'piedra', 'puerta', 'río',
        'rojo', 'señor', 'sol', 'tiempo', 'tierra', 'trabajar', 'verde', 'vida', 'viento'
    ]

def list_checkpoints(model_type: str) -> list[dict]:
    """
    Lista los checkpoints disponibles para un tipo de modelo.
    model_type: 'listener' o 'recognizer'
    """
    import json
    from datetime import datetime
    
    models_dir = Path.cwd() / "models" / model_type
    if not models_dir.exists():
        return []
        
    checkpoints = []
    for ckpt_path in models_dir.glob("*.ckpt"):
        meta_path = ckpt_path.with_suffix(".ckpt.meta.json")
        meta = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except:
                pass
        
        # Si no hay metadata, intentar inferir del nombre
        if not meta:
            meta = {
                "timestamp": ckpt_path.stat().st_mtime,
                "config": {}
            }
            
        checkpoints.append({
            "path": str(ckpt_path),
            "filename": ckpt_path.name,
            "timestamp": meta.get("timestamp", 0),
            "meta": meta
        })
        
    # Ordenar por fecha descendente
    checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
    return checkpoints

def save_training_metrics(model_type: str, name: str, data: dict) -> str:
    """Guarda las métricas de entrenamiento en un archivo JSON."""
    import json
    
    metrics_dir = Path.cwd() / "metrics" / model_type
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitizar nombre
    safe_name = "".join([c for c in name if c.isalpha() or c.isdigit() or c in (' ', '-', '_')]).rstrip()
    filename = f"{safe_name}.json"
    file_path = metrics_dir / filename
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    return str(file_path)

def load_training_metrics(model_type: str, filename: str) -> dict:
    """Carga métricas de entrenamiento desde un archivo JSON."""
    import json
    
    metrics_dir = Path.cwd() / "metrics" / model_type
    file_path = metrics_dir / filename
    
    if not file_path.exists():
        return {}
        
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def list_metrics_files(model_type: str) -> list[dict]:
    """Lista los archivos de métricas disponibles."""
    import json
    from datetime import datetime
    
    metrics_dir = Path.cwd() / "metrics" / model_type
    if not metrics_dir.exists():
        return []
        
    files = []
    for json_path in metrics_dir.glob("*.json"):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                timestamp = data.get("timestamp", json_path.stat().st_mtime)
                files.append({
                    "filename": json_path.name,
                    "path": str(json_path),
                    "timestamp": timestamp,
                    "config": data.get("config", {})
                })
        except:
            continue
            
    files.sort(key=lambda x: x["timestamp"], reverse=True)
    return files

def delete_run_artifacts(model_type: str, metrics_filename: str) -> bool:
    """Elimina el archivo de métricas y el checkpoint asociado si existe."""
    metrics_dir = Path.cwd() / "metrics" / model_type
    models_dir = Path.cwd() / "models" / model_type
    
    json_path = metrics_dir / metrics_filename
    if not json_path.exists():
        return False
        
    # Intentar borrar JSON
    try:
        json_path.unlink()
    except:
        return False
        
    # Intentar borrar checkpoint asociado (mismo nombre base)
    ckpt_name = metrics_filename.replace(".json", ".ckpt")
    ckpt_path = models_dir / ckpt_name
    if ckpt_path.exists():
        try:
            ckpt_path.unlink()
            # Borrar metadata si existe
            meta_path = ckpt_path.with_suffix(".ckpt.meta.json")
            if meta_path.exists():
                meta_path.unlink()
        except:
            pass
            
    return True

# Constantes globales
WAV2VEC_SR = 16000
WAV2VEC_DIM = 256
WAV2VEC_HZ = 49
BATCH_SIZE = 32

# Lista de letras
import string
LETTERS = list(string.ascii_lowercase)

def save_model_metadata(ckpt_path, config, metrics):
    """Guarda metadatos del modelo para su gestión."""
    import time
    from datetime import datetime
    import json
    
    meta_path = Path(ckpt_path).with_suffix(".ckpt.meta.json")
    data = {
        "timestamp": time.time(),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,
        "metrics": metrics
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

class RealTimePlotCallback(pl.Callback):
    """Callback para actualizar gráficas de Streamlit en tiempo real durante el entrenamiento."""
    def __init__(self, placeholder_loss, placeholder_acc):
        self.placeholder_loss = placeholder_loss
        self.placeholder_acc = placeholder_acc
        self.history = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Recopilar métricas
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v 
                  for k, v in trainer.callback_metrics.items()}
        metrics['epoch'] = trainer.current_epoch
        self.history.append(metrics)
        
        # Crear DataFrame
        df = pd.DataFrame(self.history)
        
        # Actualizar Gráfica de Pérdida (Loss)
        loss_cols = [c for c in df.columns if 'loss' in c]
        if loss_cols:
            self.placeholder_loss.line_chart(df[loss_cols])
            
        # Actualizar Gráfica de Precisión (Accuracy/Top1)
        acc_cols = [c for c in df.columns if 'acc' in c or 'top1' in c]
        if acc_cols:
            self.placeholder_acc.line_chart(df[acc_cols])

class ReaderPredictionCallback(pl.Callback):
    """Callback para visualizar predicciones del Reader en tiempo real."""
    def __init__(self, val_loader, placeholder):
        self.val_loader = val_loader
        self.placeholder = placeholder
        self.batch = next(iter(val_loader)) # Pre-cargar un batch fijo para consistencia visual

    def on_validation_epoch_end(self, trainer, pl_module):
        try:
            # Obtener predicciones
            words, logits = pl_module.get_predictions(self.batch)
            
            # Procesar resultados
            probs = torch.softmax(logits, dim=-1)
            top1_probs, top1_indices = torch.max(probs, dim=-1)
            
            predicted_words = [pl_module.class_names[idx] for idx in top1_indices]
            
            # Crear DataFrame para visualización
            results = []
            for i in range(min(len(words), 10)): # Mostrar max 10 ejemplos
                is_correct = words[i] == predicted_words[i]
                icon = "✅" if is_correct else "❌"
                results.append({
                    "Real": words[i],
                    "Predicción": predicted_words[i],
                    "Confianza": f"{top1_probs[i].item():.2%}",
                    "Estado": icon
                })
                
            df = pd.DataFrame(results)
            
            # Actualizar UI
            with self.placeholder.container():
                st.markdown(f"### 🔮 Predicciones (Época {trainer.current_epoch})")
                st.dataframe(df, hide_index=True, use_container_width=True)
                
                # Métrica resumen
                acc = sum([1 for r in results if r["Estado"] == "✅"]) / len(results)
                st.progress(acc, text=f"Precisión en Batch de Muestra: {acc:.1%}")
                
        except Exception as e:
            print(f"Error en visualización: {e}")

# ==========================================
# AUDIO GENERATION UTILS
# ==========================================

def change_speed(audio_segment, speed=1.0):
    """Cambia la velocidad del audio usando pydub"""
    new_sample_rate = int(audio_segment.frame_rate * speed)
    return audio_segment._spawn(audio_segment.raw_data, overrides={
        "frame_rate": new_sample_rate
    }).set_frame_rate(audio_segment.frame_rate)

def generar_audio_gtts(texto, idioma='es', velocidad=1.0):
    """Genera audio usando Google Text-to-Speech (gTTS)"""
    try:
        tts = gTTS(text=texto, lang=idioma, slow=False)
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            tts.save(temp_file.name)
            audio = AudioSegment.from_mp3(temp_file.name)
            if velocidad != 1.0:
                audio = change_speed(audio, velocidad)
            audio = normalize(audio)
            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_bytes = wav_buffer.getvalue()
            os.unlink(temp_file.name)
            return wav_bytes
    except Exception as e:
        st.error(f"Error generando audio con gTTS: {e}")
        return None

def save_audio_file(audio_bytes, dataset_name, word, filename):
    """Guarda el archivo de audio en el sistema de archivos"""
    try:
        base_dir = Path.cwd() / "data" / "audios" / dataset_name / word
        base_dir.mkdir(parents=True, exist_ok=True)
        file_path = base_dir / filename
        with open(file_path, 'wb') as f:
            f.write(audio_bytes)
        return str(file_path.relative_to(Path.cwd()))
    except Exception as e:
        st.error(f"Error guardando archivo de audio {filename}: {e}")
        return None

def generar_audio_espeak(texto, idioma='es', rate=80, pitch=70, amplitude=120):
    """Genera audio usando espeak"""
    try:
        import subprocess
        cmd = ['espeak', '-v', f'{idioma}', '-s', str(rate), '-p', str(pitch), '-a', str(amplitude), '-w', '/dev/stdout', texto]
        result = subprocess.run(cmd, capture_output=True, check=True)
        return result.stdout
    except Exception as e:
        st.error(f"Error con espeak: {e}")
        return None

def generar_audio_segun_metodo(texto, metodo='gtts', idioma='es', **kwargs):
    if metodo == 'gtts':
        return generar_audio_gtts(texto, idioma, kwargs.get('velocidad', 1.0))
    elif metodo == 'espeak':
        return generar_audio_espeak(texto, idioma, kwargs.get('rate', 80), kwargs.get('pitch', 70), kwargs.get('amplitude', 120))
    return None

def aplicar_variaciones_audio(audio_bytes, variacion_tipo, config_rangos=None):
    if not audio_bytes: return None, {}
    if config_rangos is None:
        config_rangos = {'pitch': [0.8, 1.3], 'speed': [0.7, 1.4], 'volume': [0.8, 1.2]}
    
    try:
        audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
        params = {'pitch_factor': 1.0, 'speed_factor': 1.0, 'volume_factor': 1.0, 'tipo': variacion_tipo}
        
        if variacion_tipo == 'pitch_alto':
            params['pitch_factor'] = random.uniform(1.1, config_rangos['pitch'][1])
            new_rate = int(audio.frame_rate * params['pitch_factor'])
            audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_rate}).set_frame_rate(22050)
        elif variacion_tipo == 'pitch_bajo':
            params['pitch_factor'] = random.uniform(config_rangos['pitch'][0], 0.9)
            new_rate = int(audio.frame_rate * params['pitch_factor'])
            audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_rate}).set_frame_rate(22050)
        elif variacion_tipo == 'rapido':
            params['speed_factor'] = random.uniform(1.1, config_rangos['speed'][1])
            audio = change_speed(audio, params['speed_factor'])
        elif variacion_tipo == 'lento':
            params['speed_factor'] = random.uniform(config_rangos['speed'][0], 0.9)
            audio = change_speed(audio, params['speed_factor'])
        elif variacion_tipo == 'fuerte':
            params['volume_factor'] = random.uniform(1.1, config_rangos['volume'][1])
            audio = audio + (20 * np.log10(params['volume_factor']))
        elif variacion_tipo == 'suave':
            params['volume_factor'] = random.uniform(config_rangos['volume'][0], 0.9)
            audio = audio + (20 * np.log10(params['volume_factor']))
            
        audio = normalize(audio)
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        return wav_buffer.getvalue(), params
    except Exception as e:
        st.error(f"Error aplicando variación {variacion_tipo}: {e}")
        return None, {}

def generar_variaciones_completas(texto, idioma, num_variaciones, metodo_sintesis='gtts', dataset_name='custom_dataset', rangos=None):
    """Genera audio original y variaciones."""
    resultados = []
    # st.write(f"🎵 Generando: {texto} ({idioma})") # Comentado para no saturar UI en batch
    
    audio_base = generar_audio_segun_metodo(texto, metodo_sintesis, idioma)
    if not audio_base: return []
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_original = f"{texto}_original_{timestamp}.wav"
    file_path_original = save_audio_file(audio_base, dataset_name, texto, filename_original)
    
    duracion_ms = len(AudioSegment.from_wav(io.BytesIO(audio_base)))
    resultados.append({
        'file_path': file_path_original, 'duracion_ms': duracion_ms, 'timestamp': datetime.now().isoformat(),
        'tipo': 'original', 'metodo_sintesis': metodo_sintesis, 'pitch_factor': 1.0, 'speed_factor': 1.0, 'volume_factor': 1.0
    })
    
    tipos_variacion = ['pitch_alto', 'pitch_bajo', 'rapido', 'lento', 'fuerte', 'suave']
    
    config = load_master_dataset_config()
    rangos = config.get('configuracion_audio', {}).get('rangos', {
        'pitch': [0.8, 1.3],
        'speed': [0.7, 1.4],
        'volume': [0.8, 1.2]
    })    
    for i in range(num_variaciones):
        tipo_var = random.choice(tipos_variacion)
        audio_variado, params = aplicar_variaciones_audio(audio_base, tipo_var, rangos)
        if audio_variado:
            filename_var = f"{texto}_{tipo_var}_{i}_{timestamp}.wav"
            file_path_var = save_audio_file(audio_variado, dataset_name, texto, filename_var)
            duracion_var_ms = len(AudioSegment.from_wav(io.BytesIO(audio_variado)))
            resultados.append({
                'file_path': file_path_var, 'duracion_ms': duracion_var_ms, 'timestamp': datetime.now().isoformat(),
                'tipo': tipo_var, 'metodo_sintesis': metodo_sintesis, **params
            })
            
    return resultados

def get_language_letters(language='es'):
    """Obtiene todas las letras de un idioma específico"""
    language_alphabets = {
        'es': list('abcdefghijklmnñopqrstuvwxyz'),  # Español
        'en': list('abcdefghijklmnopqrstuvwxyz'),   # Inglés
        'fr': list('abcdefghijklmnopqrstuvwxyzàáâäèéêëìíîïòóôöùúûü'),  # Francés básico
        'de': list('abcdefghijklmnopqrstuvwxyzäöüß'),  # Alemán básico
    }
    return language_alphabets.get(language, language_alphabets['es'])