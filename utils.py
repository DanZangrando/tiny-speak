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

# Constantes globales
WAV2VEC_SR = 16000
WAV2VEC_DIM = 768
WAV2VEC_HZ = 49
BATCH_SIZE = 32

# Lista de letras
import string
LETTERS = list(string.ascii_lowercase)