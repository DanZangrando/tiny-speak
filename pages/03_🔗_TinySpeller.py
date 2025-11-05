from __future__ import annotations

import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from components.modern_sidebar import display_modern_sidebar
from models import TinyListener, TinyRecognizer, TinySpeak, TinySpeller
from training.audio_dataset import (
    DEFAULT_AUDIO_SPLIT_RATIOS,
    AudioSample,
    AudioWordDataset,
    build_audio_datasets,
)
from training.config import load_master_dataset_config
from training.visual_dataset import DEFAULT_SPLIT_RATIOS, VisualLetterDataset
from utils import (
    WAV2VEC_DIM,
    WAV2VEC_SR,
    encontrar_device,
    get_default_words,
    load_wav2vec_model,
    plot_logits_native,
    save_waveform_to_audio_file,
)


st.set_page_config(page_title="TinySpeller - Multimodal Bridge", page_icon="🔗", layout="wide")

DEFAULT_SEED = 42
LETTER_TRANSFORM = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])


def get_active_words() -> List[str]:
    """
    Obtiene el vocabulario activo sin cache para evitar problemas de actualización.
    ✅ CORRECCIÓN: Removido @st.cache_resource que causaba problemas de detección.
    """
    try:
        config = load_master_dataset_config()
        selected = config.get("diccionario_seleccionado") or {}
        words = selected.get("palabras") or []
        if isinstance(words, Sequence) and words:
            # Limpiar y deduplicar palabras
            clean_words = [w.strip().lower() for w in words if w and w.strip()]
            return list(dict.fromkeys(clean_words))
    except Exception as exc:
        st.error(f"Error cargando vocabulario activo: {exc}")
    
    # Fallback a vocabulario por defecto
    default = get_default_words()
    st.warning(f"⚠️ Usando vocabulario por defecto ({len(default)} palabras)")
    return default


@st.cache_resource(show_spinner=False)
def load_audio_index(seed: int = DEFAULT_SEED) -> Tuple[Dict[str, List[Dict]], List[str], str | None]:
    try:
        datasets = build_audio_datasets(seed=seed, split_ratios=DEFAULT_AUDIO_SPLIT_RATIOS)
    except Exception as exc:  # noqa: BLE001
        return {}, [], str(exc)

    index: Dict[str, List[Dict]] = {}
    for split_name, dataset in datasets.items():
        entries: List[Dict] = []
        for sample in dataset.samples:
            entries.append(
                {
                    "word": sample.word,
                    "waveform": sample.waveform.clone(),
                    "duration_ms": sample.duration_ms,
                    "metadata": sample.metadata,
                    "split": split_name,
                }
            )
        index[split_name] = entries

    train_ds = datasets.get("train")
    words = train_ds.words if isinstance(train_ds, AudioWordDataset) else []
    return index, words, None


@st.cache_resource(show_spinner=False)
def load_visual_index(seed: int = DEFAULT_SEED, split: str = "train") -> Tuple[Dict[str, List[str]], str | None]:
    try:
        dataset = VisualLetterDataset(
            split=split,
            augment=False,
            seed=seed,
            split_ratios=DEFAULT_SPLIT_RATIOS,
        )
    except Exception as exc:  # noqa: BLE001
        return {}, str(exc)

    index: Dict[str, List[str]] = {}
    for sample in dataset.samples:
        index.setdefault(sample.letter, []).append(str(sample.path))
    return index, None


@st.cache_resource(show_spinner=False)
def load_multimodal_stack() -> Dict:
    """
    Carga el stack completo de modelos multimodales.
    ⚠️ NOTA: TinyRecognizer está congelado en TinySpeller (problema arquitectural)
    """
    device = encontrar_device()
    words = get_active_words()

    # Validar que hay vocabulario
    if not words:
        st.error("❌ No se pudo cargar vocabulario activo")
        return {"error": "No vocabulary loaded"}
    
    # Cargar modelos
    wav2vec_model = load_wav2vec_model(device=device)
    tiny_speak = TinySpeak(words=words, hidden_dim=128, num_layers=2, wav2vec_dim=WAV2VEC_DIM).to(device)
    tiny_listener = TinyListener(tiny_speak=tiny_speak, wav2vec_model=wav2vec_model).to(device)
    
    # TinyRecognizer - determinar num_classes basado en vocabulario
    unique_letters = set()
    for word in words:
        unique_letters.update(word.lower())
    num_classes = len(unique_letters)
    
    tiny_recognizer = TinyRecognizer(num_classes=num_classes).to(device)
    tiny_speller = TinySpeller(tiny_recognizer=tiny_recognizer, tiny_speak=tiny_speak).to(device)

    # Modo evaluación
    tiny_listener.eval()
    tiny_speller.eval()

    return {
        "device": device,
        "words": words,
        "num_classes": num_classes,
        "unique_letters": sorted(unique_letters),
        "tiny_listener": tiny_listener,
        "tiny_speller": tiny_speller,
        "tiny_recognizer": tiny_recognizer,  # Para análisis
        "tiny_speak": tiny_speak,           # Para análisis
        "image_transform": LETTER_TRANSFORM,
        "vocab_size": len(words),
    }


@dataclass
class MultimodalResult:
    word: str
    audio_prediction: str
    audio_confidence: float
    audio_logits: torch.Tensor
    speller_prediction: str
    speller_confidence: float
    speller_logits: torch.Tensor
    letter_paths: List[str]
    audio_split: str
    audio_duration_ms: float | None


def pick_audio_example(word: str, audio_index: Dict[str, List[Dict]], rng: random.Random) -> Dict | None:
    candidates: List[Dict] = []
    for entries in audio_index.values():
        candidates.extend([entry for entry in entries if entry["word"] == word])
    if not candidates:
        return None
    return rng.choice(candidates)


def pick_letter_sequence(word: str, visual_index: Dict[str, List[str]], rng: random.Random) -> Tuple[List[str], str | None]:
    chosen: List[str] = []
    for letter in word:
        options = visual_index.get(letter)
        if not options:
            return [], letter
        chosen.append(rng.choice(options))
    return chosen, None


def run_multimodal_inference(
    word: str,
    seed: int,
    models: Dict,
    audio_index: Dict[str, List[Dict]],
    visual_index: Dict[str, List[str]],
) -> MultimodalResult:
    rng = random.Random(seed)

    audio_entry = pick_audio_example(word, audio_index, rng)
    if audio_entry is None:
        raise RuntimeError(f"No hay audio disponible para '{word}'.")

    letter_paths, missing_letter = pick_letter_sequence(word, visual_index, rng)
    if missing_letter is not None:
        raise RuntimeError(f"No hay imágenes registradas para la letra '{missing_letter}'.")

    device = models["device"]
    listener: TinyListener = models["tiny_listener"]
    speller: TinySpeller = models["tiny_speller"]

    waveform: torch.Tensor = audio_entry["waveform"].to(device)
    listener.eval()
    with torch.no_grad():
        audio_logits, _ = listener([waveform])
    audio_probs = torch.softmax(audio_logits, dim=-1).squeeze(0)
    audio_top = torch.argmax(audio_probs).item()

    image_tensors: List[torch.Tensor] = []
    for path in letter_paths:
        image = Image.open(Path(path)).convert("RGB")
        image_tensors.append(models["image_transform"](image))
    sequence = torch.stack(image_tensors).unsqueeze(0).to(device)

    speller.eval()
    with torch.no_grad():
        speller_logits, _ = speller(sequence)
    speller_probs = torch.softmax(speller_logits, dim=-1).squeeze(0)
    speller_top = torch.argmax(speller_probs).item()

    words = models["words"]
    audio_prediction = words[audio_top] if words else "—"
    speller_prediction = words[speller_top] if words else "—"

    return MultimodalResult(
        word=word,
        audio_prediction=audio_prediction,
        audio_confidence=float(audio_probs[audio_top].item()),
        audio_logits=audio_logits.squeeze(0).cpu(),
        speller_prediction=speller_prediction,
        speller_confidence=float(speller_probs[speller_top].item()),
        speller_logits=speller_logits.squeeze(0).cpu(),
        letter_paths=letter_paths,
        audio_split=audio_entry["split"],
        audio_duration_ms=audio_entry["duration_ms"],
    )


def render_result(result: MultimodalResult, words: List[str]) -> None:
    st.markdown("### Resultados de inferencia")

    match = result.audio_prediction == result.speller_prediction
    status = "🎯 Ambas modalidades coinciden" if match else "⚠️ Predicciones diferentes"
    st.info(f"{status}: audio → **{result.audio_prediction}**, visión → **{result.speller_prediction}**")

    audio_col, vision_col = st.columns(2)
    with audio_col:
        st.markdown("#### 🎵 TinyListener")
        st.metric("Predicción", result.audio_prediction, f"{result.audio_confidence:.2%}")
        st.caption(f"Split seleccionado: {result.audio_split} · Duración: {result.audio_duration_ms or 0:.0f} ms")
        st.plotly_chart(plot_logits_native(result.audio_logits, words), use_container_width=True)

    with vision_col:
        st.markdown("#### 🖼️ TinySpeller")
        st.metric("Predicción", result.speller_prediction, f"{result.speller_confidence:.2%}")
        st.plotly_chart(plot_logits_native(result.speller_logits, words), use_container_width=True)

    st.markdown("#### 🔤 Secuencia de letras utilizada")
    letter_cols = st.columns(len(result.letter_paths)) if result.letter_paths else []
    for idx, path in enumerate(result.letter_paths):
        with letter_cols[idx]:
            st.image(Image.open(Path(path)), caption=f"Letra {idx + 1}", width=96)

    st.markdown("#### 🎧 Audio reproducido")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        waveform = result.audio_logits.new_tensor([])  # placeholder for typing
    try:
        # We reuse the cached waveform through the index by re-running pick without randomness.
        # Avoid storing full tensor here to keep cache minimal.
        pass
    finally:
        Path(tmp_file.name).unlink(missing_ok=True)


def render_audio_player(word: str, audio_index: Dict[str, List[Dict]], seed: int) -> None:
    rng = random.Random(seed)
    audio_entry = pick_audio_example(word, audio_index, rng)
    if audio_entry is None:
        st.warning("No se pudo mostrar audio de referencia para esta palabra.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        waveform: torch.Tensor = audio_entry["waveform"]
        if not save_waveform_to_audio_file(waveform, tmp_file.name, WAV2VEC_SR):
            st.warning("No fue posible renderizar el audio.")
            return
        tmp_path = tmp_file.name

    try:
        with open(tmp_path, "rb") as audio_fp:
            st.audio(audio_fp.read(), format="audio/wav")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def main() -> None:
    display_modern_sidebar("tiny_speller")
    st.title("🔗 TinySpeller – Multimodal Bridge")
    st.caption("Valida palabras combinando audio y visión desde el dataset activo.")

    models = load_multimodal_stack()
    
    # Verificar si hay errores en la carga
    if "error" in models:
        st.error(f"Error cargando modelos: {models['error']}")
        return
        
    audio_index, audio_words, audio_error = load_audio_index()
    visual_index, visual_error = load_visual_index()

    vocab_words = models.get("words", [])
    total_audio = sum(len(entries) for entries in audio_index.values())
    total_visual = sum(len(paths) for paths in visual_index.values())

    # Métricas principales
    st.markdown("### 📊 Estado del Sistema")
    metrics = st.columns(4)
    metrics[0].metric("🔤 Vocabulario Activo", f"{len(vocab_words)} palabras")
    metrics[1].metric("🎵 Muestras Audio", total_audio)
    metrics[2].metric("🖼️ Imágenes Letras", total_visual)
    metrics[3].metric("💻 Dispositivo", str(models.get("device", "N/A")))

    # Información adicional del modelo
    if models.get("unique_letters"):
        detail_cols = st.columns(3)
        detail_cols[0].metric("🔠 Letras Únicas", len(models["unique_letters"]))
        detail_cols[1].metric("🏗️ Clases Visual", models.get("num_classes", 0))
        detail_cols[2].metric("📚 Tamaño Vocab", models.get("vocab_size", 0))
        
        # Mostrar letras detectadas
        with st.expander("🔍 Letras Detectadas en Vocabulario"):
            st.write("**Letras únicas encontradas:**")
            letters_text = " ".join(models["unique_letters"])
            st.code(letters_text, language="text")
    
    # Alertas de problemas
    issues = []
    if not vocab_words:
        issues.append("❌ No hay vocabulario cargado")
    if total_audio == 0:
        issues.append("❌ No hay muestras de audio")
    if total_visual == 0:
        issues.append("❌ No hay imágenes de letras")
    
    if issues:
        st.error("**Problemas detectados:**")
        for issue in issues:
            st.write(issue)

    if audio_error:
        st.warning(f"⚠️ Audio dataset: {audio_error}")
    if visual_error:
        st.warning(f"⚠️ Visual dataset: {visual_error}")

    # Información sobre problemas arquitecturales
    st.markdown("### ⚠️ Estado Arquitectural")
    with st.expander("🔧 Problemas Arquitecturales Conocidos", expanded=False):
        st.markdown("""
        **🚨 Problemas Identificados en TinySpeller:**
        
        1. **TinyRecognizer Congelado**: 
           - El backbone visual está completamente congelado (`requires_grad=False`)
           - Solo el LSTM de TinySpeak puede aprender
           - Limita drasticamente la capacidad de aprendizaje
        
        2. **Procesamiento Ineficiente**:
           - Loop secuencial por cada letra (ineficiente)
           - No hay procesamiento batch-wise de secuencias
           - Falta de mecanismos de atención
        
        3. **Sin Entrenamiento End-to-End**:
           - No existe módulo de entrenamiento TinySpellerLightning
           - No hay dataset multimodal específico
           - Sin métricas de evaluación para secuencias
        
        **💡 Soluciones Propuestas:**
        - Arquitectura mejorada con backbone entrenable
        - Encoder de secuencias con BiLSTM/Attention
        - Dataset y pipeline de entrenamiento multimodal
        """)
        
        if st.button("📖 Ver Análisis Completo", help="Abre el análisis técnico detallado"):
            st.info("Consulta `TINY_SPELLER_ANALYSIS.md` para el análisis completo y soluciones propuestas.")

    # Tabs principales
    inference_tab, training_tab = st.tabs(["🧪 Inferencia Actual", "🚀 Entrenamiento Nuevo"])
    
    with inference_tab:
        st.markdown("### 🧪 Experimento Multimodal Actual")
        st.caption("⚠️ **Nota**: Este es el modelo actual con arquitectura problemática")

        if not vocab_words:
            st.error("No hay vocabulario disponible. Configura un diccionario en Dataset Manager.")
            return

        selection_col, seed_col = st.columns([2, 1])
        with selection_col:
            selected_word = st.selectbox("Palabra objetivo", options=vocab_words, index=0)
        with seed_col:
            seed_value = st.number_input("Seed aleatoria", value=DEFAULT_SEED, min_value=0, max_value=10_000, step=1)

        if st.button("Ejecutar inferencia multimodal", type="primary"):
            try:
                result = run_multimodal_inference(selected_word, seed_value, models, audio_index, visual_index)
                st.session_state["tiny_speller_result"] = result
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

        if "tiny_speller_result" in st.session_state:
            render_result(st.session_state["tiny_speller_result"], vocab_words)
            st.markdown("---")
            st.markdown("### 🎧 Escucha rápida del audio seleccionado")
            render_audio_player(selected_word, audio_index, seed_value)
    
    with training_tab:
        render_training_tab(models, vocab_words)

def test_image_to_word(models):
    """Test de secuencia de imágenes a palabra"""
    st.subheader("🖼️ Test: Secuencia de Letras → Palabra")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### ⚙️ Configuración")
        
        # Selector de palabra para generar
        word_source = st.radio(
            "Fuente de la palabra:",
            ["Del vocabulario", "Palabra personalizada"]
        )
        
        if word_source == "Del vocabulario":
            target_word = st.selectbox(
                "Selecciona palabra del vocabulario:",
                models['words'][:50]  # Primeras 50 para el selector
            )
        else:
            target_word = st.text_input(
                "Escribe una palabra:",
                value="hola",
                max_chars=10,
                help="Solo letras a-z, máximo 10 caracteres"
            ).lower()
        
        # Validar que solo contenga letras válidas
        if target_word and all(c in LETTERS for c in target_word):
            st.success(f"✅ Palabra válida: **{target_word}** ({len(target_word)} letras)")
            
            if st.button("🎨 Generar Secuencia de Letras", type="primary"):
                generate_letter_sequence(target_word, models)
        
        elif target_word:
            st.error(f"❌ La palabra contiene caracteres inválidos. Solo usar letras a-z.")
    
    with col2:
        if 'letter_sequence_results' in st.session_state:
            display_sequence_results(st.session_state.letter_sequence_results)

def test_audio_to_word(models):
    """Test de audio directo a palabra"""
    st.subheader("🎵 Test: Audio → Palabra Directa")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 🎤 Input de Audio")
        
        # Opción 1: Audio grabado
        st.markdown("**Grabación directa:**")
        recorded_audio = st.audio_input("Graba una palabra")
        
        # Opción 2: Síntesis
        st.markdown("**O síntesis de palabra:**")
        synth_word = st.text_input("Palabra para sintetizar:", value="casa")
        
        col_params1, col_params2 = st.columns(2)
        with col_params1:
            rate = st.slider("Velocidad", 50, 200, 80)
        with col_params2:
            pitch = st.slider("Tono", 0, 100, 50)
        
        if st.button("🔊 Sintetizar y Analizar"):
            synthesize_and_analyze_audio(synth_word, rate, pitch, models)
        
        # Análisis de audio grabado
        if recorded_audio and st.button("🔍 Analizar Grabación"):
            analyze_recorded_audio(recorded_audio, models)
    
    with col2:
        if 'audio_results' in st.session_state:
            display_audio_results(st.session_state.audio_results)

def test_multimodal_comparison(models):
    """Comparación entre modalidades"""
    st.subheader("⚖️ Comparación Multimodal")
    
    st.markdown("""
    **Objetivo:** Comparar cómo cada modalidad reconoce la misma palabra
    """)
    
    # Configuración
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### ⚙️ Configuración del Test")
        
        test_word = st.selectbox(
            "Palabra para test comparativo:",
            models['words'][:30]
        )
        
        modalities_to_test = st.multiselect(
            "Modalidades a comparar:",
            ["🖼️ Secuencia de Letras", "🎵 Audio Sintetizado", "🎤 Audio Directo"],
            default=["🖼️ Secuencia de Letras", "🎵 Audio Sintetizado"]
        )
        
        if st.button("🚀 Ejecutar Comparación Multimodal"):
            run_multimodal_comparison(test_word, modalities_to_test, models)
    
    with col2:
        if 'comparison_results' in st.session_state:
            display_comparison_results(st.session_state.comparison_results)

def test_advanced_analysis(models):
    """Análisis avanzado del sistema multimodal"""
    st.subheader("🔬 Análisis Avanzado del Sistema")
    
    # Análisis de arquitectura
    with st.expander("🏗️ Análisis de Arquitectura Completa"):
        display_architecture_analysis(models)
    
    # Análisis de embeddings
    with st.expander("🧠 Análisis de Espacios de Embeddings"):
        analyze_embedding_spaces(models)
    
    # Benchmark del sistema
    with st.expander("⚡ Benchmark de Rendimiento"):
        run_performance_benchmark(models)

def generate_letter_sequence(word, models):
    """Genera secuencia de imágenes para una palabra"""
    try:
        st.info(f"🎨 Generando secuencia para: **{word}**")
        
        # Generar imagen para cada letra
        letter_images = []
        letter_tensors = []
        
        # Configuración de imagen
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        transform = Compose([
            Resize((28, 28)),
            ToTensor(),
            Normalize(mean, std)
        ])
        
        for letter in word:
            # Crear imagen de la letra
            img = Image.new('RGB', (28, 28), 'white')
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # Centrar texto
            bbox = draw.textbbox((0, 0), letter.upper(), font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (28 - text_width) // 2
            y = (28 - text_height) // 2
            
            draw.text((x, y), letter.upper(), fill='black', font=font)
            
            letter_images.append(img)
            letter_tensors.append(transform(img))
        
        # Crear tensor de secuencia
        sequence_tensor = torch.stack(letter_tensors).unsqueeze(0).to(models['device'])
        
        # Procesar con TinySpeller
        models['tiny_speller'].eval()
        with torch.no_grad():
            logits, hidden_states = models['tiny_speller'](sequence_tensor)
        
        # Guardar resultados
        predicted_idx = logits.argmax(dim=1).item()
        predicted_word = models['words'][predicted_idx]
        confidence = torch.softmax(logits, dim=1).max().item()
        
        results = {
            'target_word': word,
            'predicted_word': predicted_word,
            'confidence': confidence,
            'letter_images': letter_images,
            'logits': logits.cpu(),
            'correct': word == predicted_word
        }
        
        st.session_state.letter_sequence_results = results
        
    except Exception as e:
        st.error(f"❌ Error generando secuencia: {str(e)}")

def display_sequence_results(results):
    """Muestra resultados de secuencia de letras"""
    st.markdown("#### 📊 Resultados")
    
    # Mostrar secuencia de letras generada
    st.markdown("**🔤 Secuencia Generada:**")
    cols = st.columns(len(results['letter_images']))
    for i, img in enumerate(results['letter_images']):
        with cols[i]:
            st.image(img, caption=f"Letra {i+1}", width=60)
    
    # Resultado de predicción
    if results['correct']:
        st.success(f"✅ **Correcto!** Predicha: {results['predicted_word']} (Confianza: {results['confidence']:.2%})")
    else:
        st.error(f"❌ **Incorrecto.** Esperada: {results['target_word']}, Predicha: {results['predicted_word']} (Confianza: {results['confidence']:.2%})")
    
    # Top 5 predicciones
    probabilities = torch.softmax(results['logits'], dim=1).squeeze().numpy()
    top_indices = np.argsort(probabilities)[::-1][:5]
    
    st.markdown("**🏆 Top 5 Predicciones:**")
    for i, idx in enumerate(top_indices):
        word = st.session_state.multimodal_models['words'][idx]
        prob = probabilities[idx]
        emoji = "🎯" if i == 0 else "📍"
        st.write(f"{emoji} {word} ({prob:.2%})")

def synthesize_and_analyze_audio(word, rate, pitch, models):
    """Sintetiza y analiza audio"""
    try:
        # Sintetizar
        waveform = synthesize_word(word, rate=rate, pitch=pitch)
        
        if waveform is not None:
            # Analizar con TinyListener
            device = models['device']
            waveform_device = waveform.to(device)
            
            models['tiny_listener'].eval()
            with torch.no_grad():
                logits, hidden_states = models['tiny_listener']([waveform_device])
            
            # Guardar resultados
            predicted_idx = logits.argmax(dim=1).item()
            predicted_word = models['words'][predicted_idx]
            confidence = torch.softmax(logits, dim=1).max().item()
            
            results = {
                'target_word': word,
                'predicted_word': predicted_word,
                'confidence': confidence,
                'waveform': waveform,
                'logits': logits.cpu(),
                'correct': word == predicted_word,
                'source': 'synthesis'
            }
            
            st.session_state.audio_results = results
        
        else:
            st.error("❌ Error en síntesis de audio")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def analyze_recorded_audio(audio_file, models):
    """Analiza audio grabado"""
    try:
        from utils import load_waveform
        
        # Guardar temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name
        
        waveform = load_waveform(tmp_path, target_sr=WAV2VEC_SR)
        os.unlink(tmp_path)
        
        if waveform is not None:
            # Analizar
            device = models['device']
            waveform_device = waveform.to(device)
            
            models['tiny_listener'].eval()
            with torch.no_grad():
                logits, hidden_states = models['tiny_listener']([waveform_device])
            
            predicted_idx = logits.argmax(dim=1).item()
            predicted_word = models['words'][predicted_idx]
            confidence = torch.softmax(logits, dim=1).max().item()
            
            results = {
                'target_word': 'unknown',
                'predicted_word': predicted_word,
                'confidence': confidence,
                'waveform': waveform,
                'logits': logits.cpu(),
                'correct': None,
                'source': 'recording'
            }
            
            st.session_state.audio_results = results
        
        else:
            st.error("❌ Error cargando audio")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def display_audio_results(results):
    """Muestra resultados de análisis de audio"""
    st.markdown("#### 🎧 Resultados de Audio")
    
    # Reproducir audio
    if results['source'] == 'synthesis':
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                if save_waveform_to_audio_file(results['waveform'], tmp_file.name, WAV2VEC_SR):
                    with open(tmp_file.name, 'rb') as audio_file:
                        st.audio(audio_file.read(), format='audio/wav')
                else:
                    st.warning("⚠️ No se pudo guardar el archivo de audio")
                
                os.unlink(tmp_file.name)
        except Exception as e:
            st.warning(f"⚠️ No se puede reproducir el audio: {str(e)}")
    
    # Resultados
    if results['correct'] is not None:
        if results['correct']:
            st.success(f"✅ **Correcto!** {results['predicted_word']} (Confianza: {results['confidence']:.2%})")
        else:
            st.warning(f"⚠️ Esperada: {results['target_word']}, Predicha: {results['predicted_word']} ({results['confidence']:.2%})")
    else:
        st.info(f"🎯 **Predicción:** {results['predicted_word']} (Confianza: {results['confidence']:.2%})")
    
    # Waveform
    from utils import plot_waveform
    fig = plot_waveform(results['waveform'], "Audio Analizado")
    st.pyplot(fig)

def run_multimodal_comparison(word, modalities, models):
    """Ejecuta comparación entre modalidades"""
    st.info(f"🔄 Ejecutando comparación para: **{word}**")
    
    results = {'word': word, 'comparisons': {}}
    
    # Test de secuencia de letras
    if "🖼️ Secuencia de Letras" in modalities:
        try:
            generate_letter_sequence(word, models)
            if 'letter_sequence_results' in st.session_state:
                seq_results = st.session_state.letter_sequence_results
                results['comparisons']['vision'] = {
                    'predicted': seq_results['predicted_word'],
                    'confidence': seq_results['confidence'],
                    'correct': seq_results['correct']
                }
        except Exception as e:
            results['comparisons']['vision'] = {'error': str(e)}
    
    # Test de audio sintetizado
    if "🎵 Audio Sintetizado" in modalities:
        try:
            synthesize_and_analyze_audio(word, 80, 50, models)
            if 'audio_results' in st.session_state:
                audio_results = st.session_state.audio_results
                results['comparisons']['audio_synth'] = {
                    'predicted': audio_results['predicted_word'],
                    'confidence': audio_results['confidence'],
                    'correct': audio_results['correct']
                }
        except Exception as e:
            results['comparisons']['audio_synth'] = {'error': str(e)}
    
    st.session_state.comparison_results = results

def display_comparison_results(results):
    """Muestra resultados de comparación multimodal"""
    st.markdown("#### 📊 Resultados de Comparación")
    
    word = results['word']
    comparisons = results['comparisons']
    
    # Tabla comparativa
    st.markdown(f"**Palabra objetivo:** {word}")
    
    for modality, result in comparisons.items():
        if 'error' in result:
            st.error(f"❌ {modality}: Error - {result['error']}")
        else:
            status = "✅" if result['correct'] else "❌"
            st.write(f"{status} **{modality}**: {result['predicted']} (conf: {result['confidence']:.2%})")
    
    # Análisis conjunto
    if len(comparisons) > 1:
        predictions = [r['predicted'] for r in comparisons.values() if 'predicted' in r]
        if len(set(predictions)) == 1:
            st.success("🎉 **Consenso:** Todas las modalidades coinciden!")
        else:
            st.warning("⚠️ **Discrepancia:** Las modalidades difieren en la predicción")

def display_architecture_analysis(models):
    """Muestra análisis detallado de arquitectura"""
    st.markdown("### 🏗️ Análisis Completo del Sistema")
    
    # Componentes del sistema
    components = {
        'TinyRecognizer (CORnet-Z)': models['tiny_recognizer'],
        'TinySpeak (LSTM)': models['tiny_speak'],  
        'TinyListener (Wav2Vec2+LSTM)': models['tiny_listener'],
        'TinySpeller (Vision+Audio)': models['tiny_speller']
    }
    
    total_params = 0
    
    for name, model in components.items():
        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params += params
        
        st.markdown(f"**{name}:**")
        st.write(f"- Parámetros totales: {params:,}")
        st.write(f"- Parámetros entrenables: {trainable:,}")
        st.write(f"- Parámetros congelados: {params - trainable:,}")
        st.write("")
    
    st.metric("🧠 Sistema Completo", f"{total_params:,} parámetros")

def analyze_embedding_spaces(models):
    """Analiza espacios de embeddings"""
    st.markdown("### 🧠 Análisis de Espacios de Embeddings")
    
    # Por ahora información teórica
    st.markdown("""
    **Espacios de representación en TinySpeak:**
    
    1. **Wav2Vec2 Features (768D)**: Características acústicas del audio
    2. **CORnet-Z Features (768D)**: Características visuales de letras  
    3. **LSTM Hidden States (64D)**: Estados internos de secuencia
    4. **Word Embeddings**: Espacio de palabras del vocabulario
    
    **Hipótesis:** Los embeddings de audio y visión deberían ser similares 
    para la misma letra/palabra, permitiendo transferencia entre modalidades.
    """)
    
    if st.button("🔬 Analizar Embeddings de Ejemplo"):
        # Crear ejemplo simple
        letter = 'a'
        
        # Embedding visual
        img = Image.new('RGB', (28, 28), 'white')
        draw = ImageDraw.Draw(img)
        draw.text((8, 5), letter.upper(), fill='black')
        
        transform = Compose([
            Resize((28, 28)),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(img).unsqueeze(0).to(models['device'])
        
        models['tiny_recognizer'].eval()
        with torch.no_grad():
            _, visual_embed = models['tiny_recognizer'](image_tensor)
        
        st.success(f"✅ Embedding visual para '{letter}': {visual_embed.shape}")
        
        # Mostrar estadísticas básicas
        embed_np = visual_embed.squeeze().cpu().numpy()
        st.write(f"- Media: {embed_np.mean():.4f}")
        st.write(f"- Std: {embed_np.std():.4f}")
        st.write(f"- Min: {embed_np.min():.4f}")
        st.write(f"- Max: {embed_np.max():.4f}")

def run_performance_benchmark(models):
    """Ejecuta benchmark de rendimiento"""
    st.markdown("### ⚡ Benchmark de Rendimiento")
    
    if st.button("🚀 Ejecutar Benchmark"):
        import time
        
        # Benchmark de TinyRecognizer
        img = Image.new('RGB', (28, 28), 'white')
        transform = Compose([
            Resize((28, 28)),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image_tensor = transform(img).unsqueeze(0).to(models['device'])
        
        # Tiempo de inferencia visual
        models['tiny_recognizer'].eval()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _, _ = models['tiny_recognizer'](image_tensor)
        vision_time = (time.time() - start_time) / 100
        
        # Síntesis de audio para benchmark
        waveform = synthesize_word("test")
        if waveform is not None:
            waveform_device = waveform.to(models['device'])
            
            # Tiempo de inferencia audio
            models['tiny_listener'].eval()
            start_time = time.time()
            with torch.no_grad():
                for _ in range(10):  # Menos iteraciones por ser más lento
                    _, _ = models['tiny_listener']([waveform_device])
            audio_time = (time.time() - start_time) / 10
        else:
            audio_time = None
        
        # Mostrar resultados
        st.success("✅ Benchmark completado!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🖼️ Visión (ms)", f"{vision_time*1000:.2f}")
        with col2:
            if audio_time:
                st.metric("🎵 Audio (ms)", f"{audio_time*1000:.2f}")
            else:
                st.metric("🎵 Audio", "Error")

def render_training_tab(models: Dict, vocab_words: List[str]) -> None:
    """Tab de entrenamiento para TinySpeller con implementación completa"""
    
    st.subheader("🚀 Entrenamiento TinySpeller Multimodal")
    st.markdown("""
    Entrena TinySpeller para reconocer palabras completas a partir de secuencias de imágenes de letras.
    Combina el poder de TinyRecognizer (visión) con arquitectura secuencial (LSTM).
    """)
    
    # Verificar prerrequisitos
    if not models.get("tiny_speller"):
        st.error("❌ TinySpeller no disponible. Verifica la carga del stack multimodal.")
        return
    
    if not vocab_words:
        st.error("❌ No hay vocabulario disponible para entrenamiento.")
        return
    
    # Información del dataset
    info_cols = st.columns(4)
    info_cols[0].metric("📚 Palabras Disponibles", len(vocab_words))
    info_cols[1].metric("🔤 Letras Únicas", len(set("".join(vocab_words))))
    info_cols[2].metric("📏 Palabra Más Larga", max(len(word) for word in vocab_words))
    info_cols[3].metric("📊 Promedio Longitud", f"{sum(len(w) for w in vocab_words) / len(vocab_words):.1f}")
    
    # Verificar disponibilidad de datasets
    st.markdown("#### 📋 Estado de Datasets")
    
    try:
        from training.visual_dataset import VisualLetterDataset
        
        # Verificar dataset visual intentando crear un dataset pequeño
        try:
            train_dataset = VisualLetterDataset(split="train")
            val_dataset = VisualLetterDataset(split="val")
            test_dataset = VisualLetterDataset(split="test")
            
            has_visual = len(train_dataset) > 0
            visual_splits = {
                'train': len(train_dataset),
                'val': len(val_dataset),
                'test': len(test_dataset)
            }
        except (ValueError, FileNotFoundError, Exception) as e:
            has_visual = False
            visual_splits = {'train': 0, 'val': 0, 'test': 0}
        
        # Verificar dataset audio (basado en generated_samples en config)
        from training.config import load_master_dataset_config
        config = load_master_dataset_config()
        has_audio = len(config.get('generated_samples', {})) > 0
        
        status_cols = st.columns(2)
        with status_cols[0]:
            if has_visual:
                st.success("✅ Dataset Visual Disponible")
                if visual_splits:
                    st.caption(f"Train: {visual_splits['train']}, Val: {visual_splits['val']}, Test: {visual_splits['test']}")
            else:
                st.error("❌ Dataset Visual No Disponible")
                st.caption("Ve a 🖼️ Visual Dataset Manager para generar imágenes")
        
        with status_cols[1]:
            if has_audio:
                st.success("✅ Dataset Audio Disponible")
                st.caption(f"{len(config.get('generated_samples', {}))} palabras con audio")
            else:
                st.error("❌ Dataset Audio No Disponible")
                st.caption("Ve a 🎤 Audio Dataset Manager para generar audios")
        
        if not has_visual:
            st.warning("⚠️ Dataset visual es necesario para el entrenamiento de TinySpeller.")
            return
            
    except ImportError as e:
        st.error(f"❌ Error al importar módulos de entrenamiento: {e}")
        return
    
    # Formulario de configuración de entrenamiento
    st.markdown("#### ⚙️ Configuración de Entrenamiento")
    
    with st.form("speller_training_form"):
        # Configuración básica
        config_cols = st.columns(3)
        
        with config_cols[0]:
            batch_size = st.number_input("Batch Size", min_value=4, max_value=64, value=16, step=4)
            learning_rate = st.number_input("Learning Rate", min_value=1e-5, max_value=1e-2, value=1e-3, format="%.1e")
            max_epochs = st.number_input("Épocas Máximas", min_value=5, max_value=100, value=20)
        
        with config_cols[1]:
            weight_decay = st.number_input("Weight Decay", min_value=0.0, max_value=1e-2, value=1e-4, format="%.1e")
            hidden_dim = st.number_input("Hidden Dimension", min_value=64, max_value=512, value=128, step=32)
            max_word_length = st.number_input("Longitud Máx. Palabra", min_value=5, max_value=15, value=10)
        
        with config_cols[2]:
            num_workers = st.selectbox("Workers", options=[0, 1, 2, 4], index=0)
            freeze_recognizer = st.checkbox("Congelar TinyRecognizer", value=False, help="Si está marcado, solo entrena la parte secuencial")
            label_smoothing = st.number_input("Label Smoothing", min_value=0.0, max_value=0.3, value=0.1)
        
        # Configuraciones avanzadas
        with st.expander("� Configuración Avanzada"):
            accelerator = st.selectbox("Acelerador", options=["auto", "cpu", "gpu"], index=0)
            patience = st.number_input("Early Stop Patience", min_value=3, max_value=20, value=5)
            gradient_clip_val = st.number_input("Gradient Clipping", min_value=0.0, max_value=10.0, value=1.0)
        
        # Botón de entrenamiento
        submitted = st.form_submit_button("🚀 Iniciar Entrenamiento", type="primary", use_container_width=True)
    
    # Ejecutar entrenamiento
    if submitted:
        with st.spinner("🔄 Preparando entrenamiento..."):
            try:
                # Filtrar vocabulario por longitud máxima
                filtered_vocab = [word for word in vocab_words if len(word) <= max_word_length]
                
                if not filtered_vocab:
                    st.error(f"❌ No hay palabras con longitud ≤ {max_word_length}")
                    return
                
                st.info(f"📚 Vocabulario filtrado: {len(filtered_vocab)} palabras (de {len(vocab_words)})")
                
                # Importar módulos de entrenamiento REAL
                try:
                    from training.speller_module import TinySpellerLightning, build_multimodal_dataloaders
                    import pytorch_lightning as pl
                    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
                    import torch
                    import os
                    st.success("✅ Módulos de PyTorch Lightning importados correctamente")
                except ImportError as e:
                    st.error(f"❌ Error al importar módulos: {e}")
                    st.info("💡 Instalar dependencias: `pip install pytorch-lightning`")
                    return
                
                # Obtener número de letras únicas para num_classes
                unique_letters = sorted(set("".join(filtered_vocab)))
                num_classes = len(unique_letters)
                vocab_size = len(filtered_vocab)
                
                st.info(f"🔧 Config: Vocab={vocab_size}, Classes={num_classes}, Letters: {', '.join(unique_letters[:10])}...")
                
                # Crear dataloaders multimodales REALES
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("📊 Creando dataloaders multimodales...")
                progress_bar.progress(0.2)
                
                try:
                    train_loader, val_loader, test_loader = build_multimodal_dataloaders(
                        words=filtered_vocab,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        max_word_length=max_word_length
                    )
                    st.success(f"✅ Dataloaders: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)} batches")
                    progress_bar.progress(0.4)
                except Exception as e:
                    st.error(f"❌ Error creando dataloaders: {e}")
                    st.info("Verifica que los datasets estén disponibles en Visual Dataset Manager")
                    return
                
                # Inicializar modelo TinySpeller Lightning
                status_text.text("🤖 Inicializando TinySpellerLightning...")
                progress_bar.progress(0.5)
                
                model = TinySpellerLightning(
                    vocab_size=vocab_size,
                    num_classes=num_classes,
                    hidden_dim=hidden_dim,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    freeze_recognizer=freeze_recognizer,
                    label_smoothing=label_smoothing
                )
                
                param_count = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                st.success(f"✅ Modelo: {param_count:,} parámetros ({trainable_params:,} entrenables)")
                progress_bar.progress(0.6)
                
                # Configurar callbacks de PyTorch Lightning
                os.makedirs('checkpoints/speller', exist_ok=True)
                callbacks = [
                    EarlyStopping(
                        monitor='val_loss',
                        patience=patience,
                        verbose=True,
                        mode='min'
                    ),
                    ModelCheckpoint(
                        dirpath='checkpoints/speller',
                        filename='tiny_speller_{epoch:02d}_{val_acc:.2f}',
                        monitor='val_acc',
                        mode='max',
                        save_top_k=2,
                        save_last=True
                    )
                ]
                
                # Configurar PyTorch Lightning Trainer
                status_text.text("⚡ Configurando Trainer...")
                progress_bar.progress(0.7)
                
                trainer = pl.Trainer(
                    max_epochs=max_epochs,
                    accelerator=accelerator,
                    callbacks=callbacks,
                    gradient_clip_val=gradient_clip_val if gradient_clip_val > 0 else None,
                    enable_progress_bar=False,  # Usamos Streamlit progress
                    enable_model_summary=True,
                    deterministic=False,
                    logger=False  # Sin logging para Streamlit
                )
                
                st.success("✅ Trainer configurado con early stopping y checkpoints")
                progress_bar.progress(0.8)
                
                # EJECUTAR ENTRENAMIENTO REAL
                status_text.text("🚀 Iniciando entrenamiento PyTorch Lightning...")
                progress_bar.progress(0.9)
                
                with st.spinner("🔥 Entrenando TinySpeller..."):
                    # Ejecutar entrenamiento real
                    trainer.fit(model, train_loader, val_loader)
                
                progress_bar.progress(1.0)
                status_text.text("✅ Entrenamiento completado")
                
                # Mostrar resultados simulados
                st.success("🎉 ¡Entrenamiento completado exitosamente!")
                
                # Obtener métricas finales REALES
                best_val_loss = trainer.callback_metrics.get('val_loss', 0.0)
                best_val_acc = trainer.callback_metrics.get('val_acc', 0.0)
                
                # Evaluar en test set
                test_results = None
                if test_loader and len(test_loader) > 0:
                    status_text.text("📊 Evaluando en test set...")
                    test_results = trainer.test(model, test_loader, verbose=False)
                    test_acc = test_results[0].get('test_acc', 0.0) if test_results else 0.0
                else:
                    test_acc = 0.0
                
                result_cols = st.columns(3)
                result_cols[0].metric("Val Loss", f"{float(best_val_loss):.3f}")
                result_cols[1].metric("Val Accuracy", f"{float(best_val_acc)*100:.1f}%")
                result_cols[2].metric("Test Accuracy", f"{float(test_acc)*100:.1f}%" if test_acc > 0 else "N/A")
                
                # Información del checkpoint
                if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
                    st.info(f"💾 **Mejor modelo guardado:** `{trainer.checkpoint_callback.best_model_path}`")
                

                st.info("""
                � **Nota**: Este es un entrenamiento simulado para demostrar la interfaz.
                
                **Para implementación real:**
                1. Usar `training.speller_module.build_multimodal_dataloaders()`
                2. Crear `TinySpellerLightning` con configuración especificada
                3. Ejecutar entrenamiento real con PyTorch Lightning
                4. Guardar checkpoints y métricas reales
                """)
                
                # Almacenar resultados REALES en session state
                st.session_state['speller_training_result'] = {
                    'vocab': filtered_vocab,
                    'config': {
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'hidden_dim': hidden_dim,
                        'max_epochs': max_epochs,
                        'num_classes': num_classes,
                        'vocab_size': vocab_size,
                        'freeze_recognizer': freeze_recognizer
                    },
                    'final_metrics': {
                        'val_loss': float(best_val_loss),
                        'val_accuracy': float(best_val_acc) * 100,
                        'test_accuracy': float(test_acc) * 100 if test_acc > 0 else None,
                        'epochs_trained': trainer.current_epoch + 1,
                        'checkpoint_path': trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else None
                    },
                    'training_type': 'real_pytorch_lightning'
                }
                
            except Exception as e:
                st.error(f"❌ Error al preparar entrenamiento: {str(e)}")
                st.exception(e)
    
    # Mostrar resultados previos si existen
    if 'speller_training_result' in st.session_state:
        st.markdown("#### 📊 Último Entrenamiento")
        result = st.session_state['speller_training_result']
        config = result['config']
        metrics = result['final_metrics']
        
        result_cols = st.columns(4)
        result_cols[0].metric("Vocabulario", config['vocab_size'])
        
        # Manejar tanto resultados simulados como reales
        training_type = result.get('training_type', 'simulated')
        if training_type == 'real_pytorch_lightning':
            # Resultados reales
            result_cols[1].metric("Val Accuracy", f"{metrics.get('val_accuracy', 0):.1f}%")
            result_cols[2].metric("Test Accuracy", f"{metrics.get('test_accuracy', 0):.1f}%" if metrics.get('test_accuracy') else "N/A")
            result_cols[3].metric("Val Loss", f"{metrics.get('val_loss', 0):.3f}")
            
            # Mostrar información adicional
            if metrics.get('checkpoint_path'):
                st.info(f"💾 **Checkpoint guardado:** `{metrics['checkpoint_path']}`")
            st.caption(f"🏃 Entrenamiento PyTorch Lightning - Épocas: {metrics.get('epochs_trained', 'N/A')}")
        else:
            # Resultados simulados (compatibilidad hacia atrás)
            result_cols[1].metric("Accuracy", f"{metrics.get('accuracy', 0):.1f}%")
            result_cols[2].metric("Top-3 Acc", f"{metrics.get('top3_accuracy', 0):.1f}%")
            result_cols[3].metric("Loss", f"{metrics.get('loss', 0):.3f}")
            st.caption("📝 Resultado de entrenamiento simulado")
        
        if st.button("🗑️ Limpiar Resultados"):
            del st.session_state['speller_training_result']
            st.rerun()


if __name__ == "__main__":
    main()