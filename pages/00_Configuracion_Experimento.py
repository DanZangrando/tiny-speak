import streamlit as st
import os
import json
import shutil
import time
import random
from pathlib import Path
from datetime import datetime

# Configurar página
st.set_page_config(
    page_title="Configuración del Experimento",
    page_icon="⚙️",
    layout="wide"
)

# Importaciones del proyecto
from components.modern_sidebar import display_modern_sidebar
from training.config import load_master_dataset_config, save_master_dataset_config
from utils.audio import generar_variaciones_completas
from utils.visual import generate_letter_image, save_image_to_file, SYSTEM_FONTS
from utils.vocabulary import get_diccionario_predefinido, get_nombres_diccionarios
from utils.graphemes import get_phoneme_inventory, get_language_letters

FIXED_LANGUAGES = ['es', 'en', 'fr']
LANG_LABELS = {'es': '🇪🇸 Español', 'en': '🇺🇸 Inglés', 'fr': '🇫🇷 Francés'}

def cleanup_datasets(mode="all"):
    """Borra físicamente las carpetas de datos seleccionadas."""
    root = Path(__file__).parent.parent
    paths = []
    if mode in ["all", "audio"]:
        paths.append(root / "data" / "audios")
    if mode in ["all", "visual"]:
        paths.append(root / "data" / "visual")
        
    for p in paths:
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

def get_dictionary_status(dict_name):
    """Obtiene el estado de un diccionario en los 3 idiomas."""
    status = {}
    for lang in FIXED_LANGUAGES:
        data = get_diccionario_predefinido(dict_name, idioma=lang)
        status[lang] = len(data['palabras']) if data else 0
    return status

def generate_audio_logic(config, selected_dict, num_variations, rangos_audio, progress_text, bar, start_prog, end_prog):
    """Lógica de generación de audio separada en palabras y fonemas por idioma."""
    all_word_samples = {}
    all_phoneme_samples = {}
    
    # 1. Preparar lista de grafemas totales para el entrenamiento visual posterior
    all_unique_graphemes = set()
    for lang in FIXED_LANGUAGES:
        # Añadir letras básicas y dígrafos del idioma
        all_unique_graphemes.update(get_language_letters(lang))
    
    # 2. Generar para cada idioma
    total_langs = len(FIXED_LANGUAGES)
    for idx, lang in enumerate(FIXED_LANGUAGES):
        progress_text.text(f"Generando Audio ({LANG_LABELS[lang]})...")
        lang_words = {}
        lang_phonemes = {}
        
        # Obtener palabras del diccionario seleccionado
        dic_data = get_diccionario_predefinido(selected_dict, idioma=lang)
        words = dic_data['palabras'] if dic_data else []
        
        # Obtener inventario fonético oficial del idioma
        phonemes = get_phoneme_inventory(lang)
        
        total_items = len(words) + len(phonemes)
        current_idx = 0
        
        # Generar Palabras
        for word in words:
            samples = generar_variaciones_completas(word, lang, num_variations, rangos=rangos_audio)
            lang_words[word] = samples
            current_idx += 1
            prog = start_prog + (idx + (current_idx/total_items)) * (end_prog - start_prog) / total_langs
            bar.progress(int(prog))
            
        # Generar Fonemas (Aislados por idioma)
        for phoneme in phonemes:
            samples = generar_variaciones_completas(phoneme, lang, num_variations, rangos=rangos_audio)
            lang_phonemes[phoneme] = samples
            current_idx += 1
            prog = start_prog + (idx + (current_idx/total_items)) * (end_prog - start_prog) / total_langs
            bar.progress(int(prog))
            
        all_word_samples[lang] = lang_words
        all_phoneme_samples[lang] = lang_phonemes
        
    return all_word_samples, all_phoneme_samples, all_unique_graphemes

def generate_visual_logic(all_unique_graphemes, vis_variations, selected_fonts, font_size_range, rotation_range, noise_range, progress_text, bar, start_prog, end_prog):
    """Lógica de generación visual para el set de grafemas total."""
    visual_samples = {}
    total_chars = len(all_unique_graphemes)
    for idx, char in enumerate(sorted(list(all_unique_graphemes))):
        progress_text.text(f"Generando Visual: '{char.upper()}' ({idx+1}/{total_chars})...")
        char_variations = []
        for v in range(vis_variations):
            f_name = random.choice(selected_fonts) if selected_fonts else "DejaVu Sans"
            f_size = random.randint(font_size_range[0], font_size_range[1])
            rot = random.uniform(-rotation_range, rotation_range)
            noi = random.uniform(0, noise_range)
            
            img = generate_letter_image(char, font_size=f_size, rotation=rot, noise_level=noi, font_name=f_name)
            if img:
                meta = save_image_to_file(img, char, {
                    "font": f_name, "font_size": f_size, "rotation": rot, "noise_level": noi
                })
                if meta: char_variations.append(meta)
        visual_samples[char] = char_variations
        prog = start_prog + (idx + 1) * (end_prog - start_prog) / total_chars
        bar.progress(int(prog))
        
    return visual_samples

def main():
    display_modern_sidebar("config")
    
    st.title("⚙️ Configuración Global del Experimento")
    st.markdown("---")
    
    # Cargar configuración actual
    config = load_master_dataset_config()
    exp_config = config.get("experiment_config", {})
    
    # === SECCION 0: FUNDAMENTOS ===
    with st.expander("📚 Fundamentos de la Generación de Datos", expanded=False):
        st.markdown("""
        ### ¿Cómo se construye el cerebro auditivo y visual?
        
        #### 🎤 Generación de Audio (Ruta Auditiva)
        Utilizamos el motor **gTTS (Google Text-to-Speech)** como base para la síntesis de voz. 
        - **Palabras y Fonemas:** Generamos sonidos tanto para palabras completas como para **grafemas individuales**. Esto es crucial para que el modelo aprenda la correspondencia grafema-fonema.
        - **Aumentación:** Para simular la variabilidad biológica del oído (tono-topía), aplicamos factores combinados de Pitch, Velocidad y Volumen.
        
        #### 🖼️ Generación Visual (Ruta de Graphemas)
        Las imágenes de caracteres se generan procedimentalmente para simular el área **VWFA**.
        - **Invariancia:** Para reconocer la \"letra\" y no el \"píxel\", aplicamos Ruido, Rotación y Diversidad de Fuentes tipográficas.
        """)

    # === SECCION 1: DASHBOARD DE ESTADO ===
    st.header("📊 Estado Actual de los Datos")
    
    cols = st.columns(3)
    for i, lang in enumerate(FIXED_LANGUAGES):
        with cols[i]:
            st.subheader(LANG_LABELS[lang])
            
            # Words metrics
            words_data = config.get("generated_samples", {}).get(lang, {})
            st.metric("Palabras (Audio)", len(words_data))
            
            # Phonemes metrics
            phonemes_data = config.get("phoneme_samples", {}).get(lang, {})
            st.metric("Fonemas (Audio)", len(phonemes_data))
            
            # Visual metrics
            lang_letters = get_language_letters(lang)
            visual_data = config.get("visual_dataset", {}).get("generated_images", {})
            present_letters = [l for l in lang_letters if l in visual_data]
            st.metric("Letras (Visual)", f"{len(present_letters)}/{len(lang_letters)}")

    st.markdown("---")
    
    # === SECCION 2: PARAMETROS ===
    st.header("🎛️ Parámetros de Generación")
    
    # -- Vocabulario --
    with st.expander("📝 Selección de Vocabulario y Diccionarios", expanded=True):
        dict_names = get_nombres_diccionarios()
        dict_options = []
        for d in dict_names:
            status = get_dictionary_status(d)
            counts = "/".join([str(status[l]) for l in FIXED_LANGUAGES])
            dict_options.append(f"{d} ({counts} palabras ES/EN/FR)")
            
        current_dict_idx = 0
        current_dict_base = exp_config.get('base_dictionary', 'animales')
        for i, d in enumerate(dict_names):
            if d == current_dict_base:
                current_dict_idx = i
                break
                
        selected_option = st.selectbox("Selecciona un Diccionario", dict_options, index=current_dict_idx)
        selected_dict = dict_names[dict_options.index(selected_option)]
        
        status = get_dictionary_status(selected_dict)
        if len(set(status.values())) > 1:
            st.warning(f"⚠️ El diccionario seleccionado no tiene la misma cantidad de palabras en todos los idiomas: {status}")
        else:
            st.success(f"✅ Diccionario balanceado: {status['es']} palabras por idioma.")

    # -- Aumentación Audio --
    with st.expander("🔊 Aumentación de Audio (Parámetros Detallados)"):
        col1, col2 = st.columns(2)
        audio_conf_root = config.get("configuracion_audio", {})
        audio_conf = audio_conf_root.get("rangos", {})
        with col1:
            num_variations = st.slider("Variaciones por palabra", 1, 15, value=audio_conf_root.get("num_variaciones", 5))
            pitch_range = st.slider("Rango de Pitch (Tono)", 0.5, 2.0, value=(audio_conf.get("pitch", [0.8, 1.3])[0], audio_conf.get("pitch", [0.8, 1.3])[1]))
        with col2:
            speed_range = st.slider("Rango de Velocidad", 0.5, 2.0, value=(audio_conf.get("speed", [0.7, 1.4])[0], audio_conf.get("speed", [0.7, 1.4])[1]))
            volume_range = st.slider("Rango de Volumen", 0.5, 1.5, value=(audio_conf.get("volume", [0.8, 1.2])[0], audio_conf.get("volume", [0.8, 1.2])[1]))

    # -- Aumentación Visual --
    with st.expander("🎨 Aumentación Visual (Fuentes e Imagen)"):
        v_conf = config.get("visual_dataset", {})
        v_params = v_conf.get("image_params", {})
        c1, c2 = st.columns(2)
        with c1:
            selected_fonts = st.multiselect("Fuentes Tipográficas", options=list(SYSTEM_FONTS.keys()), default=v_params.get("fonts", ["DejaVu Sans"]))
            vis_variations = st.slider("Variaciones por letra", 1, 50, value=v_params.get("variations_per_letter", 10))
            font_size_range = st.slider("Rango Tamaño Fuente", 10, 60, value=(v_params.get("font_size_min", 24), v_params.get("font_size_max", 40)))
        with c2:
            rotation_range = st.slider("Rango Rotación (°)", 0, 90, value=v_params.get("rotation_range", 15))
            noise_range = st.slider("Nivel Máximo de Ruido", 0.0, 1.0, value=v_params.get("noise_level", 0.1))

    # -- Configuración Global de Entrenamiento --
    with st.expander("⚙️ Configuración de Entrenamiento Global (Split de Datos)"):
        st.info("💡 Estos valores definen cómo se dividen tus datos para el entrenamiento y la evaluación de TODOS los modelos.")
        
        ratios = exp_config.get("split_ratios", {"train": 0.7, "val": 0.15, "test": 0.15})
        
        c1, c2, c3 = st.columns(3)
        with c1:
            train_perc = st.slider("Entrenamiento (%)", 10, 90, int(ratios["train"] * 100))
        with c2:
            val_perc = st.slider("Validación (%)", 5, 45, int(ratios["val"] * 100))
        with c3:
            test_perc = st.slider("Test (%)", 5, 45, int(ratios.get("test", 0.15) * 100))
            
        total_perc = train_perc + val_perc + test_perc
        if total_perc != 100:
            st.error(f"⚠️ El total debe sumar 100%. Actual: {total_perc}%")
        else:
            st.success("✅ División de datos balanceada.")
            new_ratios = {
                "train": train_perc / 100.0,
                "val": val_perc / 100.0,
                "test": test_perc / 100.0
            }

    st.markdown("---")
    
    # === SECCION 3: ACCION ===
    st.header("🚀 Ejecución")
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    regenerate_mode = None
    if col_btn1.button("🔥 TODO", type="primary", width='stretch', help="Regenera audio y visual"):
        regenerate_mode = "all"
    if col_btn2.button("🎤 SÓLO AUDIO", width='stretch', help="Regenera sólo archivos de audio"):
        regenerate_mode = "audio"
    if col_btn3.button("🖼️ SÓLO VISUAL", width='stretch', help="Regenera sólo imágenes"):
        regenerate_mode = "visual"

    if regenerate_mode:
        progress_text = st.empty()
        bar = st.progress(0)
        
        # 1. Cleanup
        progress_text.text(f"Limpiando directorios ({regenerate_mode})...")
        cleanup_datasets(regenerate_mode)
        bar.progress(5)
        
        # 2. Configuración temporal
        rangos_audio = {"pitch": list(pitch_range), "speed": list(speed_range), "volume": list(volume_range)}
        
        config["experiment_config"]["base_dictionary"] = selected_dict
        config["experiment_config"]["last_run"] = datetime.now().isoformat()
        config["experiment_config"]["split_ratios"] = new_ratios
        
        config["configuracion_audio"]["num_variaciones"] = num_variations
        config["configuracion_audio"]["rangos"] = rangos_audio
        
        config["visual_dataset"]["image_params"] = {
            "variations_per_letter": vis_variations,
            "fonts": selected_fonts,
            "font_size_min": font_size_range[0],
            "font_size_max": font_size_range[1],
            "rotation_range": rotation_range,
            "noise_level": noise_range
        }

        # 3. Generar Audio
        target_chars = set()
        if regenerate_mode in ["all", "audio"]:
            words_s, phones_s, chars = generate_audio_logic(config, selected_dict, num_variations, rangos_audio, progress_text, bar, 5, 50 if regenerate_mode == "all" else 100)
            config["generated_samples"] = words_s
            config["phoneme_samples"] = phones_s
            target_chars = chars
        else:
            # Si no regeneramos audio, reconstruir el set visual a partir de los inventarios oficiales
            for lang in FIXED_LANGUAGES:
                target_chars.update(get_language_letters(lang))
                target_chars.update(get_phoneme_inventory(lang))

        # 4. Generar Visual
        if regenerate_mode in ["all", "visual"]:
            start_p = 50 if regenerate_mode == "all" else 5
            vis_samples = generate_visual_logic(target_chars, vis_variations, selected_fonts, font_size_range, rotation_range, noise_range, progress_text, bar, start_p, 100)
            config["visual_dataset"]["generated_images"] = vis_samples
            
        # 5. Finalizar
        save_master_dataset_config(config)
        bar.progress(100)
        progress_text.success(f"✅ Regeneración ({regenerate_mode}) exitosa.")
        time.sleep(2)
        st.rerun()

if __name__ == "__main__":
    main()
