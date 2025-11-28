import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from datetime import datetime

import torch
import torch.nn.functional as F

from components.modern_sidebar import display_modern_sidebar
from training.config import load_master_dataset_config
from training.runner import train_listener, train_recognizer, train_reader
from training.reader_module import TinyReaderLightning
from training.audio_module import PhonologicalPathwayLightning
from training.visual_module import VisualPathwayLightning
from utils import list_checkpoints, get_default_words, encontrar_device

st.set_page_config(
    page_title="Experimento de Transparencia",
    page_icon="🔬",
    layout="wide"
)

def get_custom_css():
    return """
    <style>
    .main-header {
        background: linear-gradient(90deg, #6a11cb, #2575fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .card {
        background-color: var(--secondary-background-color);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #6a11cb;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """
import numpy as np
import scipy.stats as stats

def perform_statistical_analysis(df, languages=None):
    """Realiza análisis estadístico automático sobre los resultados."""
    st.markdown("### 📊 Reporte Estadístico Automático")
    
    # Filtrar métricas finales (última época)
    final_metrics = df.groupby(['language', 'model']).last().reset_index()
    
    models = final_metrics['model'].unique()
    
    for m in models:
        st.markdown(f"#### Análisis para: {m.capitalize()}")
        subset = final_metrics[final_metrics['model'] == m]
        
        # Preparar datos para ANOVA
        groups = []
        lang_labels = []
        for lang in subset['language'].unique():
            # Nota: Aquí idealmente tendríamos múltiples runs por idioma para un ANOVA real.
            # Como este script corre 1 run por idioma, el ANOVA no es aplicable directamente sobre 'runs'.
            # PERO, podemos usar las métricas de las últimas N épocas como "muestras" de la convergencia.
            
            # Tomamos las últimas 5 épocas de cada idioma para comparar estabilidad/convergencia
            lang_data = df[(df['model'] == m) & (df['language'] == lang)].tail(5)['val_loss'].values
            groups.append(lang_data)
            lang_labels.append(lang)
            
        if len(groups) < 2:
            st.info("No hay suficientes idiomas para comparar.")
            continue
            
        # One-Way ANOVA
        f_stat, p_value = stats.f_oneway(*groups)
        
        st.write(f"**ANOVA (val_loss últimas 5 épocas):** F={f_stat:.4f}, p={p_value:.4e}")
        
        if p_value < 0.05:
            st.success(f"✅ Hay diferencias significativas entre idiomas (p < 0.05).")
            
            # Pairwise comparisons (T-tests)
            st.markdown("**Comparaciones Par a Par (T-test):**")
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    l1, d1 = lang_labels[i], groups[i]
                    l2, d2 = lang_labels[j], groups[j]
                    
                    t_stat, p_val_t = stats.ttest_ind(d1, d2)
                    sig = "✅ Significativo" if p_val_t < 0.05 else "❌ No significativo"
                    
                    mean_diff = d1.mean() - d2.mean()
                    better = l1 if mean_diff < 0 else l2 # Menor loss es mejor
                    
                    st.write(f"- **{l1} vs {l2}**: p={p_val_t:.4f} ({sig}). Mejor: **{better}**")
        else:
            st.warning("❌ No se encontraron diferencias significativas entre idiomas.")

def run_experiment(exp_name, languages, l_config, r_config, reader_config):
    """Ejecuta el experimento completo."""
    
    # Crear directorio del experimento
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    exp_dir = Path(f"experiments/{timestamp}_{exp_name}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Obtener configuración del dataset actual para guardar referencia
    master_config = load_master_dataset_config()
    base_dict = master_config.get('diccionario_seleccionado', 'tiny_kalulu_original')
    # Si es un objeto (dict), intentamos sacar el nombre, si es string lo usamos directo
    if isinstance(base_dict, dict):
        base_dict_name = base_dict.get('nombre', 'tiny_kalulu_original')
    else:
        base_dict_name = str(base_dict)

    # Guardar configuración
    full_config = {
        "name": exp_name,
        "languages": languages,
        "base_dictionary": base_dict_name, # Guardamos el diccionario usado
        "listener": l_config,
        "recognizer": r_config,
        "reader": reader_config,
        "timestamp": timestamp
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(full_config, f, indent=2)
        
    results = []
    status_text = st.empty()
    progress_bar = st.progress(0)
    total_steps = len(languages) * 3 # Listener, Recognizer, Reader per language
    current_step = 0
    
    # Diccionario para guardar paths de checkpoints
    ckpt_paths = {}
    
    try:
        # Definir callback de progreso
        def update_progress(current, total, msg=""):
            pass # El progreso se maneja externamente con progress_bar
            
        # Crear contenedores para gráficas en tiempo real
        st.markdown("#### 📈 Progreso en Tiempo Real")
        chart_col1, chart_col2 = st.columns(2)
        loss_chart = chart_col1.empty()
        acc_chart = chart_col2.empty()
        placeholders = (loss_chart, acc_chart)
        
        # Placeholder para predicciones animadas (TinyReader)
        st.markdown("#### 🔮 Predicciones en Vivo")
        prediction_placeholder = st.empty()

        for lang in languages:
            # 1. Train Listener
            status_text.markdown(f"### 🎧 Entrenando TinyListener ({lang})...")
            l_ckpt, l_hist = train_listener(lang, l_config, progress_callback=update_progress, plot_placeholders=placeholders)
            ckpt_paths[f"listener_{lang}"] = l_ckpt
            for h in l_hist: h.update({"language": lang, "model": "listener", "exp_name": exp_name})
            results.extend(l_hist)
            current_step += 1
            progress_bar.progress(current_step / total_steps)
            
            # 2. Train Recognizer
            status_text.markdown(f"### 👁️ Entrenando TinyRecognizer ({lang})...")
            r_ckpt, r_hist = train_recognizer(lang, r_config, progress_callback=update_progress, plot_placeholders=placeholders)
            ckpt_paths[f"recognizer_{lang}"] = r_ckpt
            for h in r_hist: h.update({"language": lang, "model": "recognizer", "exp_name": exp_name})
            results.extend(r_hist)
            current_step += 1
            progress_bar.progress(current_step / total_steps)
            
            # 3. Train Reader
            status_text.markdown(f"### 🧠 Entrenando TinyReader ({lang})...")
            reader_ckpt, reader_hist = train_reader(
                lang, l_ckpt, r_ckpt, reader_config, 
                progress_callback=update_progress, 
                plot_placeholders=placeholders,
                prediction_placeholder=prediction_placeholder
            )
            ckpt_paths[f"reader_{lang}"] = reader_ckpt
            for h in reader_hist: h.update({"language": lang, "model": "reader", "exp_name": exp_name})
            results.extend(reader_hist)
            current_step += 1
            progress_bar.progress(current_step / total_steps)
            
        # Guardar resultados
        df = pd.DataFrame(results)
        df.to_csv(exp_dir / "results.csv", index=False)
        
        # Guardar paths de checkpoints
        with open(exp_dir / "checkpoints.json", "w") as f:
            json.dump(ckpt_paths, f, indent=2)
        
        status_text.success("✅ Experimento completado exitosamente!")
        st.balloons()
        return df, exp_dir
        
    except Exception as e:
        status_text.error(f"❌ Error en el experimento: {e}")
        st.exception(e)
        return None, None

def run_comparative_laboratory(exp_path):
    exp_dir = Path(exp_path)
    ckpt_path = exp_dir / "checkpoints.json"
    
    if not ckpt_path.exists():
        st.error("No se encontró el archivo checkpoints.json en el experimento.")
        return
        
    with open(ckpt_path, "r") as f:
        checkpoints = json.load(f)
        
    # Identificar idiomas
    langs = set([k.split("_")[1] for k in checkpoints.keys() if "_" in k])
    
    # Seleccionar palabra
    # Usamos palabras por defecto o cargamos de un modelo
    # Cargar diccionarios para traducción
    # Cargar configuración del experimento para saber qué diccionario usar
    from diccionarios import get_diccionario_predefinido
    config_path = exp_dir / "config.json"
    base_dict_name = "tiny_kalulu_original" # Default
    
    if config_path.exists():
        with open(config_path) as f:
            exp_config = json.load(f)
            # Intentar obtener el nombre del diccionario base
            # Puede estar en 'experiment_config' -> 'base_dictionary' o directamente si es legacy
            if 'experiment_config' in exp_config:
                base_dict_name = exp_config['experiment_config'].get('base_dictionary', base_dict_name)
            elif 'base_dictionary' in exp_config:
                base_dict_name = exp_config['base_dictionary']
            
            # También podría estar en la config del listener/reader si se guardó allí
            elif 'listener' in exp_config and 'vocab_name' in exp_config['listener']:
                 base_dict_name = exp_config['listener']['vocab_name']

    # Si no se encontró en config (o es el default), intentar inferir del checkpoint
    # Esto es necesario para experimentos antiguos que no guardaron el nombre del diccionario
    if base_dict_name == "tiny_kalulu_original":
        try:
            # Buscar checkpoint de listener_es (que suele tener el vocabulario base)
            ckpt_es = checkpoints.get("listener_es")
            if ckpt_es:
                meta_path = Path(ckpt_es).with_suffix(".ckpt.meta.json")
                if meta_path.exists():
                    with open(meta_path) as f:
                        vocab_ckpt = set(json.load(f).get('config', {}).get('vocab', []))
                    
                    if vocab_ckpt:
                        # Comparar con diccionarios disponibles en data/diccionarios
                        dict_dir = Path("data/diccionarios")
                        best_match = None
                        max_overlap = 0
                        
                        for dict_file in dict_dir.glob("*.txt"):
                            # Ignorar archivos de idioma específico (_en, _fr) para encontrar el base
                            if "_" in dict_file.stem and dict_file.stem.split("_")[-1] in ["en", "fr", "de"]:
                                continue
                                
                            dict_name = dict_file.stem
                            # Cargar palabras del diccionario candidato
                            d_temp = get_diccionario_predefinido(dict_name, "es")
                            if d_temp:
                                vocab_dict = set(d_temp['palabras'])
                                # Calcular overlap
                                overlap = len(vocab_ckpt.intersection(vocab_dict))
                                # Buscamos el que tenga mayor coincidencia y cubra el vocabulario del checkpoint
                                if overlap > max_overlap: 
                                    max_overlap = overlap
                                    best_match = dict_name
                        
                        if best_match and max_overlap > 0:
                            base_dict_name = best_match
                            # st.info(f"🔍 Diccionario detectado automáticamente: **{base_dict_name}**")
        except Exception as e:
            # Si falla la inferencia, seguimos con el default
            pass

    # Cargar diccionarios base usando el nombre correcto
    d_es = get_diccionario_predefinido(base_dict_name, "es")
    d_en = get_diccionario_predefinido(base_dict_name, "en")
    d_fr = get_diccionario_predefinido(base_dict_name, "fr")
    
    # Crear mapeo: índice -> {es: palabra, en: palabra, fr: palabra}
    translation_map = {}
    display_options = []
    
    # Verificar qué diccionarios se cargaron realmente
    loaded_dicts = {}
    if d_es: loaded_dicts['es'] = d_es['palabras']
    if d_en: loaded_dicts['en'] = d_en['palabras']
    if d_fr: loaded_dicts['fr'] = d_fr['palabras']
    
    if loaded_dicts:
        # Asumimos que todos tienen la misma longitud y orden (son paralelos)
        # Usamos el primer idioma disponible como referencia para la longitud
        ref_lang = list(loaded_dicts.keys())[0]
        vocab_len = len(loaded_dicts[ref_lang])
        
        for i in range(vocab_len):
            # Construir entrada del mapa
            entry = {}
            for lang, words in loaded_dicts.items():
                if i < len(words):
                    entry[lang] = words[i]
                else:
                    entry[lang] = "???"
            
            # Usar español como clave si existe, sino el primer idioma
            key_word = entry.get('es', entry[ref_lang])
            translation_map[key_word] = entry
            display_options.append(key_word)
    else:
        # Fallback si no hay diccionarios cargados
        from utils import get_default_words
        display_options = get_default_words()
        translation_map = {w: {'es': w, 'en': w, 'fr': w} for w in display_options}

    target_word_key = st.selectbox("Palabra a Imaginar (Concepto)", display_options)
    
    if st.button("🧠 Imaginar en todos los idiomas", type="primary"):
        device = encontrar_device()
        
        cols = st.columns(len(langs))
        for i, lang in enumerate(sorted(list(langs))):
            with cols[i]:
                st.markdown(f"### {lang.upper()}")
                
                # Obtener palabra traducida
                target_word = translation_map.get(target_word_key, {}).get(lang, target_word_key)
                st.info(f"Imaginando: **{target_word}**")
                
                try:
                    # Cargar Modelos
                    reader_path = checkpoints.get(f"reader_{lang}")
                    listener_path = checkpoints.get(f"listener_{lang}")
                    rec_path = checkpoints.get(f"recognizer_{lang}")
                    
                    if not reader_path or not listener_path or not rec_path:
                        st.error(f"Faltan checkpoints para {lang}.")
                        continue

                    # Cargar Reader (que contiene listener y recognizer internamente)
                    # Nota: Necesitamos el vocabulario correcto.
                    # Intentamos cargar metadata del reader
                    vocab = []
                    meta_path = Path(reader_path).with_suffix('.ckpt.meta.json')
                    if meta_path.exists():
                        with open(meta_path) as f:
                            meta_data = json.load(f)
                            vocab = meta_data.get('config', {}).get('vocab', [])
                            if not vocab: vocab = meta_data.get('vocab', []) # Fallback for older formats
                    
                    if not vocab:
                        # Fallback al diccionario del idioma
                        if lang == 'es': vocab = d_es['palabras'] if d_es else []
                        elif lang == 'en': vocab = d_en['palabras'] if d_en else []
                        elif lang == 'fr': vocab = d_fr['palabras'] if d_fr else []
                        
                    if not vocab:
                        st.error(f"No se pudo cargar el vocabulario para {lang}.")
                        continue

                    reader_module = TinyReaderLightning.load_from_checkpoint(
                        reader_path,
                        class_names=vocab,
                        listener_checkpoint_path=listener_path,
                        recognizer_checkpoint_path=rec_path,
                        map_location=device
                    )
                    reader_module.to(device)
                    reader_module.eval()
                    
                    # Inferencia
                    # 1. Get Images
                    images = reader_module._get_word_images(target_word).to(device)
                    
                    # Mostrar secuencia de imágenes
                    # images shape: [seq_len, 3, 64, 64]
                    st.write("Secuencia Imaginada:")
                    img_cols = st.columns(len(target_word))
                    for idx, char in enumerate(target_word):
                        if idx < len(images):
                            img_np = images[idx].cpu().permute(1, 2, 0).numpy()
                            # Normalizar si es necesario (asumiendo que están en [0, 1] o estandarizadas)
                            # Si están estandarizadas, des-estandarizar visualmente podría ser necesario, 
                            # pero por ahora mostramos raw.
                            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
                            
                            with img_cols[idx]:
                                st.image(img_np, caption=char.upper(), width=60)
                    
                    # 2. Recognizer
                    with torch.no_grad():
                        res = reader_module.recognizer(images)
                        word_logits = res[0] if isinstance(res, tuple) else res
                        batch_logits = word_logits.unsqueeze(0)
                        
                        # 3. Reader
                        generated_embeddings = reader_module.reader(batch_logits, target_length=100)
                        
                        # 4. Listener
                        lengths = torch.tensor([generated_embeddings.shape[1]], device=device) # Use actual sequence length
                        packed_gen = torch.nn.utils.rnn.pack_padded_sequence(
                            generated_embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
                        )
                        # PhonologicalPathway usa Mean Pooling + Classifier
                        pooled_gen = generated_embeddings.mean(dim=1)
                        listener_logits = reader_module.listener.classifier(pooled_gen)
                        probs = torch.softmax(listener_logits, dim=1)
                        pred_idx = torch.argmax(probs, dim=1).item()
                        pred_word = vocab[pred_idx]
                        confidence = probs[0, pred_idx].item()
                        
                    # Mostrar Resultados
                    st.metric("Interpretación", pred_word, delta=f"{confidence:.2%}")
                    if pred_word == target_word:
                        st.success("Correcto")
                    else:
                        st.error("Incorrecto")
                        
                except Exception as e:
                    st.error(f"Error en {lang}: {e}")

def main():
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    display_modern_sidebar("transparency_experiment")
    
    st.markdown('<h1 class="main-header">🔬 Experimento de Transparencia</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Este módulo permite automatizar el entrenamiento y comparación de modelos (Listener, Recognizer, Reader) 
    a través de múltiples idiomas para probar la **Hipótesis de Transparencia**.
    """)
    
    tabs = st.tabs(["📊 Estado del Dataset", "⚙️ Configuración y Ejecución", "📈 Análisis de Resultados", "🧪 Laboratorio Comparativo"])
    
    # ==========================================
    # TAB 1: ESTADO DEL DATASET
    # ==========================================
    with tabs[0]:
        st.markdown("### Verificación de Datos")
        config = load_master_dataset_config()
        available_langs = config.get('experiment_config', {}).get('languages', ['es', 'en', 'fr'])
        
        cols = st.columns(len(available_langs))
        for i, lang in enumerate(available_langs):
            with cols[i]:
                st.markdown(f"#### {lang.upper()}")
                # Verificar Audio
                # TODO: Implementar conteo real si es necesario, por ahora check básico
                st.info(f"Idioma habilitado en configuración.")
                
    # ==========================================
    # TAB 2: CONFIGURACIÓN Y EJECUCIÓN
    # ==========================================
    with tabs[1]:
        st.markdown("### ⚙️ Configuración del Experimento")
        
        col_name, col_langs = st.columns([1, 2])
        with col_name:
            exp_name = st.text_input("Nombre del Experimento", "transparency_v1")
        with col_langs:
            selected_langs = st.multiselect("Idiomas a Comparar", available_langs, default=available_langs)
            
        with st.expander("🎧 Configuración TinyListener", expanded=False):
            l_epochs = st.number_input("Listener Epochs", 1, 100, 10, key="l_epochs")
            l_lr = st.number_input("Listener LR", 1e-5, 1e-1, 1e-3, format="%.1e", key="l_lr")
            l_patience = st.slider("Listener Patience", 1, 20, 5, key="l_patience")
            l_min_delta = st.slider("Listener Min Delta", 0.0, 0.1, 0.00, step=0.001, format="%.3f", key="l_min_delta")
            
        with st.expander("👁️ Configuración TinyRecognizer", expanded=False):
            r_epochs = st.number_input("Recognizer Epochs", 1, 100, 10, key="r_epochs")
            r_lr = st.number_input("Recognizer LR", 1e-5, 1e-1, 1e-3, format="%.1e", key="r_lr")
            r_patience = st.slider("Recognizer Patience", 1, 20, 5, key="r_patience")
            r_min_delta = st.slider("Recognizer Min Delta", 0.0, 0.1, 0.00, step=0.001, format="%.3f", key="r_min_delta")
            
        with st.expander("🧠 Configuración TinyReader", expanded=True):
            reader_epochs = st.number_input("Reader Epochs", 1, 100, 20, key="reader_epochs")
            reader_lr = st.number_input("Reader LR", 1e-5, 1e-1, 1e-3, format="%.1e", key="reader_lr")
            reader_patience = st.slider("Reader Patience", 1, 20, 10, key="reader_patience")
            reader_min_delta = st.slider("Reader Min Delta", 0.0, 0.1, 0.00, step=0.001, format="%.3f", key="reader_min_delta")
            
            c1, c2, c3 = st.columns(3)
            w_mse = c1.number_input("W MSE", 0.0, 10.0, 1.0)
            w_cos = c2.number_input("W Cosine", 0.0, 10.0, 1.0)
            w_perceptual = c3.number_input("W Perceptual", 0.0, 10.0, 0.1)
            
        if st.button("🚀 Ejecutar Experimento Completo", type="primary"):
            if not selected_langs:
                st.error("Selecciona al menos un idioma.")
            else:
                l_config = {
                    "epochs": l_epochs, "lr": l_lr, "batch_size": 32,
                    "patience": l_patience, "min_delta": l_min_delta
                }
                r_config = {
                    "epochs": r_epochs, "lr": r_lr, "batch_size": 32,
                    "patience": r_patience, "min_delta": r_min_delta
                }
                reader_config = {
                    "epochs": reader_epochs, "lr": reader_lr, "batch_size": 32,
                    "w_mse": w_mse, "w_cos": w_cos, "w_perceptual": w_perceptual,
                    "patience": reader_patience, "min_delta": reader_min_delta
                }
                
                df_results, exp_path = run_experiment(exp_name, selected_langs, l_config, r_config, reader_config)
                if df_results is not None:
                    st.session_state['last_experiment_results'] = df_results
                    st.session_state['last_experiment_path'] = str(exp_path)
                    
    # ==========================================
    # TAB 3: ANÁLISIS DE RESULTADOS
    # ==========================================
    with tabs[2]:
        st.markdown("### 📈 Análisis de Resultados")
        
        # Selector de Experimento
        experiments_dir = Path("experiments")
        if not experiments_dir.exists():
            st.info("No hay experimentos guardados.")
        else:
            # Listar directorios de experimentos
            # Listar directorios de experimentos
            exp_dirs = [d for d in experiments_dir.iterdir() if d.is_dir() and d.name != "models"]
            exp_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            selected_exp_name = st.selectbox(
                "Seleccionar Experimento",
                [d.name for d in exp_dirs],
                key="analysis_exp_selector"
            )
            
            if selected_exp_name:
                exp_path = experiments_dir / selected_exp_name
                
                # Cargar configuración
                config_path = exp_path / "config.json"
                if config_path.exists():
                    with open(config_path) as f:
                        exp_config = json.load(f)
                    
                    st.success(f"Experimento cargado: **{exp_config.get('name', 'Sin nombre')}** ({exp_config.get('timestamp', '')})")
                    
                    # Cargar resultados
                    results_path = exp_path / "results.csv"
                    if results_path.exists():
                        df_results = pd.read_csv(results_path)
                        
                        # --- ANÁLISIS COMPARATIVO GLOBAL ---
                        st.markdown("### 📊 Comparativa Global")
                        
                        # 1. Tabla de Métricas Finales
                        st.markdown("#### 🏆 Métricas Finales por Idioma")
                        
                        # Filtrar última época por modelo y lenguaje
                        final_metrics = []
                        for lang in exp_config['languages']:
                            for model in ['listener', 'recognizer', 'reader']:
                                # Obtener datos para este modelo e idioma
                                df_subset = df_results[(df_results['language'] == lang) & (df_results['model'] == model)]
                                if not df_subset.empty:
                                    # Obtener la fila con la época más alta
                                    last_epoch = df_subset.loc[df_subset['epoch'].idxmax()]
                                    
                                    metric_row = {
                                        "Idioma": lang.upper(),
                                        "Modelo": model.capitalize(),
                                        "Epochs": int(last_epoch['epoch']) + 1,
                                        "Val Loss": f"{last_epoch['val_loss']:.4f}",
                                        "Train Loss": f"{last_epoch['train_loss']:.4f}"
                                    }
                                    
                                    # Agregar métricas específicas
                                    if model in ['listener', 'recognizer']:
                                        metric_row["Val Acc"] = f"{last_epoch['val_top1']:.2f}%"
                                        metric_row["Train Acc"] = f"{last_epoch['train_top1']:.2f}%"
                                    elif model == 'reader':
                                        metric_row["Val MSE"] = f"{last_epoch.get('val_mse', 0):.4f}"
                                        metric_row["Val Cos"] = f"{last_epoch.get('val_cos', 0):.4f}"
                                    
                                    final_metrics.append(metric_row)
                        
                        st.dataframe(pd.DataFrame(final_metrics), use_container_width=True)
                        
                        # 2. Gráficas Comparativas de Entrenamiento
                        st.markdown("#### 📈 Curvas de Entrenamiento Comparativas")
                        
                        plot_tabs = st.tabs(["Listener (Loss)", "Listener (Acc)", "Recognizer (Loss)", "Recognizer (Acc)", "Reader (Loss)"])
                        
                        # Helper para plotear comparativas
                        def plot_comparative(df, model_type, metric, title, y_label):
                            df_model = df[df['model'] == model_type]
                            if df_model.empty:
                                st.info(f"No hay datos para {model_type}")
                                return
                                
                            # Pivotar datos para st.line_chart: Index=Epoch, Columns=Language, Values=Metric
                            df_pivot = df_model.pivot(index='epoch', columns='language', values=metric)
                            
                            st.markdown(f"**{title}**")
                            st.line_chart(df_pivot)

                        with plot_tabs[0]:
                            plot_comparative(df_results, 'listener', 'val_loss', "Listener Validation Loss", "Loss")
                        with plot_tabs[1]:
                            plot_comparative(df_results, 'listener', 'val_top1', "Listener Validation Accuracy", "Accuracy (%)")
                        with plot_tabs[2]:
                            plot_comparative(df_results, 'recognizer', 'val_loss', "Recognizer Validation Loss", "Loss")
                        with plot_tabs[3]:
                            plot_comparative(df_results, 'recognizer', 'val_top1', "Recognizer Validation Accuracy", "Accuracy (%)")
                        with plot_tabs[4]:
                            plot_comparative(df_results, 'reader', 'val_loss', "Reader Validation Loss", "Loss")

                        # 3. Análisis Estadístico
                        st.markdown("#### 📐 Análisis Estadístico de Diferencias")
                        if len(exp_config['languages']) >= 2:
                            perform_statistical_analysis(df_results, exp_config['languages'])
                        else:
                            st.info("Se necesitan al menos 2 idiomas para realizar análisis estadístico.")
                            
                        st.markdown("---")
                        st.markdown("### 🔬 Evaluación Profunda (Deep Evaluation)")
                        st.info("Esta sección carga los modelos entrenados y evalúa su capacidad predictiva real en el set de validación.")
                        
                        if st.button("🚀 Ejecutar Evaluación Profunda", type="primary"):
                            # Cargar checkpoints.json
                            checkpoints_path = exp_path / "checkpoints.json"
                            if not checkpoints_path.exists():
                                st.error("No se encontró checkpoints.json")
                            else:
                                with open(checkpoints_path) as f:
                                    ckpt_map = json.load(f)
                                
                                # Contenedores para resultados
                                eval_results = []
                                
                                # Tabs por tipo de modelo
                                deep_tabs = st.tabs(["👂 Listener Analysis", "👁️ Recognizer Analysis", "🧠 Reader Analysis"])
                                
                                # --- LISTENER DEEP EVAL ---
                                with deep_tabs[0]:
                                    st.markdown("#### Comparativa de Listeners")
                                    cols = st.columns(len(exp_config['languages']))
                                    
                                    # Imports necesarios ya están arriba
                                    from training.audio_dataset import build_audio_dataloaders
                                    from components.analytics import plot_confusion_matrix, plot_probability_matrix
                                    
                                    for i, lang in enumerate(exp_config['languages']):
                                        with cols[i]:
                                            st.markdown(f"##### {lang.upper()}")
                                            l_path = ckpt_map.get(f"listener_{lang}")
                                            
                                            if l_path and Path(l_path).exists():
                                                with st.spinner(f"Evaluando {lang}..."):
                                                    try:
                                                        # Cargar vocabulario
                                                        meta_path = Path(l_path).with_suffix(".ckpt.meta.json")
                                                        vocab = []
                                                        if meta_path.exists():
                                                            with open(meta_path) as f:
                                                                vocab = json.load(f).get('config', {}).get('vocab', [])
                                                        
                                                        if not vocab:
                                                            ds, _, _, _ = build_audio_dataloaders(target_language=lang, num_workers=0, seed=42)
                                                            vocab = ds.class_names
                                                            
                                                        # Cargar modelo
                                                        model = PhonologicalPathwayLightning.load_from_checkpoint(l_path, class_names=vocab)
                                                        model.eval()
                                                        device = encontrar_device()
                                                        model.to(device)
                                                        
                                                        # Datos Val
                                                        _, _, _, loaders = build_audio_dataloaders(batch_size=16, target_language=lang, num_workers=0, seed=42)
                                                        val_loader = loaders['val']
                                                        
                                                        # Inferencia
                                                        all_preds, all_labels, all_probs = [], [], []
                                                        with torch.no_grad():
                                                            for batch in val_loader:
                                                                waveforms = [w.to(device) for w in batch['waveforms']]
                                                                labels = batch['label'].to(device)
                                                                logits = model(waveforms)
                                                                probs = torch.softmax(logits, dim=1)
                                                                preds = torch.argmax(probs, dim=1)
                                                                
                                                                all_preds.extend(preds.cpu().numpy())
                                                                all_labels.extend(labels.cpu().numpy())
                                                                all_probs.extend(probs.cpu().numpy())
                                                        
                                                        # Métricas
                                                        acc = np.mean(np.array(all_preds) == np.array(all_labels))
                                                        st.metric("Validation Accuracy", f"{acc:.2%}")
                                                        
                                                        # Matrices
                                                        st.markdown("**Matriz de Confusión**")
                                                        plot_confusion_matrix(all_labels, all_preds, vocab)
                                                        
                                                        st.markdown("**Matriz de Probabilidades**")
                                                        plot_probability_matrix(all_labels, np.array(all_probs), vocab)
                                                        
                                                        eval_results.append({
                                                            "language": lang,
                                                            "model": "listener",
                                                            "val_acc": acc
                                                        })
                                                        
                                                    except Exception as e:
                                                        st.error(f"Error: {e}")
                                            else:
                                                st.warning("Checkpoint no encontrado")

                                # --- RECOGNIZER DEEP EVAL ---
                                with deep_tabs[1]:
                                    st.markdown("#### Comparativa de Recognizers")
                                    cols_r = st.columns(len(exp_config['languages']))
                                    
                                    from training.visual_dataset import build_visual_dataloaders
                                    
                                    for i, lang in enumerate(exp_config['languages']):
                                        with cols_r[i]:
                                            st.markdown(f"##### {lang.upper()}")
                                            r_path = ckpt_map.get(f"recognizer_{lang}")
                                            
                                            if r_path and Path(r_path).exists():
                                                with st.spinner(f"Evaluando {lang}..."):
                                                    try:
                                                        # Cargar datos
                                                        ds, _, _, loaders = build_visual_dataloaders(target_language=lang, num_workers=0, seed=42)
                                                        vocab = ds.letters
                                                        
                                                        model = VisualPathwayLightning.load_from_checkpoint(r_path, num_classes=len(vocab))
                                                        model.eval()
                                                        model.to(device)
                                                        
                                                        val_loader = loaders['val']
                                                        all_preds, all_labels, all_probs = [], [], []
                                                        
                                                        with torch.no_grad():
                                                            for batch in val_loader:
                                                                imgs = batch['image'].to(device)
                                                                labels = batch['label'].to(device)
                                                                logits = model(imgs)
                                                                probs = torch.softmax(logits, dim=1)
                                                                preds = torch.argmax(probs, dim=1)
                                                                
                                                                all_preds.extend(preds.cpu().numpy())
                                                                all_labels.extend(labels.cpu().numpy())
                                                                all_probs.extend(probs.cpu().numpy())
                                                        
                                                        acc = np.mean(np.array(all_preds) == np.array(all_labels))
                                                        st.metric("Validation Accuracy", f"{acc:.2%}")
                                                        
                                                        st.markdown("**Matriz de Confusión**")
                                                        plot_confusion_matrix(all_labels, all_preds, vocab)
                                                        
                                                        st.markdown("**Matriz de Probabilidades**")
                                                        plot_probability_matrix(all_labels, np.array(all_probs), vocab)
                                                        
                                                    except Exception as e:
                                                        st.error(f"Error: {e}")
                                            else:
                                                st.warning("Checkpoint no encontrado")
                                
                                # --- READER DEEP EVAL ---
                                with deep_tabs[2]:
                                    st.markdown("#### Comparativa de Readers")
                                    cols_read = st.columns(len(exp_config['languages']))
                                    
                                    from torch.nn.utils.rnn import pad_sequence
                                    import torch.nn.functional as F
                                    
                                    for i, lang in enumerate(exp_config['languages']):
                                        with cols_read[i]:
                                            st.markdown(f"##### {lang.upper()}")
                                            read_path = ckpt_map.get(f"reader_{lang}")
                                            l_path = ckpt_map.get(f"listener_{lang}")
                                            r_path = ckpt_map.get(f"recognizer_{lang}")
                                            
                                            if read_path and Path(read_path).exists():
                                                with st.spinner(f"Evaluando Reader {lang}..."):
                                                    try:
                                                        # Cargar datos
                                                        ds, _, _, loaders = build_audio_dataloaders(batch_size=16, target_language=lang, num_workers=0, seed=42)
                                                        vocab = ds.class_names
                                                        
                                                        # Cargar modelo con paths actualizados
                                                        model = TinyReaderLightning.load_from_checkpoint(
                                                            read_path,
                                                            class_names=vocab,
                                                            listener_checkpoint_path=l_path,
                                                            recognizer_checkpoint_path=r_path,
                                                            strict=False
                                                        )
                                                        model.eval()
                                                        device = encontrar_device()
                                                        model.to(device)
                                                        
                                                        val_loader = loaders['val']
                                                        mse_list, cos_list = [], []
                                                        all_preds, all_labels, all_probs = [], [], []
                                                        
                                                        with torch.no_grad():
                                                            for batch in val_loader:
                                                                waveforms = [w.to(device) for w in batch['waveforms']]
                                                                labels = batch['label'].to(device)
                                                                
                                                                # 1. Ground Truth (Listener)
                                                                # Pad waveforms correctly
                                                                waveforms_padded = pad_sequence(waveforms, batch_first=True)
                                                                real_emb = model.listener.extract_hidden_activations(waveforms_padded)
                                                                real_emb, lengths = model.listener.mask_hidden_activations(real_emb)
                                                                real_emb, lengths = model.listener.downsample_hidden_activations(real_emb, lengths, factor=7)
                                                                real_emb = real_emb.squeeze(0)
                                                                
                                                                # 2. Generación (Reader)
                                                                words = [vocab[idx] for idx in labels]
                                                                logits_seq = []
                                                                for w in words:
                                                                    imgs = model._get_word_images(w).to(device)
                                                                    res = model.recognizer(imgs)
                                                                    logits = res[0] if isinstance(res, tuple) else res
                                                                    logits_seq.append(logits)
                                                                
                                                                padded_logits = pad_sequence(logits_seq, batch_first=True, padding_value=0.0)
                                                                max_len = real_emb.size(1)
                                                                gen_emb = model.reader(padded_logits, target_length=max_len)
                                                                
                                                                # 3. Métricas de Reconstrucción
                                                                mask = torch.arange(max_len, device=device).expand(len(waveforms), max_len) < lengths.unsqueeze(1)
                                                                
                                                                mse = F.mse_loss(gen_emb[mask], real_emb[mask])
                                                                
                                                                gen_flat = gen_emb[mask]
                                                                real_flat = real_emb[mask]
                                                                target_ones = torch.ones(gen_flat.size(0), device=device)
                                                                cos = F.cosine_embedding_loss(gen_flat, real_flat, target_ones)
                                                                
                                                                mse_list.append(mse.item())
                                                                cos_list.append(cos.item())
                                                                
                                                                # 4. Clasificación de lo Imaginado (Perceptual Evaluation)
                                                                # Empaquetar para LSTM del Listener
                                                                from torch.nn.utils.rnn import pack_padded_sequence
                                                                packed_gen = pack_padded_sequence(
                                                                    gen_emb, lengths.cpu(), batch_first=True, enforce_sorted=False
                                                                )
                                                                # PhonologicalPathway usa Mean Pooling + Classifier
                                                                # Masking para mean pooling correcto
                                                                mask_float = mask.float().unsqueeze(-1)
                                                                pooled_gen = (gen_emb * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-9)
                                                                listener_logits = model.listener.classifier(pooled_gen)
                                                                probs = torch.softmax(listener_logits, dim=1)
                                                                preds = torch.argmax(probs, dim=1)
                                                                
                                                                all_preds.extend(preds.cpu().numpy())
                                                                all_labels.extend(labels.cpu().numpy())
                                                                all_probs.extend(probs.cpu().numpy())
                                                        
                                                        avg_mse = np.mean(mse_list)
                                                        avg_cos = np.mean(cos_list)
                                                        acc = np.mean(np.array(all_preds) == np.array(all_labels))
                                                        
                                                        st.metric("MSE Loss (↓)", f"{avg_mse:.4f}")
                                                        st.metric("Cosine Loss (↓)", f"{avg_cos:.4f}")
                                                        st.metric("Imagination Accuracy (↑)", f"{acc:.2%}")
                                                        
                                                        st.markdown("**Matriz de Confusión (Imaginación)**")
                                                        plot_confusion_matrix(all_labels, all_preds, vocab)
                                                        
                                                        st.markdown("**Matriz de Probabilidades (Imaginación)**")
                                                        plot_probability_matrix(all_labels, np.array(all_probs), vocab)
                                                        
                                                        eval_results.append({
                                                            "language": lang,
                                                            "model": "reader",
                                                            "val_mse": avg_mse,
                                                            "val_cos": avg_cos,
                                                            "val_acc": acc
                                                        })
                                                        
                                                    except Exception as e:
                                                        st.error(f"Error: {e}")
                                            else:
                                                st.warning("Checkpoint no encontrado")

                                # --- CORRELATION ANALYSIS ---
                                if eval_results:
                                    st.markdown("---")
                                    st.markdown("### 📉 Correlación Entrenamiento vs Validación")
                                    st.info("Este gráfico muestra qué tan bien generaliza el modelo: ¿Un mejor entrenamiento implica una mejor predicción real?")
                                    
                                    # Convertir a DataFrame
                                    df_eval = pd.DataFrame(eval_results)
                                    
                                    # Obtener métricas de entrenamiento desde results.csv para cruzar datos
                                    train_accs = []
                                    for res in eval_results:
                                        lang = res['language']
                                        model_type = res['model']
                                        # Buscar en df_results
                                        subset = df_results[(df_results['language'] == lang) & (df_results['model'] == model_type)]
                                        if not subset.empty:
                                            train_acc = subset.iloc[-1]['train_top1'] / 100.0 # Asumiendo que está en %
                                            train_accs.append(train_acc)
                                        else:
                                            train_accs.append(0)
                                    
                                    df_eval['train_acc'] = train_accs
                                    
                                    # Plot
                                    fig, ax = plt.subplots(figsize=(8, 6))
                                    sns.scatterplot(data=df_eval, x='train_acc', y='val_acc', hue='language', style='model', s=200, ax=ax)
                                    
                                    # Línea de identidad (ideal)
                                    min_val = min(df_eval['train_acc'].min(), df_eval['val_acc'].min())
                                    max_val = max(df_eval['train_acc'].max(), df_eval['val_acc'].max())
                                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Ideal (Generalización Perfecta)')
                                    
                                    ax.set_xlabel("Training Accuracy (Final Epoch)")
                                    ax.set_ylabel("Validation Accuracy (Deep Eval)")
                                    ax.set_title("Generalization Gap: Train vs Val")
                                    ax.legend()
                                    ax.grid(True, alpha=0.3)
                                    st.pyplot(fig)




                else:
                    st.error("El experimento seleccionado parece estar incompleto (faltan archivos de configuración).")

    # ==========================================
    # TAB 4: LABORATORIO
    # ==========================================
    with tabs[3]:
        st.markdown("### 🧪 Laboratorio Comparativo")
        
        # Buscar experimentos disponibles
        experiments_dir = Path("experiments")
        available_experiments = []
        if experiments_dir.exists():
            available_experiments = sorted([d.name for d in experiments_dir.iterdir() if d.is_dir()], reverse=True)
        
        # Intentar seleccionar el último experimento usado o el más reciente
        default_idx = 0
        last_exp_path = st.session_state.get('last_experiment_path', None)
        if last_exp_path:
            last_exp_name = Path(last_exp_path).name
            if last_exp_name in available_experiments:
                default_idx = available_experiments.index(last_exp_name)
        
        selected_exp_name = st.selectbox(
            "Seleccionar Experimento",
            available_experiments,
            index=default_idx,
            key="lab_exp_selector"
        )
        
        if selected_exp_name:
            exp_path = experiments_dir / selected_exp_name
            run_comparative_laboratory(exp_path)
        else:
            st.info("No se encontraron experimentos guardados.")

if __name__ == "__main__":
    main()
