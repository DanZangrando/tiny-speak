import streamlit as st
import pandas as pd
import plotly.express as px
import time
from pathlib import Path
import uuid

from components.modern_sidebar import display_modern_sidebar
from experiments.experiment_utils import save_experiment, load_experiment, list_experiments, get_opacity_index, check_dataset_status
from training.runner import train_recognizer, train_listener, train_reader

# Configurar página
st.set_page_config(
    page_title="Experimento de Transparencia",
    page_icon="🔬",
    layout="wide"
)

def get_custom_css():
    return """
    <style>
    .main-header {
        background: linear-gradient(90deg, #6B66FF, #DF66FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: var(--secondary-background-color);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #6B66FF;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """

def main():
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    display_modern_sidebar("transparency_experiment")
    
    st.markdown('<h1 class="main-header">🔬 Experimento de Transparencia Ortográfica</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Este experimento automatizado investiga cómo la **opacidad ortográfica** de diferentes idiomas afecta la capacidad de aprendizaje de TinyReader.
    
    **Hipótesis:** Idiomas con ortografía transparente (ej. Español, Italiano) serán aprendidos más eficientemente por TinyReader que idiomas opacos (ej. Inglés, Francés), reflejando la dificultad humana.
    """)
    
    tabs = st.tabs(["📋 Diseño & Verificación", "⚙️ Ejecución", "📊 Análisis de Resultados"])
    
    # ==========================================
    # TAB 1: DISEÑO & VERIFICACIÓN
    # ==========================================
    with tabs[0]:
        st.header("Diseño Experimental")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 1. Selección de Idiomas")
            available_languages = ['es', 'en', 'fr', 'de', 'it', 'pt']
            selected_languages = st.multiselect(
                "Idiomas a incluir en el experimento:",
                available_languages,
                default=['es', 'en']
            )
            
            st.info(f"Se entrenarán modelos completos para: {', '.join(selected_languages)}")
            
        with col2:
            st.markdown("### 2. Verificación de Datasets")
            if st.button("🔍 Verificar Disponibilidad de Datos"):
                status_data = []
                all_ok = True
                for lang in selected_languages:
                    status = check_dataset_status(lang)
                    status_data.append({
                        "Idioma": lang,
                        "Audio": "✅" if status['audio'] else "❌",
                        "Visual": "✅" if status['visual'] else "❌ (Gen)"
                    })
                    if not status['audio']: all_ok = False
                
                st.table(pd.DataFrame(status_data))
                
                if all_ok:
                    st.success("Todos los datasets necesarios están disponibles.")
                else:
                    st.error("Faltan datasets de audio para algunos idiomas. Por favor, descárgalos o genéralos antes de continuar.")

        st.markdown("### 3. Pipeline de Entrenamiento (Por Idioma)")
        st.graphviz_chart("""
        digraph G {
            rankdir=LR;
            A [label="Dataset", shape=box];
            B [label="TinyEyes\\nRecognizer", shape=ellipse];
            C [label="TinyEars\\nPhonemes", shape=ellipse];
            D [label="TinyEars\\nWords", shape=ellipse];
            E [label="TinySpeller\\n(G2P)", shape=diamond];
            F [label="TinyReader\\n(P2W)", shape=diamond];
            
            A -> B;
            A -> C;
            A -> D;
            B -> E;
            C -> E;
            B -> F;
            D -> F;
            E -> F [style=dashed, label="Pesos"];
        }
        """)

    # ==========================================
    # TAB 2: EJECUCIÓN
    # ==========================================
    with tabs[1]:
        st.header("Ejecución del Experimento")
        
        if not selected_languages:
            st.warning("Selecciona al menos un idioma en la pestaña de Diseño.")
        else:
            with st.expander("⚙️ Configuración de Entrenamiento", expanded=True):
                col_conf1, col_conf2 = st.columns(2)
                with col_conf1:
                    exp_name = st.text_input("Nombre del Experimento", value=f"Exp_Transparencia_{time.strftime('%Y%m%d')}")
                    epochs_eyes = st.number_input("Épocas TinyEyes", value=5, min_value=1)
                    epochs_ears = st.number_input("Épocas TinyEars (Phon/Word)", value=10, min_value=1)
                with col_conf2:
                    epochs_speller = st.number_input("Épocas TinySpeller (G2P)", value=15, min_value=1)
                    epochs_reader = st.number_input("Épocas TinyReader (P2W)", value=20, min_value=1)
            
            if st.button("🚀 INICIAR EXPERIMENTO AUTOMATIZADO", type="primary"):
                experiment_id = str(uuid.uuid4())[:8]
                st.session_state['current_experiment_id'] = experiment_id
                
                full_results = {
                    "id": experiment_id,
                    "name": exp_name,
                    "languages": selected_languages,
                    "config": {
                        "epochs_eyes": epochs_eyes,
                        "epochs_ears": epochs_ears,
                        "epochs_speller": epochs_speller,
                        "epochs_reader": epochs_reader
                    },
                    "models": {},
                    "metrics": {}
                }
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                log_area = st.empty()
                
                total_steps = len(selected_languages) * 5 # 5 stages per language
                current_step = 0
                
                logs = []
                def log(msg):
                    logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
                    log_area.code("\n".join(logs[-10:])) # Show last 10 lines
                    status_text.markdown(f"**{msg}**")
                
                try:
                    for lang in selected_languages:
                        full_results["models"][lang] = {}
                        full_results["metrics"][lang] = {}
                        
                        # 1. TinyEyes
                        log(f"[{lang}] Entrenando TinyEyes (Recognizer)...")
                        eyes_path, eyes_hist = train_recognizer(lang, {"epochs": epochs_eyes})
                        full_results["models"][lang]["eyes"] = eyes_path
                        full_results["metrics"][lang]["eyes"] = eyes_hist # Save full history
                        current_step += 1
                        progress_bar.progress(current_step / total_steps)
                        
                        # 2. TinyEars (Phonemes)
                        log(f"[{lang}] Entrenando TinyEars (Phonemes)...")
                        ears_ph_path, ears_ph_hist = train_listener(lang, {"epochs": epochs_ears, "use_phonemes": True})
                        full_results["models"][lang]["ears_phonemes"] = ears_ph_path
                        full_results["metrics"][lang]["ears_phonemes"] = ears_ph_hist # Save full history
                        current_step += 1
                        progress_bar.progress(current_step / total_steps)
                        
                        # 3. TinyEars (Words)
                        log(f"[{lang}] Entrenando TinyEars (Words)...")
                        ears_w_path, ears_w_hist = train_listener(lang, {"epochs": epochs_ears, "use_phonemes": False})
                        full_results["models"][lang]["ears_words"] = ears_w_path
                        full_results["metrics"][lang]["ears_words"] = ears_w_hist # Save full history
                        current_step += 1
                        progress_bar.progress(current_step / total_steps)
                        
                        # 4. TinySpeller (G2P)
                        log(f"[{lang}] Entrenando TinySpeller (G2P)...")
                        speller_path, speller_hist = train_reader(
                            language=lang,
                            listener_ckpt=ears_ph_path, # Uses Phoneme Listener
                            recognizer_ckpt=eyes_path,
                            config={
                                "epochs": epochs_speller,
                                "training_phase": "g2p",
                                "use_two_stage": True,
                                "phoneme_listener_ckpt": ears_ph_path
                            }
                        )
                        full_results["models"][lang]["speller"] = speller_path
                        full_results["metrics"][lang]["speller"] = speller_hist # Save full history
                        current_step += 1
                        progress_bar.progress(current_step / total_steps)
                        
                        # 5. TinyReader (P2W) - Full Pipeline
                        log(f"[{lang}] Entrenando TinyReader (P2W)...")
                        reader_path, reader_hist = train_reader(
                            language=lang,
                            listener_ckpt=ears_w_path, # Uses Word Listener
                            recognizer_ckpt=eyes_path,
                            config={
                                "epochs": epochs_reader,
                                "training_phase": "p2w",
                                "use_two_stage": True,
                                "phoneme_listener_ckpt": ears_ph_path,
                                "pretrained_speller_ckpt": speller_path # Load trained G2P weights
                            }
                        )
                        full_results["models"][lang]["reader"] = reader_path
                        full_results["metrics"][lang]["reader"] = reader_hist # Save full history
                        current_step += 1
                        progress_bar.progress(current_step / total_steps)
                        
                        log(f"✅ [{lang}] Completado.")
                    
                    # Save Experiment
                    save_path = save_experiment(experiment_id, full_results)
                    st.success(f"Experimento completado y guardado en: {save_path}")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Error durante el experimento: {e}")
                    st.exception(e)

    # ==========================================
    # TAB 3: ANÁLISIS
    # ==========================================
    with tabs[2]:
        st.header("Análisis de Resultados")
        
        experiments = list_experiments()
        if not experiments:
            st.info("No hay experimentos guardados.")
        else:
            exp_opts = {f"{e['name']} ({e['id']})": e['id'] for e in experiments}
            sel_exp_name = st.selectbox("Seleccionar Experimento", list(exp_opts.keys()))
            sel_exp_id = exp_opts[sel_exp_name]
            
            exp_data = load_experiment(sel_exp_id)
            
            if exp_data:
                st.markdown(f"**Idiomas:** {', '.join(exp_data['languages'])}")
                
                # Prepare Data for Plotting
                plot_data = []
                for lang in exp_data['languages']:
                    metrics = exp_data['metrics'].get(lang, {})
                    
                    # Get Reader Accuracy (Top-1 or similar)
                    # Assuming metric key is 'val_acc' or similar. If not found, use 0.
                    reader_metrics = metrics.get('reader', {})
                    
                    # Handle both dict (old) and list (new) formats
                    final_metrics = {}
                    if isinstance(reader_metrics, list):
                        if reader_metrics:
                            final_metrics = reader_metrics[-1]
                    elif isinstance(reader_metrics, dict):
                        final_metrics = reader_metrics
                    
                    # Try different keys
                    acc = final_metrics.get('val_acc_epoch', final_metrics.get('val_acc', 0.0))
                    loss = final_metrics.get('val_loss', 0.0)
                    
                    opacity = get_opacity_index(lang)
                    
                    plot_data.append({
                        "Idioma": lang,
                        "Opacidad (Teórica)": opacity,
                        "Precisión Reader": acc,
                        "Pérdida Reader": loss
                    })
                
                df_plot = pd.DataFrame(plot_data)
                
                col_plots1, col_plots2 = st.columns(2)
                
                with col_plots1:
                    st.markdown("### Precisión por Idioma")
                    fig_bar = px.bar(df_plot, x="Idioma", y="Precisión Reader", color="Opacidad (Teórica)",
                                     title="Performance TinyReader vs Idioma",
                                     color_continuous_scale="RdBu_r") # Red (Opaque) to Blue (Transparent)
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                with col_plots2:
                    st.markdown("### Hipótesis de Transparencia")
                    fig_scat = px.scatter(df_plot, x="Opacidad (Teórica)", y="Precisión Reader",
                                          text="Idioma", size_max=60,
                                          title="Correlación Opacidad vs Precisión",
                                          labels={"Opacidad (Teórica)": "Opacidad (0=Transp, 1=Opaco)"})
                    fig_scat.update_traces(textposition='top center', marker=dict(size=20))
                    # Add trendline if enough points?
                    st.plotly_chart(fig_scat, use_container_width=True)
                
                st.markdown("### Detalles Métricas")
                st.dataframe(df_plot)
                
                st.markdown("""
                **Interpretación:**
                Si la hipótesis es correcta, deberíamos ver una **correlación negativa**: a mayor opacidad (más a la derecha), menor precisión (más abajo).
                """)
                
                st.divider()
                st.markdown("### 📈 Detalle por Idioma (Curvas de Aprendizaje)")
                
                sel_lang_detail = st.selectbox("Seleccionar Idioma para Detalle", exp_data['languages'])
                
                if sel_lang_detail:
                    lang_metrics = exp_data['metrics'].get(sel_lang_detail, {})
                    
                    # Tabs para cada modelo
                    tabs_models = st.tabs(["TinyEyes", "TinyEars (Ph)", "TinyEars (W)", "TinySpeller", "TinyReader"])
                    
                    model_keys = ["eyes", "ears_phonemes", "ears_words", "speller", "reader"]
                    
                    for i, key in enumerate(model_keys):
                        with tabs_models[i]:
                            data = lang_metrics.get(key, [])
                            
                            if not data:
                                st.info("No hay datos disponibles.")
                            elif isinstance(data, dict):
                                st.warning("Datos en formato antiguo (solo última época). Ejecuta un nuevo experimento para ver curvas.")
                                st.json(data)
                            elif isinstance(data, list):
                                df_hist = pd.DataFrame(data)
                                
                                col_g1, col_g2 = st.columns(2)
                                
                                with col_g1:
                                    st.markdown("#### Pérdida (Loss)")
                                    loss_cols = [c for c in df_hist.columns if 'loss' in c]
                                    if loss_cols:
                                        st.line_chart(df_hist[loss_cols])
                                    else:
                                        st.info("No hay métricas de pérdida.")
                                        
                                with col_g2:
                                    st.markdown("#### Precisión (Accuracy)")
                                    acc_cols = [c for c in df_hist.columns if 'acc' in c or 'top1' in c]
                                    if acc_cols:
                                        st.line_chart(df_hist[acc_cols])
                                    else:
                                        st.info("No hay métricas de precisión.")
                                        
                                st.dataframe(df_hist, use_container_width=True)

if __name__ == "__main__":
    main()
