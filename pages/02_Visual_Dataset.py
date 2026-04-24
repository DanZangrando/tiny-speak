import streamlit as st
import json
from pathlib import Path
from PIL import Image
from components.modern_sidebar import display_modern_sidebar
from training.config import load_master_dataset_config
from utils.graphemes import get_language_letters

# Configurar página
st.set_page_config(
    page_title="Explorador Visual - TinySpeak",
    page_icon="🖼️",
    layout="wide"
)

def main():
    display_modern_sidebar("visual_dataset")
    
    st.title("🖼️ Explorador de Dataset Visual")
    st.markdown("Inspecciona las representaciones gráficas de los caracteres utilizados en el experimento.")
    
    config = load_master_dataset_config()
    visual_data = config.get("visual_dataset", {}).get("generated_images", {})
    
    if not visual_data:
        st.warning("⚠️ No se han encontrado imágenes generadas. Ve a la página de Configuración para generar el dataset.")
        return

    # Dashboard de caracteres
    st.markdown("---")
    
    # Selector de Idioma para filtrar letras
    selected_lang = st.segmented_control(
        "Filtrar por Idioma", 
        options=['es', 'en', 'fr'], 
        format_func=lambda x: {'es': '🇪🇸 Español', 'en': '🇺🇸 Inglés', 'fr': '🇫🇷 Francés'}.get(x, x.upper()),
        default='es'
    )
    
    lang_letters = get_language_letters(selected_lang)
    present_letters = [l for l in lang_letters if l in visual_data]
    
    st.info(f"Mostrando caracteres detectados para **{selected_lang.upper()}**: {len(present_letters)} caracteres.")
    
    # Grid de inspección
    if not present_letters:
        st.error("No hay imágenes generadas para las letras de este idioma.")
    else:
        selected_char = st.selectbox("🔍 Selecciona un carácter para inspeccionar variaciones:", sorted(present_letters))
        
        if selected_char:
            variations = visual_data[selected_char]
            st.write(f"### Carácter: **{selected_char.upper()}** ({len(variations)} variaciones)")
            
            # Mostrar variaciones en un grid
            cols_per_row = 5
            rows = [variations[i:i + cols_per_row] for i in range(0, len(variations), cols_per_row)]
            
            root = Path(__file__).parent.parent
            
            for row_vars in rows:
                cols = st.columns(len(row_vars))
                for i, var in enumerate(row_vars):
                    with cols[i]:
                        image_path = root / var.get('file_path', '')
                        if image_path.exists():
                            img = Image.open(image_path)
                            st.image(img, width='stretch')
                            p = var.get('params', {})
                            st.caption(f"Rot: {p.get('rotation', 0):.1f}°\nNoi: {p.get('noise_level', 0):.2f}")
                        else:
                            st.error("No file")

    st.markdown("---")
    st.header("📊 Resumen del Alfabeto Global")
    
    all_chars = sorted(list(visual_data.keys()))
    char_list_str = " ".join([f"`{c.upper()}`" for c in all_chars])
    st.markdown(f"**Caracteres totales en el dataset:** {len(all_chars)}")
    st.markdown(char_list_str)

if __name__ == "__main__":
    main()