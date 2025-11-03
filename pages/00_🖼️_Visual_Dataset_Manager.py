import streamlit as st
import json
import os
import base64
import io
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import string
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from components.modern_sidebar import display_modern_sidebar

st.set_page_config(
    page_title="📊 Visual Dataset Manager - TinySpeak",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Aplicar estilos CSS modernos
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 2rem;
}

.modern-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
    border: none;
    text-align: center;
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    border-left: 4px solid #00d4ff;
    margin-bottom: 0.5rem;
}

.status-success {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: bold;
}

.status-warning {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: bold;
}

.sidebar-header {
    background: linear-gradient(90deg, #667eea, #764ba2);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

def load_master_config():
    """Cargar configuración desde master_dataset_config.json"""
    config_file = "../master_dataset_config.json"
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Asegurar que existe la sección visual
            if 'visual_dataset' not in config:
                config['visual_dataset'] = {
                    "name": "Visual Dataset TinySpeak",
                    "letters": list(string.ascii_lowercase),
                    "vocabulary": config.get('master_config_reference', 'casa_familia'),  # Sincronizar con master
                    "fonts": ["arial.ttf", "times.ttf", "calibri.ttf"],
                    "font_sizes": [20, 24, 28, 32, 36, 40],
                    "rotation_range": 15,
                    "noise_levels": [0.0, 0.1, 0.2, 0.3],
                    "image_size": [64, 64],
                    "generated_images": {},
                    "version": "2.0",
                    "created": datetime.now().isoformat()
                }
                save_master_config(config)
            
            return config
        
        except Exception as e:
            st.error(f"Error cargando configuración maestra: {e}")
            return create_default_master_config()
    else:
        return create_default_master_config()

def create_default_master_config():
    """Crea configuración maestra por defecto"""
    config = {
        "master_config_reference": "casa_familia",
        "visual_dataset": {
            "name": "Visual Dataset TinySpeak",
            "letters": list(string.ascii_lowercase),
            "vocabulary": "casa_familia",
            "fonts": ["arial.ttf", "times.ttf", "calibri.ttf"],
            "font_sizes": [20, 24, 28, 32, 36, 40],
            "rotation_range": 15,
            "noise_levels": [0.0, 0.1, 0.2, 0.3],
            "image_size": [64, 64],
            "generated_images": {},
            "version": "2.0",
            "created": datetime.now().isoformat()
        }
    }
    
    save_master_config(config)
    return config

def save_master_config(config):
    """Guardar configuración a master_dataset_config.json"""
    try:
        config_file = "../master_dataset_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False, default=str)
        return True
    except Exception as e:
        st.error(f"Error guardando configuración: {e}")
        return False

def update_visual_vocabulary():
    """Actualiza el vocabulario visual para sincronizar con el master config"""
    config = load_master_config()
    
    # Sincronizar vocabulario visual con master_config_reference
    master_vocab = config.get('master_config_reference', 'casa_familia')
    config['visual_dataset']['vocabulary'] = master_vocab
    
    # Obtener letras únicas del vocabulario master
    if master_vocab in config.get('generated_samples', {}):
        vocab_letters = set()
        for sample in config['generated_samples'][master_vocab]:
            for char in sample.get('word', '').lower():
                if char.isalpha():
                    vocab_letters.add(char)
        
        config['visual_dataset']['letters'] = sorted(list(vocab_letters))
    
    save_master_config(config)
    return config

def generate_letter_image(letter, font_size=32, rotation=0, noise_level=0.0, font_name="arial.ttf"):
    """Genera una imagen de una letra con parámetros específicos"""
    try:
        # Crear imagen base
        img_size = (64, 64)
        img = Image.new('L', img_size, color=255)  # Fondo blanco
        draw = ImageDraw.Draw(img)
        
        # Intentar cargar font (usar font por defecto si falla)
        try:
            font = ImageFont.truetype(font_name, font_size)
        except:
            font = ImageFont.load_default()
        
        # Calcular posición centrada
        bbox = draw.textbbox((0, 0), letter.upper(), font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (img_size[0] - text_width) // 2
        y = (img_size[1] - text_height) // 2
        
        # Dibujar letra
        draw.text((x, y), letter.upper(), fill=0, font=font)  # Negro sobre blanco
        
        # Aplicar rotación si se especifica
        if rotation != 0:
            img = img.rotate(rotation, fillcolor=255)
        
        # Aplicar ruido si se especifica
        if noise_level > 0:
            img_array = np.array(img)
            noise = np.random.normal(0, noise_level * 255, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)
        
        return img
    
    except Exception as e:
        st.error(f"Error generando imagen para '{letter}': {e}")
        return None

def image_to_base64(image):
    """Convierte imagen PIL a base64"""
    try:
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        return img_base64
    except Exception as e:
        st.error(f"Error convirtiendo imagen a base64: {e}")
        return None

def main():
    # Mostrar sidebar moderna
    display_modern_sidebar()
    
    # Header moderno
    st.markdown('<h1 class="main-header">🖼️ Visual Dataset Manager</h1>', unsafe_allow_html=True)
    
    # Sidebar con configuración
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h3>🔧 Configuración Visual</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Sincronizar con configuración maestra
        if st.button("🔄 Sincronizar con Master Config", type="primary"):
            update_visual_vocabulary()
            st.success("✅ Vocabulario sincronizado!")
            st.rerun()
    
    # Cargar configuración
    config = load_master_config()
    visual_config = config['visual_dataset']
    
    # Mostrar estado de sincronización
    st.markdown("---")
    sync_col1, sync_col2, sync_col3 = st.columns(3)
    
    with sync_col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎯 Vocabulario Activo</h4>
            <p><strong>{visual_config.get('vocabulary', 'N/A')}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with sync_col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🔤 Letras Objetivo</h4>
            <p><strong>{len(visual_config.get('letters', []))}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with sync_col3:
        total_images = sum(len(samples) for samples in visual_config.get('generated_images', {}).values())
        st.markdown(f"""
        <div class="metric-card">
            <h4>🖼️ Imágenes Generadas</h4>
            <p><strong>{total_images}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs de funcionalidad
    tab1, tab2, tab3, tab4 = st.tabs(["🎛️ Configuración", "🖼️ Generación", "📁 Gestión", "📋 Configuración del Dataset"])
    
    with tab1:
        configuration_tab(visual_config, config)
    
    with tab2:
        generation_tab(visual_config, config)
    
    with tab3:
        management_tab(visual_config, config)
    
    with tab4:
        dataset_configuration_tab(visual_config, config)

def configuration_tab(visual_config, master_config):
    """Tab de configuración de parámetros visuales"""
    st.header("🎛️ Configuración de Parámetros Visuales")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Configuración Básica")
        
        # Nombre del dataset
        new_name = st.text_input(
            "Nombre del dataset visual",
            value=visual_config.get('name', 'Visual Dataset TinySpeak'),
            key="visual_dataset_name"
        )
        
        # Letras objetivo (solo mostrar, se sincronizan con master)
        letters_display = ', '.join(visual_config.get('letters', [])[:10])
        if len(visual_config.get('letters', [])) > 10:
            letters_display += f" ... (+{len(visual_config['letters']) - 10} más)"
        
        st.text_input(
            "Letras objetivo (sincronizadas con Master Config)",
            value=letters_display,
            disabled=True,
            help="Las letras se sincronizan automáticamente con el vocabulario maestro"
        )
        
        # Tamaño de imagen
        img_width = st.number_input("Ancho de imagen (px)", min_value=32, max_value=128, value=visual_config.get('image_size', [64, 64])[0])
        img_height = st.number_input("Alto de imagen (px)", min_value=32, max_value=128, value=visual_config.get('image_size', [64, 64])[1])
        
        # Versión
        version = st.text_input("Versión del dataset", value=visual_config.get('version', '2.0'))
    
    with col2:
        st.subheader("🎨 Parámetros de Generación")
        
        # Tamaños de fuente
        font_sizes = st.multiselect(
            "Tamaños de fuente disponibles",
            options=[16, 20, 24, 28, 32, 36, 40, 44, 48],
            default=visual_config.get('font_sizes', [20, 24, 28, 32, 36, 40]),
            key="font_sizes_config"
        )
        
        # Rango de rotación
        rotation_range = st.slider(
            "Rango de rotación (±grados)",
            min_value=0,
            max_value=45,
            value=visual_config.get('rotation_range', 15),
            help="Las imágenes se rotarán aleatoriamente dentro de este rango"
        )
        
        # Niveles de ruido
        noise_levels = st.multiselect(
            "Niveles de ruido",
            options=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
            default=visual_config.get('noise_levels', [0.0, 0.1, 0.2, 0.3]),
            format_func=lambda x: f"{x:.2f}" if x > 0 else "Sin ruido",
            key="noise_levels_config"
        )
        
        # Fuentes disponibles
        available_fonts = ["arial.ttf", "times.ttf", "calibri.ttf", "courier.ttf", "georgia.ttf"]
        selected_fonts = st.multiselect(
            "Fuentes a utilizar",
            options=available_fonts,
            default=visual_config.get('fonts', ["arial.ttf", "times.ttf", "calibri.ttf"]),
            key="fonts_config"
        )
    
    # Guardar configuración
    st.markdown("---")
    if st.button("💾 Guardar Configuración Visual", type="primary"):
        # Actualizar configuración visual en master config
        master_config['visual_dataset']['name'] = new_name
        master_config['visual_dataset']['image_size'] = [img_width, img_height]
        master_config['visual_dataset']['version'] = version
        master_config['visual_dataset']['font_sizes'] = font_sizes
        master_config['visual_dataset']['rotation_range'] = rotation_range
        master_config['visual_dataset']['noise_levels'] = noise_levels
        master_config['visual_dataset']['fonts'] = selected_fonts
        
        if save_master_config(master_config):
            st.success("✅ Configuración visual guardada en master config!")
        else:
            st.error("❌ Error guardando configuración")

def generation_tab(visual_config, master_config):
    """Tab de generación de imágenes"""
    st.header("🖼️ Generación de Imágenes de Letras")
    
    # Información del vocabulario actual
    st.info(f"🎯 Generando imágenes para el vocabulario: **{visual_config.get('vocabulary', 'N/A')}**")
    
    generation_col1, generation_col2 = st.columns(2)
    
    with generation_col1:
        st.subheader("🔤 Selección de Letras")
        
        # Mostrar letras disponibles
        available_letters = visual_config.get('letters', [])
        
        if available_letters:
            # Opción de generar todas las letras
            generate_all = st.checkbox("Generar todas las letras", key="generate_all_letters")
            
            if not generate_all:
                # Selección manual de letras
                selected_letters = st.multiselect(
                    "Seleccionar letras específicas",
                    options=available_letters,
                    default=available_letters[:5] if len(available_letters) >= 5 else available_letters,
                    key="selected_letters_manual"
                )
            else:
                selected_letters = available_letters
                st.write(f"✅ Se generarán imágenes para todas las {len(available_letters)} letras")
        else:
            st.warning("⚠️ No hay letras disponibles. Sincroniza con el Master Config primero.")
            selected_letters = []
    
    with generation_col2:
        st.subheader("⚙️ Parámetros de Generación")
        
        # Cantidad de variaciones por letra
        variations_per_letter = st.number_input(
            "Variaciones por letra",
            min_value=1,
            max_value=50,
            value=10,
            help="Número de imágenes diferentes que se generarán para cada letra"
        )
        
        # Parámetros específicos de esta generación
        use_random_params = st.checkbox(
            "Usar parámetros aleatorios",
            value=True,
            help="Si se activa, cada imagen usará parámetros aleatorios dentro de los rangos configurados"
        )
        
        if not use_random_params:
            # Parámetros fijos
            fixed_font_size = st.selectbox("Tamaño de fuente fijo", visual_config.get('font_sizes', [32]))
            fixed_rotation = st.slider("Rotación fija", -45, 45, 0)
            fixed_noise = st.selectbox("Nivel de ruido fijo", visual_config.get('noise_levels', [0.0]))
            fixed_font = st.selectbox("Fuente fija", visual_config.get('fonts', ['arial.ttf']))
    
    # Botón de generación
    st.markdown("---")
    
    if selected_letters:
        if st.button("🚀 Generar Imágenes", type="primary", key="generate_images_btn"):
            generate_images_for_letters(
                selected_letters, 
                variations_per_letter, 
                visual_config, 
                master_config,
                use_random_params,
                {
                    'font_size': fixed_font_size if not use_random_params else None,
                    'rotation': fixed_rotation if not use_random_params else None,
                    'noise_level': fixed_noise if not use_random_params else None,
                    'font': fixed_font if not use_random_params else None
                }
            )
    else:
        st.warning("⚠️ Selecciona al menos una letra para generar imágenes")
    
    # Vista previa de parámetros
    if selected_letters:
        st.markdown("---")
        st.subheader("👀 Vista Previa")
        
        preview_letter = st.selectbox("Letra para vista previa", selected_letters, key="preview_letter")
        
        if preview_letter:
            preview_col1, preview_col2, preview_col3 = st.columns(3)
            
            # Generar 3 imágenes de muestra
            sample_params = [
                {'font_size': 24, 'rotation': -10, 'noise_level': 0.0, 'font': 'arial.ttf'},
                {'font_size': 32, 'rotation': 0, 'noise_level': 0.1, 'font': 'times.ttf'},
                {'font_size': 36, 'rotation': 15, 'noise_level': 0.2, 'font': 'calibri.ttf'}
            ]
            
            for i, (col, params) in enumerate(zip([preview_col1, preview_col2, preview_col3], sample_params)):
                with col:
                    st.write(f"**Muestra {i+1}:**")
                    try:
                        preview_img = generate_letter_image(
                            preview_letter,
                            params['font_size'],
                            params['rotation'],
                            params['noise_level'],
                            params['font']
                        )
                        
                        if preview_img:
                            st.image(preview_img, width=100)
                            st.caption(f"Font: {params['font_size']}, Rot: {params['rotation']}°, Ruido: {params['noise_level']}")
                        
                    except Exception as e:
                        st.error(f"Error en preview: {e}")

def generate_images_for_letters(letters, variations_per_letter, visual_config, master_config, use_random_params, fixed_params):
    """Genera imágenes para las letras especificadas"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_operations = len(letters) * variations_per_letter
    current_operation = 0
    
    # Asegurar que existe la estructura de imágenes generadas
    if 'generated_images' not in master_config['visual_dataset']:
        master_config['visual_dataset']['generated_images'] = {}
    
    for letter_idx, letter in enumerate(letters):
        status_text.text(f"Generando imágenes para la letra '{letter.upper()}'...")
        
        # Inicializar lista para esta letra si no existe
        if letter not in master_config['visual_dataset']['generated_images']:
            master_config['visual_dataset']['generated_images'][letter] = []
        
        letter_images = []
        
        for variation in range(variations_per_letter):
            try:
                if use_random_params:
                    # Parámetros aleatorios
                    font_size = random.choice(visual_config.get('font_sizes', [32]))
                    rotation = random.uniform(-visual_config.get('rotation_range', 15), visual_config.get('rotation_range', 15))
                    noise_level = random.choice(visual_config.get('noise_levels', [0.0]))
                    font = random.choice(visual_config.get('fonts', ['arial.ttf']))
                else:
                    # Parámetros fijos
                    font_size = fixed_params['font_size']
                    rotation = fixed_params['rotation']
                    noise_level = fixed_params['noise_level']
                    font = fixed_params['font']
                
                # Generar imagen
                img = generate_letter_image(letter, font_size, rotation, noise_level, font)
                
                if img:
                    # Convertir a base64
                    img_base64 = image_to_base64(img)
                    
                    if img_base64:
                        # Crear entrada de imagen
                        image_entry = {
                            'image_base64': img_base64,
                            'params': {
                                'font_size': font_size,
                                'rotation': round(rotation, 2),
                                'noise_level': noise_level,
                                'font': font
                            },
                            'letter': letter.upper(),
                            'timestamp': datetime.now().isoformat(),
                            'image_size': visual_config.get('image_size', [64, 64])
                        }
                        
                        letter_images.append(image_entry)
                
                current_operation += 1
                progress_bar.progress(current_operation / total_operations)
                
            except Exception as e:
                st.warning(f"Error generando variación {variation + 1} para letra '{letter}': {e}")
                current_operation += 1
                progress_bar.progress(current_operation / total_operations)
        
        # Agregar nuevas imágenes a las existentes
        master_config['visual_dataset']['generated_images'][letter].extend(letter_images)
        
        status_text.text(f"✅ Completada letra '{letter.upper()}' - {len(letter_images)} imágenes generadas")
    
    # Guardar configuración actualizada
    if save_master_config(master_config):
        st.success(f"🎉 ¡Generación completada! Se generaron {sum(len(master_config['visual_dataset']['generated_images'].get(l, [])) for l in letters)} imágenes para {len(letters)} letras")
        progress_bar.progress(1.0)
        status_text.text("✅ Todas las imágenes guardadas exitosamente")
    else:
        st.error("❌ Error guardando las imágenes generadas")

def management_tab(visual_config, master_config):
    """Tab de gestión del dataset visual"""
    st.header("📁 Gestión del Dataset Visual")
    
    management_col1, management_col2 = st.columns(2)
    
    with management_col1:
        st.subheader("🗂️ Limpieza y Mantenimiento")
        
        # Mostrar estadísticas actuales
        generated_images = master_config['visual_dataset'].get('generated_images', {})
        total_images = sum(len(samples) for samples in generated_images.values())
        letters_with_images = len([l for l in generated_images.keys() if len(generated_images[l]) > 0])
        
        st.markdown(f"""
        **📊 Estado Actual:**
        - 🔤 Letras con imágenes: {letters_with_images}
        - 🖼️ Total de imágenes: {total_images}
        - 💾 Tamaño estimado: ~{total_images * 2} KB
        """)
        
        # Opciones de limpieza
        st.markdown("**🧹 Opciones de Limpieza:**")
        
        if st.button("🗑️ Limpiar Todas las Imágenes", key="clear_all_images"):
            if st.session_state.get('confirm_clear_all', False):
                master_config['visual_dataset']['generated_images'] = {}
                if save_master_config(master_config):
                    st.success("✅ Todas las imágenes han sido eliminadas")
                    st.rerun()
                else:
                    st.error("❌ Error eliminando imágenes")
                st.session_state['confirm_clear_all'] = False
            else:
                st.session_state['confirm_clear_all'] = True
                st.warning("⚠️ Haz clic nuevamente para confirmar la eliminación de todas las imágenes")
        
        # Limpiar por letra específica
        if generated_images:
            letter_to_clear = st.selectbox(
                "Seleccionar letra para limpiar",
                options=list(generated_images.keys()),
                key="letter_to_clear"
            )
            
            if letter_to_clear and st.button(f"🗑️ Limpiar letra '{letter_to_clear.upper()}'"):
                master_config['visual_dataset']['generated_images'][letter_to_clear] = []
                if save_master_config(master_config):
                    st.success(f"✅ Imágenes de la letra '{letter_to_clear.upper()}' eliminadas")
                    st.rerun()
    
    with management_col2:
        st.subheader("📤 Exportar/Importar")
        
        # Exportar configuración
        if total_images > 0:
            st.markdown("**💾 Exportar Dataset:**")
            
            export_format = st.selectbox(
                "Formato de exportación",
                ["Configuración completa (JSON)", "Solo configuración (sin imágenes)", "Estadísticas resumidas"]
            )
            
            if st.button("📤 Exportar Dataset Visual"):
                export_visual_dataset(master_config['visual_dataset'], export_format)
        
        # Importar configuración
        st.markdown("**📥 Importar Dataset:**")
        
        uploaded_file = st.file_uploader(
            "Cargar archivo de configuración visual",
            type=['json'],
            help="Archivo JSON con configuración visual compatible"
        )
        
        if uploaded_file is not None:
            try:
                imported_config = json.load(uploaded_file)
                
                # Validar estructura
                if 'generated_images' in imported_config or 'name' in imported_config:
                    st.success("✅ Archivo válido detectado")
                    
                    # Mostrar preview
                    imported_images = sum(len(samples) for samples in imported_config.get('generated_images', {}).values())
                    st.info(f"📊 El archivo contiene {imported_images} imágenes")
                    
                    if st.button("📥 Importar Configuración Visual"):
                        # Merge con configuración existente
                        if 'generated_images' in imported_config:
                            for letter, images in imported_config['generated_images'].items():
                                if letter not in master_config['visual_dataset']['generated_images']:
                                    master_config['visual_dataset']['generated_images'][letter] = []
                                master_config['visual_dataset']['generated_images'][letter].extend(images)
                        
                        if save_master_config(master_config):
                            st.success("✅ Dataset visual importado exitosamente!")
                            st.rerun()
                        else:
                            st.error("❌ Error importando dataset")
                
                else:
                    st.error("❌ Archivo no válido - falta estructura requerida")
            
            except Exception as e:
                st.error(f"❌ Error procesando archivo: {e}")

def dataset_configuration_tab(visual_config, master_config):
    """Tab de configuración avanzada del dataset"""
    st.header("📋 Configuración Avanzada del Dataset")
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.subheader("🔗 Sincronización con Master Config")
        
        # Mostrar estado de sincronización
        master_vocab = master_config.get('master_config_reference', 'N/A')
        visual_vocab = visual_config.get('vocabulary', 'N/A')
        
        if master_vocab == visual_vocab:
            st.success(f"✅ Sincronizado con: **{master_vocab}**")
        else:
            st.warning(f"⚠️ Desincronizado: Master={master_vocab}, Visual={visual_vocab}")
            
            if st.button("🔄 Forzar Sincronización"):
                update_visual_vocabulary()
                st.success("✅ Sincronización forzada completada")
                st.rerun()
        
        # Información del vocabulario maestro
        if master_vocab in master_config.get('generated_samples', {}):
            vocab_samples = master_config['generated_samples'][master_vocab]
            st.info(f"📚 Vocabulario maestro contiene {len(vocab_samples)} muestras de audio")
            
            # Mostrar palabras del vocabulario
            words = [sample.get('word', '') for sample in vocab_samples[:10]]
            st.write("**Palabras de muestra:**", ', '.join(words))
    
    with config_col2:
        st.subheader("⚙️ Configuración Técnica")
        
        # Configuraciones avanzadas
        batch_size = st.number_input(
            "Tamaño de lote para generación",
            min_value=1,
            max_value=100,
            value=10,
            help="Número de imágenes a procesar simultáneamente"
        )
        
        max_images_per_letter = st.number_input(
            "Máximo de imágenes por letra",
            min_value=10,
            max_value=1000,
            value=100,
            help="Límite máximo de imágenes almacenadas por letra"
        )
        
        auto_cleanup = st.checkbox(
            "Auto-limpieza al exceder límite",
            value=False,
            help="Eliminar automáticamente imágenes más antiguas al superar el límite"
        )
        
        # Configuración de calidad
        st.markdown("**🎨 Configuración de Calidad:**")
        
        image_quality = st.slider(
            "Calidad de compresión (%)",
            min_value=50,
            max_value=100,
            value=90
        )
        
        use_antialiasing = st.checkbox(
            "Usar anti-aliasing",
            value=True,
            help="Mejora la calidad visual de las letras"
        )
        
        # Guardar configuraciones avanzadas
        if st.button("💾 Guardar Configuración Avanzada"):
            master_config['visual_dataset']['advanced_config'] = {
                'batch_size': batch_size,
                'max_images_per_letter': max_images_per_letter,
                'auto_cleanup': auto_cleanup,
                'image_quality': image_quality,
                'use_antialiasing': use_antialiasing
            }
            
            if save_master_config(master_config):
                st.success("✅ Configuración avanzada guardada!")
            else:
                st.error("❌ Error guardando configuración")
    
    # Información del sistema
    st.markdown("---")
    st.subheader("🔍 Información del Sistema")
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.markdown("**📁 Archivos:**")
        st.write(f"- Master config: ✅ Existe")
        st.write(f"- Visual config: 🔗 Integrado")
        
    with info_col2:
        st.markdown("**📊 Estadísticas:**")
        generated_images = master_config['visual_dataset'].get('generated_images', {})
        st.write(f"- Letras: {len(generated_images)}")
        st.write(f"- Imágenes: {sum(len(samples) for samples in generated_images.values())}")
        
    with info_col3:
        st.markdown("**🕐 Timestamps:**")
        created = visual_config.get('created', 'N/A')
        if created != 'N/A':
            try:
                fecha = datetime.fromisoformat(created)
                st.write(f"- Creado: {fecha.strftime('%d/%m/%Y')}")
            except:
                st.write(f"- Creado: {created[:10]}")
        st.write(f"- Versión: {visual_config.get('version', 'N/A')}")

def export_visual_dataset(visual_config, export_format):
    """Exporta el dataset visual en el formato especificado"""
    
    if export_format == "Solo configuración (sin imágenes)":
        # Exportar solo configuración sin las imágenes
        config_export = visual_config.copy()
        config_export['generated_images'] = {}
        
        config_json = json.dumps(config_export, indent=2, ensure_ascii=False)
        st.download_button(
            label="💾 Descargar Configuración Visual",
            data=config_json,
            file_name=f"visual_dataset_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    elif export_format == "Estadísticas resumidas":
        # Exportar solo estadísticas
        generated_images = visual_config.get('generated_images', {})
        stats = {
            'name': visual_config.get('name', 'Visual Dataset'),
            'letters_count': len(generated_images),
            'total_images': sum(len(samples) for samples in generated_images.values()),
            'letters_with_images': [letter for letter, images in generated_images.items() if len(images) > 0],
            'export_timestamp': datetime.now().isoformat(),
            'version': visual_config.get('version', '1.0')
        }
        
        stats_json = json.dumps(stats, indent=2, ensure_ascii=False)
        st.download_button(
            label="📊 Descargar Estadísticas",
            data=stats_json,
            file_name=f"visual_dataset_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    else:  # Configuración completa (JSON)
        # Exportar configuración completa con todas las imágenes
        config_json = json.dumps(visual_config, indent=2, ensure_ascii=False, default=str)
        st.download_button(
            label="💾 Descargar Dataset Completo",
            data=config_json,
            file_name=f"visual_dataset_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()