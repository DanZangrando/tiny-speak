"""
🖼️ Visual Dataset Manager - Gestión de Datasets Visuales
Página para generar, modificar y analizar datasets de imágenes de letras
"""

import streamlit as st
import torch
import numpy as np
import json
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import io
import base64

# Importar módulos
from utils import LETTERS, encontrar_device

# Configurar página
st.set_page_config(
    page_title="Visual Dataset Manager",
    page_icon="🖼️",
    layout="wide"
)

def main():
    """Función principal de la página"""
    st.title("🖼️ Visual Dataset Manager")
    st.markdown("**Gestiona, genera y analiza datasets visuales de letras manuscritas para entrenar TinyRecognizer**")
    
    # Inicializar session state
    if 'visual_dataset_config' not in st.session_state:
        st.session_state.visual_dataset_config = load_visual_dataset_config()
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 Configuración", 
        "🖼️ Generación de Imágenes", 
        "📊 Análisis del Dataset",
        "💾 Exportar/Importar"
    ])
    
    with tab1:
        visual_dataset_configuration()
    
    with tab2:
        image_generation()
    
    with tab3:
        visual_dataset_analysis()
    
    with tab4:
        export_import_visual_dataset()

def load_visual_dataset_config():
    """Carga la configuración del dataset visual"""
    config_path = Path("visual_dataset_config.json")
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return {
            "name": "TinySpeak_Visual_Dataset",
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "letters": list(LETTERS),
            "image_params": {
                "size": [64, 64],
                "fonts": ["arial", "times", "courier", "helvetica"],
                "font_sizes": [24, 32, 40, 48],
                "rotations": [-15, -10, -5, 0, 5, 10, 15],
                "noise_levels": [0, 0.1, 0.2, 0.3],
                "blur_levels": [0, 1, 2],
                "thickness_variations": [1, 2, 3]
            },
            "variations_per_letter": 20,
            "total_images": 0,
            "generated_images": {}
        }

def save_visual_dataset_config(config):
    """Guarda la configuración del dataset visual de manera segura"""
    config_path = Path("visual_dataset_config.json")
    config_temp_path = Path("visual_dataset_config_temp.json")
    
    try:
        config["modified"] = datetime.now().isoformat()
        
        # Convertir imágenes a base64 para serialización JSON
        config_to_save = config.copy()
        if 'generated_images' in config_to_save:
            for letter in config_to_save['generated_images']:
                for i, sample in enumerate(config_to_save['generated_images'][letter]):
                    if 'image' in sample and hasattr(sample['image'], 'save'):
                        # Convertir PIL Image a base64
                        buffer = io.BytesIO()
                        sample['image'].save(buffer, format='PNG')
                        img_str = base64.b64encode(buffer.getvalue()).decode()
                        config_to_save['generated_images'][letter][i] = sample.copy()
                        config_to_save['generated_images'][letter][i]['image'] = img_str
        
        # Guardar en archivo temporal primero
        with open(config_temp_path, 'w', encoding='utf-8') as f:
            json.dump(config_to_save, f, indent=2, ensure_ascii=False)
        
        # Si la escritura fue exitosa, mover el archivo temporal al final
        import shutil
        shutil.move(str(config_temp_path), str(config_path))
        
        st.session_state.visual_dataset_config = config
        
    except Exception as e:
        # Limpiar archivo temporal si hay error
        if config_temp_path.exists():
            config_temp_path.unlink()
        st.error(f"Error guardando configuración visual: {e}")
        raise e

def visual_dataset_configuration():
    """Configuración del dataset visual"""
    st.header("📁 Configuración del Dataset Visual")
    
    config = st.session_state.visual_dataset_config.copy()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔧 Parámetros Generales")
        
        config["name"] = st.text_input(
            "Nombre del Dataset Visual", 
            value=config["name"]
        )
        
        config["version"] = st.text_input(
            "Versión", 
            value=config["version"]
        )
        
        config["variations_per_letter"] = st.slider(
            "Variaciones por letra",
            min_value=1,
            max_value=50,
            value=config["variations_per_letter"],
            help="Número de variaciones de imagen a generar por cada letra"
        )
        
        # Gestión de letras
        st.subheader("🔤 Gestión de Letras")
        
        available_letters = [l for l in LETTERS if l not in config["letters"]]
        current_letters = config["letters"]
        
        if available_letters:
            selected_letters = st.multiselect(
                "Agregar letras",
                options=available_letters,
                help="Selecciona letras adicionales para incluir en el dataset"
            )
            
            if st.button("➕ Agregar letras seleccionadas"):
                config["letters"].extend(selected_letters)
                st.rerun()
        
        # Mostrar letras actuales
        if current_letters:
            st.write(f"**Letras en el dataset ({len(current_letters)}):**")
            
            # Crear una grid de letras
            cols_per_row = 8
            for i in range(0, len(current_letters), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(current_letters):
                        letter = current_letters[idx]
                        col.write(f"**{letter}**")
                        if col.button(f"🗑️", key=f"delete_letter_{idx}"):
                            config["letters"].remove(letter)
                            st.rerun()
    
    with col2:
        st.subheader("🖼️ Parámetros de Imagen")
        
        # Tamaño de imagen
        col_w, col_h = st.columns(2)
        with col_w:
            width = st.number_input("Ancho", min_value=32, max_value=256, value=config["image_params"]["size"][0])
        with col_h:
            height = st.number_input("Alto", min_value=32, max_value=256, value=config["image_params"]["size"][1])
        config["image_params"]["size"] = [width, height]
        
        # Fuentes disponibles
        st.write("**Fuentes disponibles:**")
        fonts = st.multiselect(
            "Seleccionar fuentes",
            options=["arial", "times", "courier", "helvetica", "georgia", "verdana"],
            default=config["image_params"]["fonts"]
        )
        config["image_params"]["fonts"] = fonts
        
        # Tamaños de fuente
        font_size_min, font_size_max = st.slider(
            "Rango de tamaños de fuente",
            min_value=12,
            max_value=72,
            value=[min(config["image_params"]["font_sizes"]), max(config["image_params"]["font_sizes"])]
        )
        config["image_params"]["font_sizes"] = list(range(font_size_min, font_size_max + 4, 4))
        
        # Rotaciones
        rotation_range = st.slider(
            "Rango de rotación (grados)",
            min_value=0,
            max_value=45,
            value=15
        )
        config["image_params"]["rotations"] = list(range(-rotation_range, rotation_range + 1, 5))
        
        # Niveles de ruido
        noise_max = st.slider(
            "Nivel máximo de ruido",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1
        )
        config["image_params"]["noise_levels"] = [i * 0.1 for i in range(int(noise_max * 10) + 1)]
        
        # Estadísticas
        st.subheader("📊 Estadísticas")
        total_planned = len(config["letters"]) * config["variations_per_letter"]
        st.metric("Total de imágenes planificadas", total_planned)
        st.metric("Letras en dataset", len(config["letters"]))
        st.metric("Imágenes generadas", config["total_images"])
    
    # Botón para guardar
    if st.button("💾 Guardar Configuración Visual", type="primary"):
        save_visual_dataset_config(config)
        st.success("✅ Configuración visual guardada exitosamente!")

def image_generation():
    """Generación de imágenes"""
    st.header("🖼️ Generación de Imágenes")
    
    config = st.session_state.visual_dataset_config
    
    if not config["letters"]:
        st.warning("⚠️ No hay letras configuradas. Ve a la pestaña de Configuración para agregar letras.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎛️ Controles de Generación")
        
        # Selector de letras
        selected_letters = st.multiselect(
            "Letras a generar",
            options=config["letters"],
            default=config["letters"][:10] if len(config["letters"]) > 10 else config["letters"],
            help="Selecciona las letras para las que quieres generar imágenes"
        )
        
        # Número de variaciones
        num_variations = st.slider(
            "Variaciones a generar",
            min_value=1,
            max_value=config["variations_per_letter"],
            value=min(5, config["variations_per_letter"])
        )
        
        # Vista previa de parámetros
        st.write("**Vista previa de parámetros:**")
        st.json({
            "Tamaño": f"{config['image_params']['size'][0]}x{config['image_params']['size'][1]}",
            "Fuentes": len(config['image_params']['fonts']),
            "Tamaños fuente": f"{min(config['image_params']['font_sizes'])}-{max(config['image_params']['font_sizes'])}",
            "Rotaciones": f"±{max(abs(min(config['image_params']['rotations'])), abs(max(config['image_params']['rotations'])))}°"
        })
        
        # Vista previa
        st.subheader("👁️ Vista Previa")
        if st.button("🔍 Generar Vista Previa"):
            if selected_letters:
                preview_letter = selected_letters[0]
                preview_image = generate_single_image(
                    preview_letter,
                    config["image_params"],
                    preview=True
                )
                if preview_image:
                    st.image(preview_image, caption=f"Vista previa: {preview_letter}", width=150)
        
        if st.button("🖼️ Generar Imágenes", type="primary"):
            generate_image_samples(selected_letters, num_variations, config)
    
    with col2:
        st.subheader("🖼️ Imágenes Generadas")
        
        # Mostrar imágenes generadas recientes
        if 'generated_images' in config and config['generated_images']:
            st.write("**Últimas imágenes generadas:**")
            
            # Grid de imágenes
            cols_per_row = 4
            all_samples = []
            
            # Recopilar todas las muestras
            for letter, samples in list(config['generated_images'].items())[-8:]:  # Últimas 8 letras
                for sample in samples[-2:]:  # Últimas 2 muestras por letra
                    all_samples.append((letter, sample))
            
            # Mostrar en grid
            for i in range(0, len(all_samples), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(all_samples):
                        letter, sample = all_samples[idx]
                        
                        with col:
                            if 'image' in sample:
                                # Si es string base64, convertir de vuelta a imagen
                                if isinstance(sample['image'], str):
                                    try:
                                        img_data = base64.b64decode(sample['image'])
                                        image = Image.open(io.BytesIO(img_data))
                                    except:
                                        image = None
                                else:
                                    image = sample['image']
                                
                                if image:
                                    st.image(image, caption=f"{letter}", width=100)
                                    st.caption(f"Fuente: {sample['params'].get('font', 'N/A')}")
        else:
            st.info("No hay imágenes generadas aún. Usa los controles de la izquierda para generar imágenes.")

def generate_single_image(letter, image_params, preview=False):
    """Genera una sola imagen para vista previa"""
    try:
        # Seleccionar parámetros aleatorios
        font_name = np.random.choice(image_params["fonts"])
        font_size = np.random.choice(image_params["font_sizes"])
        rotation = np.random.choice(image_params["rotations"])
        noise_level = np.random.choice(image_params["noise_levels"])
        
        # Crear imagen
        width, height = image_params["size"]
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        # Intentar cargar fuente (usar fuente por defecto si falla)
        try:
            font = ImageFont.truetype(f"{font_name}.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Calcular posición centrada
        bbox = draw.textbbox((0, 0), letter, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        # Dibujar texto
        draw.text((x, y), letter, font=font, fill='black')
        
        # Aplicar rotación si es necesario
        if rotation != 0:
            image = image.rotate(rotation, expand=False, fillcolor='white')
        
        # Aplicar ruido si es necesario
        if noise_level > 0:
            img_array = np.array(image)
            noise = np.random.normal(0, noise_level * 255, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(img_array)
        
        return image
        
    except Exception as e:
        st.error(f"Error generando imagen: {e}")
        return None

def generate_image_samples(letters, num_variations, config):
    """Genera muestras de imágenes"""
    
    if 'generated_images' not in config:
        config['generated_images'] = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_images = len(letters) * num_variations
    current_image = 0
    
    for letter in letters:
        if letter not in config['generated_images']:
            config['generated_images'][letter] = []
        
        status_text.text(f"Generando variaciones para: {letter}")
        
        for i in range(num_variations):
            # Generar parámetros aleatorios
            font_name = np.random.choice(config["image_params"]["fonts"])
            font_size = np.random.choice(config["image_params"]["font_sizes"])
            rotation = np.random.choice(config["image_params"]["rotations"])
            noise_level = np.random.choice(config["image_params"]["noise_levels"])
            
            # Generar imagen
            try:
                image = generate_single_image(letter, config["image_params"])
                
                if image is not None:
                    sample = {
                        'id': f"{letter}_{len(config['generated_images'][letter])}",
                        'letter': letter,
                        'params': {
                            'font': font_name,
                            'font_size': font_size,
                            'rotation': rotation,
                            'noise_level': noise_level
                        },
                        'image': image,
                        'generated_at': datetime.now().isoformat()
                    }
                    
                    config['generated_images'][letter].append(sample)
                    config['total_images'] += 1
            
            except Exception as e:
                st.error(f"Error generando imagen para '{letter}': {e}")
            
            current_image += 1
            progress_bar.progress(current_image / total_images)
    
    progress_bar.empty()
    status_text.empty()
    
    # Guardar configuración actualizada
    save_visual_dataset_config(config)
    st.success(f"✅ Generadas {current_image} imágenes exitosamente!")

def visual_dataset_analysis():
    """Análisis del dataset visual"""
    st.header("📊 Análisis del Dataset Visual")
    
    config = st.session_state.visual_dataset_config
    
    if not config.get('generated_images'):
        st.info("No hay imágenes generadas para analizar. Ve a la pestaña de Generación para crear imágenes.")
        return
    
    # Estadísticas generales
    col1, col2, col3, col4 = st.columns(4)
    
    total_letters = len(config['generated_images'])
    total_images = sum(len(samples) for samples in config['generated_images'].values())
    avg_per_letter = total_images / total_letters if total_letters > 0 else 0
    
    col1.metric("Total Letras", total_letters)
    col2.metric("Total Imágenes", total_images)
    col3.metric("Promedio por Letra", f"{avg_per_letter:.1f}")
    col4.metric("Tamaño Dataset", f"{total_images * 0.1:.1f} MB")  # Estimación
    
    # Gráficos de distribución
    st.subheader("📈 Distribución de Imágenes")
    
    # Preparar datos para visualización
    letters = []
    counts = []
    font_sizes = []
    rotations = []
    noise_levels = []
    
    for letter, samples in config['generated_images'].items():
        letters.append(letter)
        counts.append(len(samples))
        
        for sample in samples:
            font_sizes.append(sample['params']['font_size'])
            rotations.append(sample['params']['rotation'])
            noise_levels.append(sample['params']['noise_level'])
    
    # Gráfico de barras - Imágenes por letra
    fig_counts = px.bar(
        x=letters, 
        y=counts,
        title="Número de Imágenes por Letra",
        labels={'x': 'Letra', 'y': 'Número de Imágenes'},
        color=counts,
        color_continuous_scale='Viridis'
    )
    fig_counts.update_layout(height=400)
    st.plotly_chart(fig_counts, use_container_width=True)
    
    # Histogramas de parámetros
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig_font = px.histogram(
            x=font_sizes,
            title="Distribución de Tamaños de Fuente",
            labels={'x': 'Tamaño de Fuente', 'y': 'Frecuencia'},
            nbins=20
        )
        fig_font.update_layout(height=300)
        st.plotly_chart(fig_font, use_container_width=True)
    
    with col2:
        fig_rotation = px.histogram(
            x=rotations,
            title="Distribución de Rotaciones",
            labels={'x': 'Rotación (grados)', 'y': 'Frecuencia'},
            nbins=20
        )
        fig_rotation.update_layout(height=300)
        st.plotly_chart(fig_rotation, use_container_width=True)
    
    with col3:
        fig_noise = px.histogram(
            x=noise_levels,
            title="Distribución de Niveles de Ruido",
            labels={'x': 'Nivel de Ruido', 'y': 'Frecuencia'},
            nbins=20
        )
        fig_noise.update_layout(height=300)
        st.plotly_chart(fig_noise, use_container_width=True)
    
    # Galería de muestras
    st.subheader("🖼️ Galería de Muestras")
    
    # Selector de letra para mostrar galería
    selected_letter = st.selectbox(
        "Seleccionar letra para ver galería",
        options=list(config['generated_images'].keys())
    )
    
    if selected_letter and selected_letter in config['generated_images']:
        samples = config['generated_images'][selected_letter]
        
        # Mostrar hasta 12 muestras en grid
        cols_per_row = 6
        samples_to_show = samples[:12]
        
        for i in range(0, len(samples_to_show), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(samples_to_show):
                    sample = samples_to_show[idx]
                    
                    with col:
                        # Mostrar imagen
                        if 'image' in sample:
                            image = sample['image']
                            if isinstance(image, str):
                                try:
                                    img_data = base64.b64decode(image)
                                    image = Image.open(io.BytesIO(img_data))
                                except:
                                    continue
                            
                            st.image(image, width=80)
                            st.caption(f"Font: {sample['params'].get('font_size', 'N/A')}")

def export_import_visual_dataset():
    """Exportar e importar dataset visual"""
    st.header("💾 Exportar/Importar Dataset Visual")
    
    config = st.session_state.visual_dataset_config
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Exportar Dataset Visual")
        
        if config.get('generated_images'):
            # Información del export
            total_images = sum(len(samples) for samples in config['generated_images'].values())
            st.info(f"Dataset visual listo para exportar: {len(config['generated_images'])} letras, {total_images} imágenes")
            
            # Opciones de export
            export_format = st.selectbox(
                "Formato de exportación visual",
                ["JSON + Imágenes", "JSON solamente", "Configuración solamente"]
            )
            
            if st.button("📤 Exportar Dataset Visual", type="primary"):
                export_visual_dataset(config, export_format)
        else:
            st.warning("No hay imágenes generadas para exportar.")
    
    with col2:
        st.subheader("📥 Importar Dataset Visual")
        
        uploaded_file = st.file_uploader(
            "Seleccionar archivo de configuración visual",
            type=['json'],
            help="Sube un archivo JSON con la configuración del dataset visual"
        )
        
        if uploaded_file is not None:
            try:
                imported_config = json.load(uploaded_file)
                
                # Validar estructura básica
                if 'name' in imported_config and 'letters' in imported_config:
                    st.success("✅ Archivo válido detectado")
                    
                    # Mostrar preview
                    st.write("**Preview del dataset visual:**")
                    st.json({
                        "Nombre": imported_config.get('name', 'Sin nombre'),
                        "Letras": len(imported_config.get('letters', [])),
                        "Imágenes": imported_config.get('total_images', 0)
                    })
                    
                    if st.button("📥 Importar Configuración Visual"):
                        st.session_state.visual_dataset_config = imported_config
                        save_visual_dataset_config(imported_config)
                        st.success("✅ Dataset visual importado exitosamente!")
                        st.rerun()
                else:
                    st.error("❌ Archivo no válido. Falta estructura requerida.")
            
            except Exception as e:
                st.error(f"❌ Error leyendo archivo: {e}")

def export_visual_dataset(config, export_format):
    """Exporta el dataset visual en el formato especificado"""
    
    if export_format == "Configuración solamente":
        # Solo exportar configuración
        config_export = config.copy()
        if 'generated_images' in config_export:
            # Remover imágenes para hacer el archivo más pequeño
            config_export['generated_images'] = {}
        
        config_json = json.dumps(config_export, indent=2, ensure_ascii=False)
        st.download_button(
            label="💾 Descargar Configuración Visual",
            data=config_json,
            file_name=f"{config['name']}_visual_config.json",
            mime="application/json"
        )
    
    elif export_format == "JSON solamente":
        # Exportar configuración completa con metadatos
        config_json = json.dumps(config, indent=2, ensure_ascii=False, default=str)
        st.download_button(
            label="💾 Descargar Dataset Visual (JSON)",
            data=config_json,
            file_name=f"{config['name']}_visual_dataset.json",
            mime="application/json"
        )
    
    else:  # JSON + Imágenes
        st.info("Para exportar con archivos de imagen, usa la funcionalidad de descarga individual de cada muestra.")

if __name__ == "__main__":
    main()