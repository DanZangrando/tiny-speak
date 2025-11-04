import streamlit as st
import json
import os
import base64
import io
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import numpy as np
from collections import Counter
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from components.modern_sidebar import display_modern_sidebar

st.set_page_config(
    page_title="📊 Visual Analytics - TinySpeak",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Aplicar estilos CSS modernos
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea, #764ba2);
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

.analytics-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    text-align: center;
}

.status-success {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
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
    # Usar ruta absoluta basada en la ubicación del archivo actual
    current_dir = Path(__file__).parent.parent
    config_file = current_dir / "master_dataset_config.json"
    
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except Exception as e:
            st.error(f"Error cargando configuración maestra: {e}")
            return None
    else:
        st.error(f"❌ No se encontró el archivo {config_file}")
        return None

def create_dataframe_from_images(visual_config):
    """Crea un DataFrame detallado a partir de las imágenes generadas"""
    
    generated_images = visual_config.get('generated_images', {})
    
    if not generated_images:
        return None
    
    data = []
    
    for letter, images in generated_images.items():
        for idx, image_data in enumerate(images):
            params = image_data.get('params', {})
            
            # Categorizar parámetros
            font_size = params.get('font_size', 32)
            rotation = params.get('rotation', 0)
            noise_level = params.get('noise_level', 0.0)
            
            # Crear categorías
            if font_size <= 24:
                size_cat = 'Pequeño'
            elif font_size <= 32:
                size_cat = 'Mediano'
            else:
                size_cat = 'Grande'
            
            if abs(rotation) <= 5:
                rot_cat = 'Sin rotación'
            elif abs(rotation) <= 15:
                rot_cat = 'Rotación ligera'
            else:
                rot_cat = 'Rotación fuerte'
            
            if noise_level == 0:
                noise_cat = 'Sin ruido'
            elif noise_level <= 0.1:
                noise_cat = 'Ruido bajo'
            elif noise_level <= 0.2:
                noise_cat = 'Ruido medio'
            else:
                noise_cat = 'Ruido alto'
            
            entry = {
                'Letra': letter.upper(),
                'Imagen_ID': f"{letter}_{idx+1}",
                'Font_Size': font_size,
                'Rotacion': rotation,
                'Noise_Level': noise_level,
                'Font': params.get('font', 'arial.ttf'),
                'Categoria_Tamaño': size_cat,
                'Categoria_Rotacion': rot_cat,
                'Categoria_Ruido': noise_cat,
                'Timestamp': image_data.get('timestamp', ''),
                'Tamaño_Imagen': str(image_data.get('image_size', [64, 64]))
            }
            
            data.append(entry)
    
    return pd.DataFrame(data)

def main():
    # Mostrar sidebar moderna
    display_modern_sidebar()
    
    # Header moderno
    st.markdown('<h1 class="main-header">📊 Visual Analytics</h1>', unsafe_allow_html=True)
    
    # Sidebar con opciones de análisis
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h3>📈 Opciones de Análisis</h3>
        </div>
        """, unsafe_allow_html=True)
        
        analysis_type = st.selectbox(
            "Tipo de análisis",
            ["📊 Resumen General", "🔤 Análisis por Letra", "📈 Distribuciones", "🖼️ Galería Interactiva"],
            key="analysis_type"
        )
        
        # Opciones de filtrado
        st.markdown("**🔍 Filtros:**")
        
        # Estos se llenarán dinámicamente según los datos
    
    # Cargar configuración
    config = load_master_config()
    
    if config is None:
        st.error("❌ No se puede cargar la configuración. Verifica que existe el archivo master_dataset_config.json")
        return
    
    visual_config = config.get('visual_dataset', {})
    
    if not visual_config.get('generated_images'):
        st.warning("⚠️ No hay imágenes generadas para analizar. Usa el Visual Dataset Manager para generar imágenes primero.")
        return
    
    # Crear DataFrame para análisis
    df_detailed = create_dataframe_from_images(visual_config)
    
    if df_detailed is None or df_detailed.empty:
        st.warning("⚠️ No se pudieron procesar las imágenes para análisis.")
        return
    
    # Mostrar análisis según la selección
    if analysis_type == "📊 Resumen General":
        show_general_summary(visual_config, df_detailed)
    elif analysis_type == "🔤 Análisis por Letra":
        show_letter_analysis(visual_config, df_detailed)
    elif analysis_type == "📈 Distribuciones":
        show_distributions_analysis(visual_config, df_detailed)
    elif analysis_type == "🖼️ Galería Interactiva":
        show_interactive_gallery(visual_config, df_detailed)

def show_general_summary(visual_config, df_detailed):
    """Muestra resumen general del dataset visual"""
    st.header("📊 Resumen General del Dataset Visual")
    
    # Métricas principales
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        total_images = len(df_detailed)
        st.markdown(f"""
        <div class="metric-card">
            <h3>🖼️ Total Imágenes</h3>
            <h2>{total_images:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col2:
        unique_letters = df_detailed['Letra'].nunique()
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔤 Letras Únicas</h3>
            <h2>{unique_letters}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col3:
        avg_images_per_letter = total_images / unique_letters if unique_letters > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Promedio por Letra</h3>
            <h2>{avg_images_per_letter:.1f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col4:
        vocabulary_name = visual_config.get('vocabulary', 'N/A')
        st.markdown(f"""
        <div class="metric-card">
            <h3>📚 Vocabulario</h3>
            <h2>{vocabulary_name}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Gráficos de resumen
    st.markdown("---")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Distribución de imágenes por letra
        letters_count = df_detailed['Letra'].value_counts().sort_index()
        
        fig_letters = px.bar(
            x=letters_count.index,
            y=letters_count.values,
            title="🔤 Distribución de Imágenes por Letra",
            labels={'x': 'Letras', 'y': 'Cantidad'},
            color=letters_count.values,
            color_continuous_scale='viridis'
        )
        fig_letters.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_letters, width='stretch')
    
    with chart_col2:
        # Distribución de categorías de tamaño
        size_count = df_detailed['Categoria_Tamaño'].value_counts()
        
        fig_size = px.pie(
            values=size_count.values,
            names=size_count.index,
            title="📏 Distribución por Tamaño de Fuente",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_size.update_layout(height=400)
        st.plotly_chart(fig_size, width='stretch')
    
    # Estadísticas detalladas
    st.markdown("---")
    st.header("📈 Estadísticas Detalladas")
    
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    
    with stats_col1:
        st.subheader("📏 Tamaños de Fuente")
        font_stats = df_detailed['Font_Size'].describe()
        st.dataframe(font_stats, width='stretch')
        
        st.subheader("🎯 Fuentes Más Usadas")
        font_usage = df_detailed['Font'].value_counts().head(5)
        st.dataframe(font_usage, width='stretch')
    
    with stats_col2:
        st.subheader("🔄 Rotaciones")
        rotation_stats = df_detailed['Rotacion'].describe()
        st.dataframe(rotation_stats, width='stretch')
        
        st.subheader("🌫️ Niveles de Ruido")
        noise_stats = df_detailed['Noise_Level'].describe()
        st.dataframe(noise_stats, width='stretch')
    
    with stats_col3:
        st.subheader("🕐 Información Temporal")
        
        # Parsear timestamps y mostrar estadísticas temporales
        try:
            df_temp = df_detailed.copy()
            df_temp['Fecha'] = pd.to_datetime(df_detailed['Timestamp'], errors='coerce')
            df_temp = df_temp.dropna(subset=['Fecha'])
            
            if not df_temp.empty:
                fecha_min = df_temp['Fecha'].min()
                fecha_max = df_temp['Fecha'].max()
                
                st.write(f"**Primera imagen:** {fecha_min.strftime('%d/%m/%Y %H:%M')}")
                st.write(f"**Última imagen:** {fecha_max.strftime('%d/%m/%Y %H:%M')}")
                st.write(f"**Período total:** {(fecha_max - fecha_min).days} días")
                
                # Gráfico de imágenes por día
                daily_counts = df_temp.groupby(df_temp['Fecha'].dt.date).size()
                
                if len(daily_counts) > 1:
                    fig_timeline = px.line(
                        x=daily_counts.index,
                        y=daily_counts.values,
                        title="📅 Imágenes Generadas por Día"
                    )
                    st.plotly_chart(fig_timeline, width='stretch')
            else:
                st.write("📅 No hay información temporal válida")
        
        except Exception as e:
            st.write(f"❌ Error procesando timestamps: {e}")
    
    # Matriz de correlación (para parámetros numéricos)
    st.markdown("---")
    st.header("🔗 Correlaciones entre Parámetros")
    
    numeric_cols = ['Font_Size', 'Rotacion', 'Noise_Level']
    correlation_matrix = df_detailed[numeric_cols].corr()
    
    fig_corr = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        title="🔗 Matriz de Correlación de Parámetros",
        color_continuous_scale='RdBu'
    )
    st.plotly_chart(fig_corr, width='stretch')

def show_letter_analysis(visual_config, df_detailed):
    """Análisis detallado por letra específica"""
    st.header("🔤 Análisis por Letra Específica")
    
    # Selector de letra
    available_letters = sorted(df_detailed['Letra'].unique())
    
    analysis_col1, analysis_col2 = st.columns([1, 3])
    
    with analysis_col1:
        selected_letter = st.selectbox(
            "Seleccionar letra para análisis",
            available_letters,
            key="selected_letter_analysis"
        )
        
        # Filtros adicionales
        st.subheader("🔍 Filtros")
        
        size_filter = st.multiselect(
            "Tamaño de fuente",
            df_detailed['Categoria_Tamaño'].unique(),
            default=df_detailed['Categoria_Tamaño'].unique(),
            key="size_filter"
        )
        
        rotation_filter = st.multiselect(
            "Rotación",
            df_detailed['Categoria_Rotacion'].unique(),
            default=df_detailed['Categoria_Rotacion'].unique(),
            key="rotation_filter"
        )
        
        noise_filter = st.multiselect(
            "Ruido",
            df_detailed['Categoria_Ruido'].unique(),
            default=df_detailed['Categoria_Ruido'].unique(),
            key="noise_filter"
        )
    
    with analysis_col2:
        if selected_letter:
            # Filtrar datos para la letra seleccionada
            letter_data = df_detailed[
                (df_detailed['Letra'] == selected_letter) &
                (df_detailed['Categoria_Tamaño'].isin(size_filter)) &
                (df_detailed['Categoria_Rotacion'].isin(rotation_filter)) &
                (df_detailed['Categoria_Ruido'].isin(noise_filter))
            ]
            
            if letter_data.empty:
                st.warning("⚠️ No hay datos que coincidan con los filtros seleccionados.")
                return
            
            # Métricas de la letra
            st.subheader(f"📊 Estadísticas de la letra '{selected_letter}'")
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("🖼️ Total de imágenes", len(letter_data))
            
            with metric_col2:
                avg_font_size = letter_data['Font_Size'].mean()
                st.metric("📏 Tamaño promedio", f"{avg_font_size:.1f}")
            
            with metric_col3:
                avg_noise = letter_data['Noise_Level'].mean()
                st.metric("🌫️ Ruido promedio", f"{avg_noise:.3f}")
            
            # Gráficos específicos de la letra
            st.markdown("---")
            
            graph_col1, graph_col2 = st.columns(2)
            
            with graph_col1:
                # Distribución de tamaños de fuente
                fig_font_dist = px.histogram(
                    letter_data,
                    x='Font_Size',
                    title=f"📏 Distribución de Tamaños - Letra '{selected_letter}'",
                    nbins=15,
                    color_discrete_sequence=['#667eea']
                )
                st.plotly_chart(fig_font_dist, width='stretch')
            
            with graph_col2:
                # Scatter plot rotación vs ruido
                fig_scatter = px.scatter(
                    letter_data,
                    x='Rotacion',
                    y='Noise_Level',
                    size='Font_Size',
                    title=f"🔄 Rotación vs Ruido - Letra '{selected_letter}'",
                    color='Font_Size',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_scatter, width='stretch')
            
            # Tabla detallada de parámetros
            st.markdown("---")
            st.subheader("📋 Tabla Detallada de Parámetros")
            
            # Mostrar solo las columnas más relevantes
            display_cols = ['Imagen_ID', 'Font_Size', 'Rotacion', 'Noise_Level', 'Font', 'Timestamp']
            display_data = letter_data[display_cols].copy()
            
            # Formatear timestamp
            if 'Timestamp' in display_data.columns:
                try:
                    display_data['Timestamp'] = pd.to_datetime(display_data['Timestamp']).dt.strftime('%d/%m/%Y %H:%M')
                except:
                    pass  # Mantener formato original si falla
            
            st.dataframe(display_data, width='stretch', height=300)
            
            # Opción de descarga de datos de la letra
            if st.button(f"📥 Descargar datos de la letra '{selected_letter}'"):
                csv_data = letter_data.to_csv(index=False)
                st.download_button(
                    label="💾 Descargar CSV",
                    data=csv_data,
                    file_name=f"letra_{selected_letter}_datos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

def show_distributions_analysis(visual_config, df_detailed):
    """Análisis de distribuciones de parámetros"""
    st.header("📈 Análisis de Distribuciones")
    
    # Controles de visualización
    dist_col1, dist_col2 = st.columns([1, 3])
    
    with dist_col1:
        st.subheader("🎛️ Controles")
        
        # Selector de parámetro a analizar
        parameter = st.selectbox(
            "Parámetro a analizar",
            ['Font_Size', 'Rotacion', 'Noise_Level'],
            format_func=lambda x: {
                'Font_Size': '📏 Tamaño de Fuente',
                'Rotacion': '🔄 Rotación',
                'Noise_Level': '🌫️ Nivel de Ruido'
            }[x]
        )
        
        # Tipo de gráfico
        chart_type = st.selectbox(
            "Tipo de visualización",
            ['histogram', 'box', 'violin', 'strip'],
            format_func=lambda x: {
                'histogram': '📊 Histograma',
                'box': '📦 Box Plot',
                'violin': '🎻 Violin Plot',
                'strip': '🔴 Strip Plot'
            }[x]
        )
        
        # Agrupación
        group_by = st.selectbox(
            "Agrupar por",
            ['Ninguno', 'Letra', 'Categoria_Tamaño', 'Categoria_Rotacion', 'Categoria_Ruido'],
            format_func=lambda x: {
                'Ninguno': '❌ Sin agrupación',
                'Letra': '🔤 Por Letra',
                'Categoria_Tamaño': '📏 Por Tamaño',
                'Categoria_Rotacion': '🔄 Por Rotación',
                'Categoria_Ruido': '🌫️ Por Ruido'
            }.get(x, x)
        )
    
    with dist_col2:
        st.subheader(f"📊 Distribución de {parameter}")
        
        # Generar gráfico según selecciones
        if chart_type == 'histogram':
            if group_by == 'Ninguno':
                fig = px.histogram(
                    df_detailed,
                    x=parameter,
                    title=f"Histograma de {parameter}",
                    nbins=20,
                    color_discrete_sequence=['#667eea']
                )
            else:
                fig = px.histogram(
                    df_detailed,
                    x=parameter,
                    color=group_by,
                    title=f"Histograma de {parameter} por {group_by}",
                    nbins=20,
                    barmode='overlay',
                    opacity=0.7
                )
        
        elif chart_type == 'box':
            if group_by == 'Ninguno':
                fig = px.box(
                    df_detailed,
                    y=parameter,
                    title=f"Box Plot de {parameter}"
                )
            else:
                fig = px.box(
                    df_detailed,
                    x=group_by,
                    y=parameter,
                    title=f"Box Plot de {parameter} por {group_by}",
                    color=group_by
                )
        
        elif chart_type == 'violin':
            if group_by == 'Ninguno':
                fig = px.violin(
                    df_detailed,
                    y=parameter,
                    title=f"Violin Plot de {parameter}",
                    box=True
                )
            else:
                fig = px.violin(
                    df_detailed,
                    x=group_by,
                    y=parameter,
                    title=f"Violin Plot de {parameter} por {group_by}",
                    color=group_by,
                    box=True
                )
        
        else:  # strip
            if group_by == 'Ninguno':
                fig = px.strip(
                    df_detailed,
                    y=parameter,
                    title=f"Strip Plot de {parameter}"
                )
            else:
                fig = px.strip(
                    df_detailed,
                    x=group_by,
                    y=parameter,
                    title=f"Strip Plot de {parameter} por {group_by}",
                    color=group_by
                )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, width='stretch')
    
    # Estadísticas descriptivas
    st.markdown("---")
    st.header("📊 Estadísticas Descriptivas")
    
    if group_by == 'Ninguno':
        # Estadísticas generales
        stats = df_detailed[parameter].describe()
        
        stats_col1, stats_col2 = st.columns(2)
        
        with stats_col1:
            st.subheader("📈 Estadísticas Básicas")
            st.dataframe(stats, width='stretch')
        
        with stats_col2:
            st.subheader("📊 Información Adicional")
            
            # Percentiles adicionales
            percentiles = [5, 25, 50, 75, 95]
            perc_values = df_detailed[parameter].quantile([p/100 for p in percentiles])
            
            perc_df = pd.DataFrame({
                'Percentil': [f"{p}%" for p in percentiles],
                'Valor': perc_values.values
            })
            
            st.dataframe(perc_df, width='stretch')
    
    else:
        # Estadísticas agrupadas
        grouped_stats = df_detailed.groupby(group_by)[parameter].describe()
        st.dataframe(grouped_stats, width='stretch')
        
        # Gráfico de medias por grupo
        group_means = df_detailed.groupby(group_by)[parameter].mean().sort_values(ascending=False)
        
        fig_means = px.bar(
            x=group_means.index,
            y=group_means.values,
            title=f"📊 Media de {parameter} por {group_by}",
            labels={'x': group_by, 'y': f'Media {parameter}'},
            color=group_means.values,
            color_continuous_scale='viridis'
        )
        
        st.plotly_chart(fig_means, width='stretch')

def show_interactive_gallery(visual_config, df_detailed):
    """Galería interactiva de imágenes"""
    st.header("🖼️ Galería Interactiva de Imágenes")
    
    # Controles de la galería
    gallery_col1, gallery_col2, gallery_col3 = st.columns(3)
    
    with gallery_col1:
        selected_letter = st.selectbox(
            "🔤 Seleccionar letra",
            options=sorted(df_detailed['Letra'].unique()),
            key="gallery_letter_select"
        )
    
    with gallery_col2:
        num_images = st.slider(
            "📊 Número de imágenes",
            min_value=4,
            max_value=50,
            value=12,
            key="gallery_num_images"
        )
    
    with gallery_col3:
        cols_per_row = st.selectbox(
            "📋 Columnas por fila",
            [3, 4, 6, 8],
            index=2,
            key="gallery_cols_per_row"
        )
    
    # Filtros adicionales
    st.markdown("---")
    st.subheader("🔍 Filtros Avanzados")
    
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        size_range = st.slider(
            "📏 Rango de tamaño de fuente",
            min_value=int(df_detailed['Font_Size'].min()),
            max_value=int(df_detailed['Font_Size'].max()),
            value=(int(df_detailed['Font_Size'].min()), int(df_detailed['Font_Size'].max())),
            key="gallery_size_range"
        )
    
    with filter_col2:
        rotation_range = st.slider(
            "🔄 Rango de rotación (±grados)",
            min_value=int(df_detailed['Rotacion'].min()),
            max_value=int(df_detailed['Rotacion'].max()),
            value=(int(df_detailed['Rotacion'].min()), int(df_detailed['Rotacion'].max())),
            key="gallery_rotation_range"
        )
    
    with filter_col3:
        noise_range = st.slider(
            "🌫️ Rango de ruido",
            min_value=float(df_detailed['Noise_Level'].min()),
            max_value=float(df_detailed['Noise_Level'].max()),
            value=(float(df_detailed['Noise_Level'].min()), float(df_detailed['Noise_Level'].max())),
            step=0.01,
            key="gallery_noise_range"
        )
    
    # Aplicar filtros
    filtered_data = df_detailed[
        (df_detailed['Letra'] == selected_letter) &
        (df_detailed['Font_Size'] >= size_range[0]) &
        (df_detailed['Font_Size'] <= size_range[1]) &
        (df_detailed['Rotacion'] >= rotation_range[0]) &
        (df_detailed['Rotacion'] <= rotation_range[1]) &
        (df_detailed['Noise_Level'] >= noise_range[0]) &
        (df_detailed['Noise_Level'] <= noise_range[1])
    ]
    
    st.markdown("---")
    
    if filtered_data.empty:
        st.warning("⚠️ No hay imágenes que coincidan con los filtros seleccionados.")
        return
    
    st.markdown(f"**🎯 Mostrando {min(num_images, len(filtered_data))} de {len(filtered_data)} imágenes para la letra '{selected_letter}' (filtradas)**")
    
    # Obtener imágenes de la configuración
    generated_images = visual_config.get('generated_images', {})
    letter_images = generated_images.get(selected_letter.lower(), [])
    
    if not letter_images:
        st.warning(f"⚠️ No se encontraron imágenes para la letra '{selected_letter}'")
        return
    
    # Mostrar galería
    images_to_show = letter_images[:num_images]
    
    for i in range(0, len(images_to_show), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            if i + j < len(images_to_show):
                image_data = images_to_show[i + j]
                
                with col:
                    # Card para cada imagen
                    st.markdown(f"""
                    <div class="metric-card" style="text-align: center; padding: 1rem;">
                        <h6>📄 Muestra {i + j + 1}</h6>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Mostrar imagen
                    try:
                        # El campo correcto es 'image', no 'image_base64'
                        img_base64 = image_data.get('image', '')
                        if img_base64:
                            img_data = base64.b64decode(img_base64)
                            img = Image.open(io.BytesIO(img_data))
                            st.image(img, use_column_width=True)
                            
                            # Información de parámetros
                            params = image_data.get('params', {})
                            st.caption(f"📏 Font: {params.get('font_size', 'N/A')}")
                            st.caption(f"🔄 Rot: {params.get('rotation', 0):.1f}°")
                            st.caption(f"🌫️ Noise: {params.get('noise_level', 0):.2f}")
                            st.caption(f"🎨 Font: {params.get('font', 'N/A')}")
                            
                        else:
                            st.error("❌ Imagen no disponible")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    # Opciones de exportación de galería
    st.markdown("---")
    st.subheader("💾 Exportar Galería")
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        if st.button("📊 Exportar Datos de Galería"):
            csv_data = filtered_data.to_csv(index=False)
            st.download_button(
                label="💾 Descargar CSV",
                data=csv_data,
                file_name=f"galeria_{selected_letter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with export_col2:
        if st.button("📋 Generar Reporte"):
            # Crear reporte de la galería
            report = {
                'letra': selected_letter,
                'total_imagenes': len(filtered_data),
                'filtros_aplicados': {
                    'tamaño_fuente': f"{size_range[0]}-{size_range[1]}",
                    'rotacion': f"{rotation_range[0]}-{rotation_range[1]}",
                    'ruido': f"{noise_range[0]}-{noise_range[1]}"
                },
                'estadisticas': {
                    'font_size_promedio': filtered_data['Font_Size'].mean(),
                    'rotacion_promedio': filtered_data['Rotacion'].mean(),
                    'ruido_promedio': filtered_data['Noise_Level'].mean()
                },
                'timestamp': datetime.now().isoformat()
            }
            
            report_json = json.dumps(report, indent=2, ensure_ascii=False)
            st.download_button(
                label="📋 Descargar Reporte",
                data=report_json,
                file_name=f"reporte_galeria_{selected_letter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()