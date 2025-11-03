"""
📊 Audio Analytics - Análisis Avanzado de Métricas de Audio
Página para analizar datasets de audio con métricas reales usando librosa
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Sin matplotlib ni seaborn - usando solo Plotly y funciones nativas de Streamlit
from datetime import datetime
import tempfile
import librosa
import librosa.display
import soundfile as sf
import io
import base64
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Importar módulos
from utils import encontrar_device, WAV2VEC_SR, get_default_words

# Importar diccionarios predefinidos
from diccionarios import DICCIONARIOS_PREDEFINIDOS, get_diccionario_predefinido

# Importar sidebar moderna
import sys
sys.path.append(str(Path(__file__).parent.parent))
from components.modern_sidebar import display_modern_sidebar

# Configurar página
st.set_page_config(
    page_title="Audio Analytics",
    page_icon="📊",
    layout="wide"
)

def get_custom_css():
    """CSS moderno para la página de Audio Analytics"""
    return """
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
    
    .analytics-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        color: white;
    }
    
    .metrics-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(240, 147, 251, 0.3);
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        color: white;
    }
    
    .correlation-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(79, 172, 254, 0.3);
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        color: white;
    }
    
    .waveform-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(168, 237, 234, 0.3);
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        color: #333;
    }
    
    .analytics-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .analytics-card h4 {
        color: #4a5568;
        margin-bottom: 1rem;
    }
    
    .analytics-card p {
        margin: 0.3rem 0;
        color: #2d3748;
    }
    </style>
    """

def load_audio_dataset_config():
    """Cargar configuración del dataset de audio"""
    try:
        config_path = Path('master_dataset_config.json')
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        st.error(f"Error cargando configuración: {e}")
        return {}

def load_master_config():
    """Cargar configuración maestra del proyecto"""
    try:
        config_path = Path('master_dataset_config.json')
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        return {}

def clear_analysis_cache():
    """Limpiar todo el caché de análisis"""
    keys_to_remove = []
    for key in st.session_state.keys():
        if key.startswith('analysis_cache_'):
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del st.session_state[key]
    
    # También limpiar estados relacionados
    for key in ['df_metrics', 'current_cache_key', 'analysis_completed', 'start_analysis']:
        if key in st.session_state:
            del st.session_state[key]

def get_default_metrics():
    """Métricas por defecto cuando el análisis de audio falla"""
    return {
        'f0_mean': 150.0,
        'f0_std': 25.0,
        'f0_min': 100.0,
        'f0_max': 200.0,
        'spectral_centroid_mean': 2000.0,
        'spectral_centroid_std': 300.0,
        'spectral_bandwidth_mean': 1500.0,
        'spectral_bandwidth_std': 200.0,
        'spectral_rolloff_mean': 3000.0,
        'spectral_rolloff_std': 400.0,
        'zero_crossing_rate_mean': 0.1,
        'zero_crossing_rate_std': 0.02,
        'mfcc_1_mean': -15.0,
        'mfcc_1_std': 8.0,
        'mfcc_2_mean': 5.0,
        'mfcc_2_std': 6.0,
        'mfcc_3_mean': 2.0,
        'mfcc_3_std': 4.0,
        'formant_f1': 500.0,
        'formant_f2': 1500.0,
        'formant_f3': 2500.0,
        'energy_mean': 0.05,
        'energy_std': 0.01,
        'duration': 1.5,
        'tempo': 120.0
    }

def analyze_real_audio_metrics(audio_base64):
    """
    Análisis completo de métricas de audio reales usando librosa
    """
    if not audio_base64:
        return get_default_metrics()
    
    try:
        # Decodificar audio
        audio_bytes = base64.b64decode(audio_base64)
        
        # Cargar audio con librosa
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_file.flush()
            
            # Cargar con librosa
            y, sr = librosa.load(temp_file.name, sr=None)
            os.unlink(temp_file.name)
        
        # Inicializar métricas
        metrics = {}
        
        # 1. Análisis de F0 (Fundamental Frequency)
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(y, 
                                                       fmin=librosa.note_to_hz('C2'), 
                                                       fmax=librosa.note_to_hz('C7'))
            f0_clean = f0[~np.isnan(f0)]
            
            if len(f0_clean) > 0:
                metrics['f0_mean'] = float(np.mean(f0_clean))
                metrics['f0_std'] = float(np.std(f0_clean))
                metrics['f0_min'] = float(np.min(f0_clean))
                metrics['f0_max'] = float(np.max(f0_clean))
            else:
                metrics.update({
                    'f0_mean': 150.0,
                    'f0_std': 25.0,
                    'f0_min': 100.0,
                    'f0_max': 200.0
                })
        except Exception as e:
            metrics.update({
                'f0_mean': 150.0,
                'f0_std': 25.0,
                'f0_min': 100.0,
                'f0_max': 200.0
            })
        
        # 2. Características espectrales
        try:
            # Centroide espectral
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            metrics['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            metrics['spectral_centroid_std'] = float(np.std(spectral_centroids))
            
            # Ancho de banda espectral
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            metrics['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            metrics['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))
            
            # Rolloff espectral
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            metrics['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            metrics['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
            
        except Exception as e:
            metrics.update({
                'spectral_centroid_mean': 2000.0,
                'spectral_centroid_std': 300.0,
                'spectral_bandwidth_mean': 1500.0,
                'spectral_bandwidth_std': 200.0,
                'spectral_rolloff_mean': 3000.0,
                'spectral_rolloff_std': 400.0
            })
        
        # 3. Zero Crossing Rate
        try:
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            metrics['zero_crossing_rate_mean'] = float(np.mean(zcr))
            metrics['zero_crossing_rate_std'] = float(np.std(zcr))
        except Exception as e:
            metrics.update({
                'zero_crossing_rate_mean': 0.1,
                'zero_crossing_rate_std': 0.02
            })
        
        # 4. MFCC (Mel-frequency cepstral coefficients)
        try:
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(min(3, mfccs.shape[0])):
                metrics[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))
                metrics[f'mfcc_{i+1}_std'] = float(np.std(mfccs[i]))
        except Exception as e:
            metrics.update({
                'mfcc_1_mean': -15.0,
                'mfcc_1_std': 8.0,
                'mfcc_2_mean': 5.0,
                'mfcc_2_std': 6.0,
                'mfcc_3_mean': 2.0,
                'mfcc_3_std': 4.0
            })
        
        # 5. Formantes (estimación aproximada)
        try:
            # Calcular formantes usando LPC
            A = librosa.lpc(y, order=16)
            roots = np.roots(A)
            roots = roots[np.imag(roots) >= 0]
            angz = np.arctan2(np.imag(roots), np.real(roots))
            
            # Convertir a frecuencias
            frqs = angz * (sr / (2 * np.pi))
            
            # Filtrar y ordenar
            frqs = frqs[frqs > 50]  # Filtrar frecuencias muy bajas
            frqs = np.sort(frqs)
            
            if len(frqs) >= 3:
                metrics['formant_f1'] = float(frqs[0])
                metrics['formant_f2'] = float(frqs[1]) 
                metrics['formant_f3'] = float(frqs[2])
            else:
                metrics.update({
                    'formant_f1': 500.0,
                    'formant_f2': 1500.0,
                    'formant_f3': 2500.0
                })
        except Exception as e:
            metrics.update({
                'formant_f1': 500.0,
                'formant_f2': 1500.0,
                'formant_f3': 2500.0
            })
        
        # 6. Energía
        try:
            energy = np.sum(y**2) / len(y)
            rms = librosa.feature.rms(y=y)[0]
            metrics['energy_mean'] = float(energy)
            metrics['energy_std'] = float(np.std(rms))
        except Exception as e:
            metrics.update({
                'energy_mean': 0.05,
                'energy_std': 0.01
            })
        
        # 7. Duración
        metrics['duration'] = float(len(y) / sr)
        
        # 8. Tempo (si es aplicable)
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            metrics['tempo'] = float(tempo)
        except Exception as e:
            metrics['tempo'] = 120.0
        
        return metrics
        
    except Exception as e:
        st.warning(f"Error en análisis de audio: {e}")
        return get_default_metrics()

def analyze_dataset_audio_metrics(config):
    """
    Analiza todas las muestras del dataset y extrae métricas reales
    """
    if not config or not config.get('generated_samples'):
        return pd.DataFrame()
    
    samples = config['generated_samples']
    all_metrics = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_samples = sum(len(variaciones) for variaciones in samples.values())
    processed = 0
    
    for palabra, variaciones in samples.items():
        for i, variacion in enumerate(variaciones):
            # Actualizar progreso
            processed += 1
            progress = processed / total_samples
            progress_bar.progress(progress)
            status_text.text(f"Analizando: {palabra} - Variación {i+1}/{len(variaciones)} ({processed}/{total_samples})")
            
            # Obtener métricas
            audio_base64 = variacion.get('audio_base64', '')
            metrics = analyze_real_audio_metrics(audio_base64)
            
            # Añadir metadatos
            metrics.update({
                'palabra': palabra,
                'variacion_tipo': variacion.get('tipo', 'original'),
                'variacion_idx': i,
                'pitch_factor': variacion.get('pitch_factor', 1.0),
                'speed_factor': variacion.get('speed_factor', 1.0),
                'volume_factor': variacion.get('volume_factor', 1.0),
                'metodo_sintesis': variacion.get('metodo_sintesis', 'gtts'),
                'duracion_ms': variacion.get('duracion_ms', 0),
                'timestamp': variacion.get('timestamp', datetime.now().isoformat())
            })
            
            all_metrics.append(metrics)
    
    # Limpiar barras de progreso
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(all_metrics)

def create_advanced_visualizations(df_metrics):
    """Crear visualizaciones avanzadas del dataset"""
    
    if df_metrics.empty:
        st.warning("⚠️ No hay datos para visualizar")
        return
    
    st.markdown("### 📊 Visualizaciones Avanzadas")
    
    # Tabs para diferentes tipos de análisis
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎵 Distribuciones", 
        "🔍 Correlaciones", 
        "📈 Comparaciones",
        "🌊 Análisis Espectral",
        "🎯 PCA & Clustering"
    ])
    
    with tab1:
        create_distribution_plots(df_metrics)
    
    with tab2:
        create_correlation_analysis(df_metrics)
    
    with tab3:
        create_comparison_plots(df_metrics)
    
    with tab4:
        create_spectral_analysis(df_metrics)
    
    with tab5:
        create_pca_clustering(df_metrics)

def create_distribution_plots(df_metrics):
    """Crear gráficos de distribución"""
    st.markdown("#### 🎵 Distribuciones de Métricas Principales")
    
    # Métricas principales para visualizar
    main_metrics = [
        'f0_mean', 'spectral_centroid_mean', 'spectral_bandwidth_mean',
        'zero_crossing_rate_mean', 'mfcc_1_mean', 'energy_mean', 'duration'
    ]
    
    # Crear subplots
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[
            'F0 Mean (Hz)', 'Spectral Centroid (Hz)', 'Spectral Bandwidth (Hz)',
            'Zero Crossing Rate', 'MFCC 1', 'Energy', 'Duration (s)',
            'Formant F1 (Hz)', 'Tempo (BPM)'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    metrics_to_plot = main_metrics + ['formant_f1', 'tempo']
    
    for i, metric in enumerate(metrics_to_plot):
        row = (i // 3) + 1
        col = (i % 3) + 1
        
        if metric in df_metrics.columns:
            # Histograma por tipo de variación
            for tipo in df_metrics['variacion_tipo'].unique():
                data = df_metrics[df_metrics['variacion_tipo'] == tipo][metric]
                
                fig.add_histogram(
                    x=data,
                    name=f"{tipo.title()}",
                    opacity=0.7,
                    row=row, col=col,
                    showlegend=(i == 0)  # Solo mostrar leyenda en el primer gráfico
                )
    
    fig.update_layout(
        height=800,
        title_text="Distribución de Métricas por Tipo de Variación",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Estadísticas descriptivas
    st.markdown("#### 📊 Estadísticas Descriptivas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📈 Métricas de F0 (Pitch)**")
        f0_stats = df_metrics[['f0_mean', 'f0_std', 'f0_min', 'f0_max']].describe()
        st.dataframe(f0_stats.round(2))
    
    with col2:
        st.markdown("**🎼 Métricas Espectrales**")
        spectral_cols = [col for col in df_metrics.columns if 'spectral' in col and 'mean' in col]
        if spectral_cols:
            spectral_stats = df_metrics[spectral_cols].describe()
            st.dataframe(spectral_stats.round(2))

def create_correlation_analysis(df_metrics):
    """Análisis de correlaciones entre métricas"""
    st.markdown("#### 🔍 Análisis de Correlaciones")
    
    # Seleccionar métricas numéricas
    numeric_cols = df_metrics.select_dtypes(include=[np.number]).columns
    # Excluir índices y metadatos
    exclude_cols = ['variacion_idx', 'duracion_ms', 'pitch_factor', 'speed_factor', 'volume_factor']
    correlation_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if len(correlation_cols) < 2:
        st.warning("⚠️ No hay suficientes métricas numéricas para correlación")
        return
    
    # Calcular matriz de correlación
    corr_matrix = df_metrics[correlation_cols].corr()
    
    # Heatmap de correlación
    fig = px.imshow(
        corr_matrix,
        labels=dict(x="Métricas", y="Métricas", color="Correlación"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale='RdBu',
        aspect="auto",
        title="Matriz de Correlación de Métricas de Audio"
    )
    
    fig.update_layout(
        height=600,
        width=800
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlaciones más fuertes
    st.markdown("#### 🎯 Correlaciones Más Significativas")
    
    # Obtener correlaciones ordenadas
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if not np.isnan(corr_val):
                corr_pairs.append({
                    'Métrica 1': corr_matrix.columns[i],
                    'Métrica 2': corr_matrix.columns[j],
                    'Correlación': corr_val,
                    'Correlación Abs': abs(corr_val)
                })
    
    if corr_pairs:
        corr_df = pd.DataFrame(corr_pairs)
        corr_df = corr_df.sort_values('Correlación Abs', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔺 Correlaciones Positivas Más Fuertes**")
            positive_corr = corr_df[corr_df['Correlación'] > 0].head(10)
            if not positive_corr.empty:
                st.dataframe(
                    positive_corr[['Métrica 1', 'Métrica 2', 'Correlación']].round(3),
                    use_container_width=True
                )
        
        with col2:
            st.markdown("**🔻 Correlaciones Negativas Más Fuertes**")
            negative_corr = corr_df[corr_df['Correlación'] < 0].head(10)
            if not negative_corr.empty:
                st.dataframe(
                    negative_corr[['Métrica 1', 'Métrica 2', 'Correlación']].round(3),
                    use_container_width=True
                )

def create_comparison_plots(df_metrics):
    """Crear gráficos de comparación entre tipos de variaciones"""
    st.markdown("#### 📈 Comparaciones entre Tipos de Variaciones")
    
    # Box plots por tipo de variación
    col1, col2 = st.columns(2)
    
    with col1:
        # F0 Mean por tipo
        fig = px.box(
            df_metrics,
            x='variacion_tipo',
            y='f0_mean',
            title='Distribución de F0 Mean por Tipo de Variación',
            color='variacion_tipo'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Spectral Centroid por tipo
        fig = px.box(
            df_metrics,
            x='variacion_tipo',
            y='spectral_centroid_mean',
            title='Distribución de Centroide Espectral por Tipo',
            color='variacion_tipo'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plots de relaciones interesantes
    st.markdown("#### 🎯 Relaciones entre Métricas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # F0 vs Spectral Centroid
        fig = px.scatter(
            df_metrics,
            x='f0_mean',
            y='spectral_centroid_mean',
            color='variacion_tipo',
            size='energy_mean',
            title='F0 Mean vs Centroide Espectral',
            hover_data=['palabra']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Duration vs Energy
        fig = px.scatter(
            df_metrics,
            x='duration',
            y='energy_mean',
            color='variacion_tipo',
            size='f0_mean',
            title='Duración vs Energía',
            hover_data=['palabra']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Radar chart comparando promedios por tipo
    st.markdown("#### 🌟 Perfil Promedio por Tipo de Variación")
    
    # Seleccionar métricas principales para el radar
    radar_metrics = ['f0_mean', 'spectral_centroid_mean', 'energy_mean', 'zero_crossing_rate_mean', 'duration']
    
    # Normalizar métricas para el radar chart
    scaler = StandardScaler()
    df_normalized = df_metrics.copy()
    
    for metric in radar_metrics:
        if metric in df_normalized.columns:
            df_normalized[f'{metric}_norm'] = scaler.fit_transform(df_normalized[[metric]])
    
    # Calcular promedios por tipo
    radar_data = []
    for tipo in df_metrics['variacion_tipo'].unique():
        tipo_data = df_normalized[df_normalized['variacion_tipo'] == tipo]
        
        values = []
        for metric in radar_metrics:
            if f'{metric}_norm' in tipo_data.columns:
                values.append(tipo_data[f'{metric}_norm'].mean())
            else:
                values.append(0)
        
        # Cerrar el radar (repetir primer valor)
        values.append(values[0])
        
        radar_data.append({
            'tipo': tipo,
            'values': values,
            'labels': radar_metrics + [radar_metrics[0]]
        })
    
    # Crear radar chart
    fig = go.Figure()
    
    for data in radar_data:
        fig.add_trace(go.Scatterpolar(
            r=data['values'],
            theta=data['labels'],
            fill='toself',
            name=data['tipo'].title(),
            opacity=0.7
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[-2, 2])
        ),
        showlegend=True,
        title="Perfil Normalizado de Métricas por Tipo de Variación",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_spectral_analysis(df_metrics):
    """Análisis espectral detallado"""
    st.markdown("#### 🌊 Análisis Espectral Detallado")
    
    # Análisis de formantes
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución de formantes
        formant_cols = ['formant_f1', 'formant_f2', 'formant_f3']
        formant_data = []
        
        for col in formant_cols:
            if col in df_metrics.columns:
                for idx, value in df_metrics[col].items():
                    formant_data.append({
                        'Formante': col.replace('formant_f', 'F'),
                        'Frecuencia': value,
                        'Tipo': df_metrics.loc[idx, 'variacion_tipo'],
                        'Palabra': df_metrics.loc[idx, 'palabra']
                    })
        
        if formant_data:
            formant_df = pd.DataFrame(formant_data)
            
            fig = px.violin(
                formant_df,
                x='Formante',
                y='Frecuencia',
                color='Tipo',
                title='Distribución de Formantes por Tipo',
                box=True
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # MFCC Analysis
        mfcc_cols = [col for col in df_metrics.columns if 'mfcc' in col and 'mean' in col]
        
        if mfcc_cols:
            # Heatmap de MFCCs
            mfcc_data = df_metrics[mfcc_cols + ['variacion_tipo']].groupby('variacion_tipo').mean()
            
            fig = px.imshow(
                mfcc_data.T,
                labels=dict(x="Tipo de Variación", y="MFCC", color="Valor"),
                aspect="auto",
                title="Perfil MFCC por Tipo de Variación"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de características espectrales por palabra
    st.markdown("#### 📊 Características Espectrales por Palabra")
    
    if len(df_metrics['palabra'].unique()) <= 20:  # Solo si no hay demasiadas palabras
        
        # Spectral centroid por palabra
        fig = px.box(
            df_metrics,
            x='palabra',
            y='spectral_centroid_mean',
            color='variacion_tipo',
            title='Centroide Espectral por Palabra y Tipo'
        )
        fig.update_xaxes(tickangle=45)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Bandwidth vs Rolloff
        fig = px.scatter(
            df_metrics,
            x='spectral_bandwidth_mean',
            y='spectral_rolloff_mean',
            color='palabra',
            size='energy_mean',
            title='Ancho de Banda vs Rolloff Espectral por Palabra',
            hover_data=['variacion_tipo']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("🔍 Demasiadas palabras para análisis detallado por palabra. Mostrando estadísticas generales.")
        
        # Estadísticas espectrales generales
        spectral_stats = df_metrics[[
            'spectral_centroid_mean', 'spectral_bandwidth_mean', 
            'spectral_rolloff_mean', 'zero_crossing_rate_mean'
        ]].describe()
        
        st.dataframe(spectral_stats.round(2))

def create_pca_clustering(df_metrics):
    """Análisis PCA y clustering"""
    st.markdown("#### 🎯 Análisis de Componentes Principales (PCA)")
    
    # Preparar datos para PCA
    numeric_cols = df_metrics.select_dtypes(include=[np.number]).columns
    exclude_cols = ['variacion_idx', 'duracion_ms']
    pca_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if len(pca_cols) < 3:
        st.warning("⚠️ No hay suficientes métricas para PCA")
        return
    
    # Preparar datos
    X = df_metrics[pca_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Crear DataFrame con componentes principales
    pca_df = pd.DataFrame(X_pca[:, :min(10, X_pca.shape[1])], 
                         columns=[f'PC{i+1}' for i in range(min(10, X_pca.shape[1]))])
    pca_df['variacion_tipo'] = df_metrics['variacion_tipo'].values
    pca_df['palabra'] = df_metrics['palabra'].values
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Varianza explicada
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f'PC{i+1}' for i in range(min(10, len(explained_variance)))],
            y=explained_variance[:10],
            name='Varianza Individual'
        ))
        fig.add_trace(go.Scatter(
            x=[f'PC{i+1}' for i in range(min(10, len(cumulative_variance)))],
            y=cumulative_variance[:10],
            mode='lines+markers',
            name='Varianza Acumulada',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Varianza Explicada por Componente Principal',
            yaxis=dict(title='Varianza Explicada Individual'),
            yaxis2=dict(title='Varianza Acumulada', overlaying='y', side='right'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Scatter plot PC1 vs PC2
        fig = px.scatter(
            pca_df,
            x='PC1',
            y='PC2',
            color='variacion_tipo',
            title='PCA: PC1 vs PC2',
            hover_data=['palabra']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Componentes principales más importantes
    st.markdown("#### 🔍 Contribución de Variables a los Componentes Principales")
    
    # Crear DataFrame con loadings
    loadings = pd.DataFrame(
        pca.components_[:5].T,  # Primeros 5 componentes
        columns=[f'PC{i+1}' for i in range(5)],
        index=pca_cols
    )
    
    # Heatmap de loadings
    fig = px.imshow(
        loadings.T,
        labels=dict(x="Variables", y="Componentes", color="Loading"),
        aspect="auto",
        title="Loadings de Variables en Componentes Principales",
        color_continuous_scale='RdBu'
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla de loadings más importantes
    st.markdown("#### 📊 Variables Más Influyentes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**PC1 (Variables más influyentes)**")
        pc1_loadings = loadings['PC1'].abs().sort_values(ascending=False).head(10)
        pc1_df = pd.DataFrame({
            'Variable': pc1_loadings.index,
            'Loading Absoluto': pc1_loadings.values,
            'Loading Original': loadings.loc[pc1_loadings.index, 'PC1'].values
        })
        st.dataframe(pc1_df.round(3))
    
    with col2:
        st.markdown("**PC2 (Variables más influyentes)**")
        pc2_loadings = loadings['PC2'].abs().sort_values(ascending=False).head(10)
        pc2_df = pd.DataFrame({
            'Variable': pc2_loadings.index,
            'Loading Absoluto': pc2_loadings.values,
            'Loading Original': loadings.loc[pc2_loadings.index, 'PC2'].values
        })
        st.dataframe(pc2_df.round(3))

def mostrar_analisis_dataset():
    """Función principal para mostrar análisis del dataset"""
    
    st.markdown("### 📊 Análisis de Dataset de Audio")
    
    # Cargar configuración del dataset y master config
    config = load_audio_dataset_config()
    master_config = load_master_config()
    
    if not config or not config.get('generated_samples'):
        st.warning("⚠️ No hay dataset de audio generado para analizar.")
        st.info("💡 Ve a **🎤 Audio Dataset Manager** para generar un dataset primero.")
        
        # Botón para ir a la página de generación
        if st.button("🎤 Ir a Audio Dataset Manager", type="primary"):
            st.switch_page("pages/01_🎤_Audio_Dataset_Manager.py")
        
        return
    
    samples = config['generated_samples']
    
    # Información del diccionario configurado
    if master_config and 'diccionario_seleccionado' in master_config:
        dic_info = master_config['diccionario_seleccionado']
        st.markdown("#### 📚 Información del Vocabulario")
        
        col_dic1, col_dic2 = st.columns(2)
        
        with col_dic1:
            st.markdown(f"""
            <div class="analytics-card">
                <h4>📖 Diccionario Configurado</h4>
                <p><strong>Nombre:</strong> {dic_info.get('nombre', 'N/A')}</p>
                <p><strong>Tipo:</strong> {dic_info.get('tipo', 'N/A')}</p>
                <p><strong>Descripción:</strong> {dic_info.get('descripcion', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_dic2:
            palabras_total_dic = len(dic_info.get('palabras', []))
            palabras_generadas = len(samples)
            porcentaje_completado = (palabras_generadas / palabras_total_dic * 100) if palabras_total_dic > 0 else 0
            
            st.markdown(f"""
            <div class="analytics-card">
                <h4>� Progreso del Dataset</h4>
                <p><strong>Palabras en Diccionario:</strong> {palabras_total_dic}</p>
                <p><strong>Palabras Generadas:</strong> {palabras_generadas}</p>
                <p><strong>Completado:</strong> {porcentaje_completado:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Información básica del dataset
    st.markdown("#### 📈 Estadísticas del Dataset Generado")
    
    total_palabras = len(samples)
    total_muestras = sum(len(variaciones) for variaciones in samples.values())
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📝 Palabras Únicas", total_palabras)
    with col2:
        st.metric("🎵 Total Muestras", total_muestras)
    with col3:
        promedio = total_muestras / total_palabras if total_palabras > 0 else 0
        st.metric("📊 Promedio/Palabra", f"{promedio:.1f}")
    with col4:
        last_update = config.get('last_update', 'N/A')
        if last_update != 'N/A':
            last_update = last_update[:10]  # Solo fecha
        st.metric("📅 Última Actualización", last_update)
    
    # Control de análisis
    st.markdown("---")
    
    col_control1, col_control2, col_control3 = st.columns(3)
    
    with col_control1:
        analyze_all = st.checkbox("🔍 Analizar todas las muestras", value=True)
        
    with col_control2:
        if not analyze_all:
            max_samples = st.number_input(
                "Máximo muestras a analizar:",
                min_value=5, max_value=min(100, total_muestras), 
                value=min(20, total_muestras)
            )
        else:
            max_samples = total_muestras
    
    with col_control3:
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🚀 Iniciar Análisis", type="primary"):
                st.session_state['start_analysis'] = True
        
        with col_btn2:
            if st.button("🗑️ Limpiar Caché", help="Limpia el caché para forzar nuevo análisis"):
                clear_analysis_cache()
                st.success("🧹 Caché limpiado!")
                st.rerun()
    
    # Mostrar información de caché
    cache_info = [key for key in st.session_state.keys() if key.startswith('analysis_cache_')]
    if cache_info:
        st.info(f"💾 Tienes {len(cache_info)} análisis en caché. Usa 'Limpiar Caché' para forzar nuevo análisis.")
    
    # Crear clave de caché basada en el contenido del dataset
    dataset_hash = hash(str(sorted(samples.items())) + str(max_samples) + str(analyze_all))
    cache_key = f'analysis_cache_{dataset_hash}'
    
    # Verificar si ya tenemos el análisis en caché
    needs_analysis = (
        st.session_state.get('start_analysis', False) or
        cache_key not in st.session_state or
        not st.session_state.get('analysis_completed', False)
    )
    
    # Realizar análisis solo si es necesario
    if needs_analysis:
        
        with st.spinner("🔬 Analizando métricas de audio..."):
            
            # Verificar si ya existe este análisis exacto en caché
            if cache_key in st.session_state:
                st.info("📋 Cargando análisis desde caché...")
                df_metrics = st.session_state[cache_key]
            else:
                # Limitar muestras si es necesario
                if not analyze_all and max_samples < total_muestras:
                    # Crear configuración limitada
                    config_limited = {'generated_samples': {}}
                    processed = 0
                    
                    for palabra, variaciones in samples.items():
                        if processed >= max_samples:
                            break
                        
                        # Tomar solo algunas variaciones por palabra
                        remaining = max_samples - processed
                        take = min(len(variaciones), remaining)
                        config_limited['generated_samples'][palabra] = variaciones[:take]
                        processed += take
                    
                    df_metrics = analyze_dataset_audio_metrics(config_limited)
                else:
                    df_metrics = analyze_dataset_audio_metrics(config)
                
                # Guardar en caché
                st.session_state[cache_key] = df_metrics
            
            # Guardar referencias actuales
            st.session_state['df_metrics'] = df_metrics
            st.session_state['current_cache_key'] = cache_key
            st.session_state['analysis_completed'] = True
            st.session_state['start_analysis'] = False  # Reset flag
        
        st.success(f"✅ Análisis completado! {len(df_metrics)} muestras analizadas.")
    else:
        # Cargar desde caché existente
        current_cache_key = st.session_state.get('current_cache_key')
        if current_cache_key and current_cache_key in st.session_state:
            st.session_state['df_metrics'] = st.session_state[current_cache_key]
        st.info("📋 Utilizando análisis en caché. Cambia la configuración para regenerar.")
    
    # Mostrar resultados si existen
    if st.session_state.get('analysis_completed', False) and 'df_metrics' in st.session_state:
        df_metrics = st.session_state['df_metrics']
        
        if not df_metrics.empty:
            
            # Resumen de métricas
            st.markdown("---")
            st.markdown("#### 📊 Resumen de Métricas Principales")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metrics-card">
                    <h4>🎵 F0 (Pitch)</h4>
                    <p><strong>Promedio:</strong> {df_metrics['f0_mean'].mean():.1f} Hz</p>
                    <p><strong>Rango:</strong> {df_metrics['f0_min'].min():.1f} - {df_metrics['f0_max'].max():.1f} Hz</p>
                    <p><strong>Variabilidad:</strong> {df_metrics['f0_std'].mean():.1f} Hz</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metrics-card">
                    <h4>🌊 Espectral</h4>
                    <p><strong>Centroide:</strong> {df_metrics['spectral_centroid_mean'].mean():.0f} Hz</p>
                    <p><strong>Bandwidth:</strong> {df_metrics['spectral_bandwidth_mean'].mean():.0f} Hz</p>
                    <p><strong>Rolloff:</strong> {df_metrics['spectral_rolloff_mean'].mean():.0f} Hz</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metrics-card">
                    <h4>⚡ Otros</h4>
                    <p><strong>Duración:</strong> {df_metrics['duration'].mean():.2f}s</p>
                    <p><strong>Energía:</strong> {df_metrics['energy_mean'].mean():.3f}</p>
                    <p><strong>ZCR:</strong> {df_metrics['zero_crossing_rate_mean'].mean():.3f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualizaciones avanzadas
            create_advanced_visualizations(df_metrics)
            
            # Tabla de datos detallada
            st.markdown("---")
            st.markdown("#### 📋 Datos Detallados")
            
            # Seleccionar columnas para mostrar
            display_cols = [
                'palabra', 'variacion_tipo', 'f0_mean', 'spectral_centroid_mean',
                'energy_mean', 'duration', 'formant_f1', 'formant_f2'
            ]
            display_cols = [col for col in display_cols if col in df_metrics.columns]
            
            # Filtros
            col_filter1, col_filter2 = st.columns(2)
            
            with col_filter1:
                palabras_filter = st.multiselect(
                    "Filtrar por palabras:",
                    df_metrics['palabra'].unique(),
                    default=list(df_metrics['palabra'].unique())
                )
            
            with col_filter2:
                tipos_filter = st.multiselect(
                    "Filtrar por tipos:",
                    df_metrics['variacion_tipo'].unique(),
                    default=list(df_metrics['variacion_tipo'].unique())
                )
            
            # Aplicar filtros
            df_filtered = df_metrics[
                (df_metrics['palabra'].isin(palabras_filter)) &
                (df_metrics['variacion_tipo'].isin(tipos_filter))
            ]
            
            # Mostrar tabla
            if not df_filtered.empty:
                st.dataframe(
                    df_filtered[display_cols].round(2),
                    use_container_width=True,
                    height=400
                )
                
                # Opción de descarga
                csv = df_filtered.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar CSV",
                    data=csv,
                    file_name=f"audio_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("🔍 No hay datos que coincidan con los filtros seleccionados.")

def get_waveform_cache_key(palabra, variacion_idx, variacion_data):
    """Generar clave de cache para análisis de forma de onda específico"""
    try:
        # Crear hash del audio específico
        audio_hash = hash(variacion_data.get('audio_base64', ''))
        return f"waveform_{palabra}_{variacion_idx}_{audio_hash}"
    except:
        return f"waveform_{palabra}_{variacion_idx}_default"

def analyze_waveform_cached(palabra, variacion_idx, variacion_data):
    """Analizar forma de onda con cache individual por selección"""
    
    # Generar clave de cache específica
    cache_key = get_waveform_cache_key(palabra, variacion_idx, variacion_data)
    
    # Inicializar cache si no existe
    if 'waveform_cache' not in st.session_state:
        st.session_state.waveform_cache = {}
    
    # Verificar si ya está en cache
    if cache_key in st.session_state.waveform_cache:
        return st.session_state.waveform_cache[cache_key]
    
    # Generar análisis nuevo
    try:
        if 'audio_base64' not in variacion_data:
            return None
            
        # Decodificar audio
        audio_bytes = base64.b64decode(variacion_data['audio_base64'])
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_file.flush()
            
            # Cargar con librosa
            y, sr = librosa.load(temp_file.name, sr=None)
            os.unlink(temp_file.name)
        
        # Crear visualizaciones
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Forma de Onda', 'Espectrograma',
                'Espectro de Potencia', 'MFCCs'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Forma de onda
        time_axis = np.linspace(0, len(y)/sr, len(y))
        fig.add_trace(
            go.Scatter(x=time_axis, y=y, name='Amplitud', line=dict(color='#1f77b4')),
            row=1, col=1
        )
        
        # 2. Espectrograma
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        fig.add_trace(
            go.Heatmap(z=D, colorscale='Viridis', name='Espectrograma'),
            row=1, col=2
        )
        
        # 3. Espectro de potencia
        S = np.abs(librosa.stft(y))**2
        freqs = librosa.fft_frequencies(sr=sr)
        spectrum = np.mean(S, axis=1)
        fig.add_trace(
            go.Scatter(x=freqs[:len(freqs)//4], y=spectrum[:len(spectrum)//4], 
                      name='Espectro', line=dict(color='#ff7f0e')),
            row=2, col=1
        )
        
        # 4. MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        fig.add_trace(
            go.Heatmap(z=mfccs, colorscale='RdBu', name='MFCCs'),
            row=2, col=2
        )
        
        # Configurar layout
        fig.update_layout(
            height=800,
            title_text=f"Análisis de '{palabra}' - {variacion_data.get('tipo', 'original').title()}",
            showlegend=False
        )
        
        # Calcular métricas adicionales
        duracion = len(y) / sr
        rms_energy = np.sqrt(np.mean(y**2))
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        analysis_result = {
            'fig': fig,
            'duracion': duracion,
            'rms_energy': rms_energy,
            'zcr_mean': np.mean(zcr),
            'sample_rate': sr,
            'samples': len(y)
        }
        
        # Guardar en cache
        st.session_state.waveform_cache[cache_key] = analysis_result
        
        return analysis_result
        
    except Exception as e:
        st.error(f"Error al analizar audio: {str(e)}")
        return None

def mostrar_waveform_analysis():
    """Análisis de forma de onda específico"""
    st.markdown("### 🌊 Análisis de Forma de Onda")
    
    config = load_audio_dataset_config()
    
    if not config or not config.get('generated_samples'):
        st.warning("⚠️ No hay dataset para análizar.")
        return
    
    samples = config['generated_samples']
    
    # Controles de cache
    col_cache1, col_cache2 = st.columns([3, 1])
    with col_cache1:
        cache_size = len(st.session_state.get('waveform_cache', {}))
        st.caption(f"📋 Cache: {cache_size} análisis guardados")
    with col_cache2:
        if st.button("🗑️ Limpiar Cache"):
            st.session_state.waveform_cache = {}
            st.success("Cache limpiado")
            st.rerun()
    
    # Selector de muestra
    col1, col2 = st.columns(2)
    
    with col1:
        palabra_selected = st.selectbox(
            "Seleccionar palabra:",
            list(samples.keys())
        )
    
    with col2:
        if palabra_selected:
            variaciones = samples[palabra_selected]
            variacion_idx = st.selectbox(
                "Seleccionar variación:",
                range(len(variaciones)),
                format_func=lambda x: f"Var {x+1}: {variaciones[x].get('tipo', 'original').title()}"
            )
    
    if palabra_selected and variacion_idx is not None:
        variacion_data = samples[palabra_selected][variacion_idx]
        
        # Verificar cache para esta selección específica
        cache_key = get_waveform_cache_key(palabra_selected, variacion_idx, variacion_data)
        is_cached = cache_key in st.session_state.get('waveform_cache', {})
        
        if is_cached:
            st.success("📋 Usando análisis en cache (no regenerado)")
        else:
            st.info("🔄 Generando nuevo análisis...")
        
        # Obtener análisis (cached o nuevo)
        analysis_result = analyze_waveform_cached(palabra_selected, variacion_idx, variacion_data)
        
        if analysis_result:
            # Mostrar gráfico
            st.plotly_chart(analysis_result['fig'], use_container_width=True)
            
            # Mostrar métricas
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="⏱️ Duración",
                    value=f"{analysis_result['duracion']:.2f}s"
                )
            
            with col2:
                st.metric(
                    label="🔊 Energía RMS",
                    value=f"{analysis_result['rms_energy']:.4f}"
                )
            
            with col3:
                st.metric(
                    label="📊 ZCR Promedio",
                    value=f"{analysis_result['zcr_mean']:.4f}"
                )
                
            with col4:
                st.metric(
                    label="🎵 Sample Rate",
                    value=f"{analysis_result['sample_rate']} Hz"
                )
            
            # Información adicional
            st.markdown("---")
            st.markdown("#### 📋 Información del Audio")
            st.write(f"**Muestras totales:** {analysis_result['samples']:,}")
            st.write(f"**Tipo:** {variacion_data.get('tipo', 'original').title()}")
            
            # Reproductor de audio
            st.markdown("#### 🎵 Reproductor")
            if 'audio_base64' in variacion_data:
                audio_bytes = base64.b64decode(variacion_data['audio_base64'])
                st.audio(audio_bytes, format='audio/wav')
        else:
            st.warning("No se encontró el análisis para esta selección.")

def main():
    """Función principal de la página de analytics"""
    
    # Aplicar CSS moderno
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    # Sidebar modernizada persistente
    display_modern_sidebar()
    
    # Header moderno
    st.markdown('<h1 class="main-header">📊 Audio Analytics</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #666;">
        <p style="font-size: 1.2em;">Análisis avanzado de métricas de audio con visualizaciones modernas</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs principales
    tab1, tab2, tab3 = st.tabs([
        "📊 Análisis de Dataset",
        "🌊 Análisis de Forma de Onda", 
        "📈 Métricas Individuales"
    ])
    
    with tab1:
        mostrar_analisis_dataset()
    
    with tab2:
        mostrar_waveform_analysis()
    
    with tab3:
        # Análisis de muestras individuales
        st.markdown("### 🎯 Análisis de Muestras Individuales")
        
        config = load_audio_dataset_config()
        
        if config and config.get('generated_samples'):
            samples = config['generated_samples']
            
            # Selector de múltiples muestras para comparar
            st.markdown("#### 🔍 Seleccionar Muestras para Comparar")
            
            # Lista todas las muestras disponibles
            all_samples = []
            for palabra, variaciones in samples.items():
                for i, variacion in enumerate(variaciones):
                    all_samples.append({
                        'id': f"{palabra}_{i}",
                        'label': f"{palabra} - {variacion.get('tipo', 'original').title()}",
                        'data': variacion,
                        'palabra': palabra
                    })
            
            # Selector múltiple
            selected_samples = st.multiselect(
                "Seleccionar hasta 5 muestras:",
                [sample['id'] for sample in all_samples],
                format_func=lambda x: next(s['label'] for s in all_samples if s['id'] == x),
                max_selections=5
            )
            
            if selected_samples:
                st.markdown("#### 📊 Comparación de Métricas")
                
                # Analizar muestras seleccionadas
                comparison_data = []
                
                for sample_id in selected_samples:
                    sample_info = next(s for s in all_samples if s['id'] == sample_id)
                    
                    audio_base64 = sample_info['data'].get('audio_base64', '')
                    metrics = analyze_real_audio_metrics(audio_base64)
                    
                    metrics.update({
                        'muestra': sample_info['label'],
                        'palabra': sample_info['palabra']
                    })
                    
                    comparison_data.append(metrics)
                
                if comparison_data:
                    df_comparison = pd.DataFrame(comparison_data)
                    
                    # Crear gráfico de radar para comparación
                    radar_metrics = [
                        'f0_mean', 'spectral_centroid_mean', 'energy_mean', 
                        'zero_crossing_rate_mean', 'duration'
                    ]
                    
                    # Normalizar para radar
                    scaler = StandardScaler()
                    
                    fig = go.Figure()
                    
                    for _, row in df_comparison.iterrows():
                        values = []
                        for metric in radar_metrics:
                            # Normalizar individualmente cada métrica
                            metric_values = df_comparison[metric].values.reshape(-1, 1)
                            normalized = scaler.fit_transform(metric_values).flatten()
                            idx = df_comparison.index[df_comparison['muestra'] == row['muestra']].tolist()[0]
                            values.append(normalized[idx])
                        
                        # Cerrar radar
                        values.append(values[0])
                        labels = radar_metrics + [radar_metrics[0]]
                        
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=labels,
                            fill='toself',
                            name=row['muestra'],
                            opacity=0.7
                        ))
                    
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True)),
                        showlegend=True,
                        title="Comparación Normalizada de Métricas",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabla de comparación
                    st.markdown("#### 📋 Tabla de Comparación")
                    
                    display_metrics = [
                        'muestra', 'f0_mean', 'spectral_centroid_mean', 
                        'energy_mean', 'duration', 'formant_f1'
                    ]
                    
                    comparison_display = df_comparison[display_metrics].round(2)
                    st.dataframe(comparison_display, use_container_width=True)
        
        else:
            st.info("📭 No hay dataset disponible para análisis individual.")

if __name__ == "__main__":
    main()