import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def plot_learning_curves(history_df):
    """
    Plotea curvas de aprendizaje interactivas usando Plotly.
    Espera un DataFrame con columnas como 'train_loss', 'val_loss', 'train_top1', 'val_top1'.
    """
    if history_df.empty:
        st.warning("No hay datos de entrenamiento para mostrar.")
        return

    # Loss Chart
    st.subheader("📉 Curvas de Pérdida (Loss)")
    loss_cols = [c for c in history_df.columns if 'loss' in c]
    if loss_cols:
        fig_loss = px.line(history_df, y=loss_cols, markers=True, title="Evolución de la Pérdida")
        fig_loss.update_layout(xaxis_title="Época", yaxis_title="Loss", hovermode="x unified")
        st.plotly_chart(fig_loss, width="stretch")

    # Accuracy Chart
    acc_cols = [c for c in history_df.columns if 'top1' in c or 'acc' in c]
    if acc_cols:
        st.subheader("📈 Curvas de Precisión (Accuracy)")
        fig_acc = px.line(history_df, y=acc_cols, markers=True, title="Evolución de la Precisión")
        fig_acc.update_layout(xaxis_title="Época", yaxis_title="Accuracy (%)", hovermode="x unified")
        st.plotly_chart(fig_acc, width="stretch")

def plot_training_history(history_list):
    """
    Toma un historico tipo lista generada guardada en .meta.json y genera la gráfica marcando el 'best checkpoint'.
    """
    if not history_list:
        st.warning("No hay historial disponible correspondiente al modelo guardado.")
        return
        
    df = pd.DataFrame(history_list)
    if 'epoch' not in df.columns:
        df['epoch'] = df.index
        
    best_epoch = None
    if 'val_loss' in df.columns:
        best_epoch = df['val_loss'].idxmin()
    
    # Loss plot
    loss_cols = [c for c in df.columns if 'loss' in c]
    if loss_cols:
        fig_loss = px.line(df, x='epoch', y=loss_cols, title="Curva de Historial (Loss)")
        if best_epoch is not None:
            fig_loss.add_vline(x=df.loc[best_epoch, 'epoch'], line_dash="dash", line_color="red", annotation_text="Best Checkpoint Mínimo")
        st.plotly_chart(fig_loss, width="stretch")

    # Accuracy / Metrics plot
    acc_cols = [c for c in df.columns if 'acc' in c or 'top1' in c or 'dtw' in c or 'perceptual' in c]
    if acc_cols:
        fig_acc = px.line(df, x='epoch', y=acc_cols, title="Curva de Historial (Métricas)")
        if best_epoch is not None:
            fig_acc.add_vline(x=df.loc[best_epoch, 'epoch'], line_dash="dash", line_color="red", annotation_text="Best Checkpoint Mínimo")
        st.plotly_chart(fig_acc, width="stretch")

def plot_confusion_matrix(y_true, y_pred, classes, title="Matriz de Confusión"):
    """
    Genera y muestra una matriz de confusión visualmente atractiva.
    Maneja tanto índices enteros como etiquetas de texto.
    """
    if not y_true or len(y_true) == 0:
        st.warning("No hay suficientes datos de predicción para generar la matriz de confusión.")
        return None

    try:
        # Detectar tipo de datos y ajustar labels
        if isinstance(y_true[0], str):
            # Si son strings, los labels son los nombres de las clases
            labels = classes
            ticks = classes
        else:
            # Si son enteros, los labels son los índices
            labels = list(range(len(classes)))
            ticks = classes
            
        # Asegurar que al menos un label esté presente en y_true para evitar error de sklearn
        valid_labels = [L for L in labels if L in y_true]
        if not valid_labels:
            st.warning("⚠️ Ninguna de las clases esperadas se encontró en las predicciones de validación todavía.")
            return None

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Normalizar para visualización (evitar division por cero)
        with np.errstate(all='ignore'):
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.nan_to_num(cm_norm)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=ticks, yticklabels=ticks, ax=ax)
        plt.ylabel('Verdadero')
        plt.xlabel('Predicho')
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        
        st.pyplot(fig)
        return cm
        
    except Exception as e:
        st.error(f"Error generando matriz de confusión: {e}")
        return None

def plot_probability_matrix(y_true, y_probs, classes, title="Mapa de Calor de Probabilidades"):
    """
    Genera un heatmap donde el eje Y es la clase verdadera y el eje X es la distribución
    de probabilidad promedio asignada por el modelo.
    Si hay muchas clases (>40), oculta las etiquetas para evitar saturación.
    """
    n_classes = len(classes)
    
    # Convertir a arrays numpy si no lo son
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    
    # Matriz (C, C)
    # M[i, j] = Promedio de probabilidad asignada a la clase j cuando la clase real es i
    prob_matrix = np.zeros((n_classes, n_classes))
    
    # Calcular promedio por clase
    # Iteramos por cada clase para sacar el promedio de sus muestras
    for i in range(n_classes):
        # Índices donde la clase real es i
        idxs = np.where(y_true == i)[0]
        if len(idxs) > 0:
            # Promedio de las probabilidades predichas para estas muestras
            prob_matrix[i] = np.mean(y_probs[idxs], axis=0)
            
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plotear heatmap
    # Usamos 'viridis' o 'magma' para resaltar probabilidades altas
    sns.heatmap(prob_matrix, cmap='viridis', vmin=0, vmax=1, ax=ax)
    
    ax.set_title(title)
    ax.set_ylabel('Clase Verdadera')
    ax.set_xlabel('Clase Predicha (Probabilidad Promedio)')
    
    # Manejo de etiquetas
    if n_classes > 40:
        # Ocultar etiquetas si son demasiadas
        ax.set_xticks([])
        ax.set_yticks([])
        st.caption(f"⚠️ Se han ocultado las etiquetas porque hay demasiadas clases ({n_classes}).")
    else:
        ax.set_xticks(np.arange(n_classes) + 0.5)
        ax.set_yticks(np.arange(n_classes) + 0.5)
        ax.set_xticklabels(classes, rotation=90, fontsize=8)
        ax.set_yticklabels(classes, rotation=0, fontsize=8)
        
    st.pyplot(fig)

def display_classification_report(y_true, y_pred, classes):
    """
    Muestra el reporte de clasificación de Scikit-Learn en un formato limpio.
    """
    # Asegurar que se reporten todas las clases, incluso si no están presentes en el batch actual
    labels = list(range(len(classes)))
    report = classification_report(y_true, y_pred, labels=labels, target_names=classes, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()
    
    st.subheader("📋 Reporte de Clasificación Detallado")
    
    # Formatear para que se vea bonito
    st.dataframe(
        df_report.style.background_gradient(cmap='viridis', subset=['precision', 'recall', 'f1-score'])
                 .format("{:.2%}", subset=['precision', 'recall', 'f1-score'])
    )

def plot_latent_space_pca(embeddings, labels, classes, title="Espacio Latente (PCA 3D)"):
    """
    Reduce la dimensionalidad de los embeddings a 3D usando PCA y los visualiza interactivamente.
    """
    from sklearn.decomposition import PCA
    import plotly.express as px
    
    # Convertir a numpy
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    # Mapear índices a nombres de clases
    label_names = [classes[i] for i in labels]
    
    # PCA 3D
    pca = PCA(n_components=3)
    components = pca.fit_transform(embeddings)
    
    # DataFrame para Plotly
    df = pd.DataFrame(components, columns=['PC1', 'PC2', 'PC3'])
    df['Clase'] = label_names
    
    fig = px.scatter_3d(
        df, x='PC1', y='PC2', z='PC3',
        color='Clase',
        title=title,
        hover_data=['Clase'],
        opacity=0.7
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig, width="stretch")

def plot_dtw_alignment(pred_audio, target_audio, title="Sincronía Temporal (DTW Alignment)"):
    """
    Visualiza la alineación temporal entre las señales predichas y reales.
    """
    import matplotlib.pyplot as plt
    
    # Convertir tensores a numpy si es necesario
    if hasattr(pred_audio, "cpu"):
        pred_audio = pred_audio.detach().cpu().numpy()
    if hasattr(target_audio, "cpu"):
        target_audio = target_audio.detach().cpu().numpy()
        
    # Si son 2D (ej. spectrograms o embeddings promediados), aplanar o promediar
    if pred_audio.ndim > 1:
        pred_audio = pred_audio.mean(axis=-1)
    if target_audio.ndim > 1:
        target_audio = target_audio.mean(axis=-1)

    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Normalizar para visualización comparativa
    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)
    
    p = norm(pred_audio)
    t = norm(target_audio)
    
    ax.plot(t, label="Real (Target)", color="#6B66FF", alpha=0.6, linewidth=2)
    ax.plot(p, label="Predicho (Sintetizado)", color="#FF66B2", alpha=0.9, linewidth=1.5)
    
    ax.set_title(title, fontsize=12, fontweight='bold', color='white')
    ax.legend(facecolor='#1e1e1e', edgecolor='none', labelcolor='white')
    ax.set_facecolor('#0e1117')
    fig.patch.set_facecolor('#0e1117')
    ax.tick_params(colors='white')
    ax.grid(alpha=0.1, color='white')
    
    plt.tight_layout()
    return fig
