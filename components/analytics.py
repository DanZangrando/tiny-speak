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
        st.plotly_chart(fig_loss, use_container_width=True)

    # Accuracy Chart
    acc_cols = [c for c in history_df.columns if 'top1' in c or 'acc' in c]
    if acc_cols:
        st.subheader("📈 Curvas de Precisión (Accuracy)")
        fig_acc = px.line(history_df, y=acc_cols, markers=True, title="Evolución de la Precisión")
        fig_acc.update_layout(xaxis_title="Época", yaxis_title="Accuracy (%)", hovermode="x unified")
        st.plotly_chart(fig_acc, use_container_width=True)

def plot_confusion_matrix(y_true, y_pred, classes, title="Matriz de Confusión"):
    """
    Genera y muestra una matriz de confusión visualmente atractiva.
    """
    # Asegurar que la matriz se calcule para TODAS las clases en el orden correcto (0..N-1)
    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Normalizar para el color (opcional, pero útil)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    
    st.pyplot(fig)
    
    return cm

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
    st.plotly_chart(fig, use_container_width=True)
