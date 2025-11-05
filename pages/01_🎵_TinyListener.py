"""
🎵 TinyListener - Entrenamiento y analítica sobre el dataset de audio actual.
"""

from __future__ import annotations

import copy
import os
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import streamlit as st
import torch
from matplotlib import pyplot as plt

from components.modern_sidebar import display_modern_sidebar
from models import TinyListener, TinySpeak
from training.audio_dataset import (
    DEFAULT_AUDIO_SPLIT_RATIOS,
    AudioWordDataset,
    build_audio_dataloaders,
    load_audio_splits,
)
from training.audio_module import TinyListenerLightning
from training.config import load_master_dataset_config
from utils import (
    WAV2VEC_DIM,
    WAV2VEC_SR,
    encontrar_device,
    get_default_words,
    load_waveform,
    plot_logits_native,
    plot_waveform_native,
    save_waveform_to_audio_file,
    synthesize_word,
)


DEFAULT_SEED = 42


@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    num_workers: int
    seed: int
    freeze_wav2vec: bool
    accelerator: str


class HistoryCallback(pl.Callback):
    """Registra métricas por época para graficarlas en Streamlit."""

    def __init__(self) -> None:
        super().__init__()
        self.train_history: List[Dict] = []
        self.val_history: List[Dict] = []
        self.learning_rates: List[Dict] = []

    @staticmethod
    def _to_float(value: torch.Tensor | float | int) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = {"epoch": trainer.current_epoch}
        for key in ("train_loss", "train_top1", "train_top3", "train_top5"):
            if key in trainer.callback_metrics:
                metrics[key] = self._to_float(trainer.callback_metrics[key])
        if trainer.optimizers:
            lr = float(trainer.optimizers[0].param_groups[0]["lr"])
            metrics["lr"] = lr
            self.learning_rates.append({"epoch": trainer.current_epoch, "lr": lr})
        self.train_history.append(metrics)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = {"epoch": trainer.current_epoch}
        for key in ("val_loss", "val_top1", "val_top3", "val_top5"):
            if key in trainer.callback_metrics:
                metrics[key] = self._to_float(trainer.callback_metrics[key])
        self.val_history.append(metrics)


def _resolve_accelerator(accelerator: str) -> Tuple[str, int | str | None]:
    if accelerator == "cpu":
        return "cpu", 1
    if accelerator == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("No hay GPU disponible para entrenamiento.")
        return "gpu", 1
    return "auto", "auto"


def sanitize_metrics(metrics: Dict) -> Dict:
    clean: Dict = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            clean[key] = float(value.detach().cpu().item())
        elif isinstance(value, (np.generic, np.ndarray)):
            clean[key] = float(np.asarray(value).item())
        else:
            clean[key] = float(value) if isinstance(value, (int, float)) else value
    return clean


def compute_confusion_matrix(predictions: List[Dict], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    if not predictions:
        return matrix
    for batch in predictions:
        true_labels = batch.get("true_labels", [])
        top_indices = batch.get("top_indices", [])
        for y_true, candidates in zip(true_labels, top_indices):
            if not candidates:
                continue
            predicted = candidates[0]
            matrix[y_true, predicted] += 1
    return matrix


def compute_per_class_accuracy(matrix: np.ndarray) -> List[float]:
    totals = matrix.sum(axis=1)
    correct = np.diag(matrix)
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.divide(correct, totals, out=np.zeros_like(correct, dtype=float), where=totals > 0)
    return acc.tolist()


def extract_misclassifications(
    predictions: List[Dict],
    class_names: Iterable[str],
    limit: int = 6,
) -> List[Dict]:
    examples: List[Dict] = []
    classes = list(class_names)
    if not predictions:
        return examples
    for batch in predictions:
        words = batch.get("words", [])
        true_labels = batch.get("true_labels", [])
        top_indices = batch.get("top_indices", [])
        top_scores = batch.get("top_scores", [])
        for word, y_true, preds, scores in zip(words, true_labels, top_indices, top_scores):
            if not preds:
                continue
            predicted_idx = preds[0]
            if predicted_idx == y_true:
                continue
            examples.append(
                {
                    "palabra": word,
                    "prediccion": classes[predicted_idx],
                    "confianza": float(scores[0]),
                    "top5": ", ".join(classes[i] for i in preds[:5]),
                }
            )
    examples.sort(key=lambda item: item["confianza"], reverse=True)
    return examples[:limit]


def plot_confusion(matrix: np.ndarray, class_names: List[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_ylabel("Etiqueta real")
    ax.set_xlabel("Predicción")
    ax.set_title("Matriz de confusión")
    thresh = matrix.max() / 2 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if value == 0:
                continue
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > thresh else "black",
                fontsize=10,
            )
    fig.tight_layout()
    return fig


def compute_distribution(splits: Dict[str, AudioWordDataset]) -> pd.DataFrame:
    words = splits["train"].words
    data = {}
    for split_name, dataset in splits.items():
        counts = {word: 0 for word in words}
        for sample in dataset.samples:
            counts[sample.word] += 1
        data[split_name] = [counts[word] for word in words]
    df = pd.DataFrame(data, index=words)
    df["total"] = df.sum(axis=1)
    return df


def count_parameters(num_classes: int) -> Dict[str, int]:
    temp_model = TinySpeak(num_classes=num_classes)
    total = sum(p.numel() for p in temp_model.parameters())
    classifier = sum(p.numel() for p in temp_model.classifier.parameters())
    lstm = total - classifier
    return {"total": total, "lstm": lstm, "classifier": classifier}


def _run_training_once(config: TrainingConfig) -> Dict:
    pl.seed_everything(config.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    train_ds, val_ds, test_ds, loaders = build_audio_dataloaders(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        split_ratios=DEFAULT_AUDIO_SPLIT_RATIOS,
    )

    module = TinyListenerLightning(
        class_names=train_ds.words,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        freeze_wav2vec=config.freeze_wav2vec,
    )

    history_cb = HistoryCallback()
    callbacks = [history_cb, pl.callbacks.LearningRateMonitor(logging_interval="epoch")]
    accelerator, devices = _resolve_accelerator(config.accelerator)
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=config.max_epochs,
        deterministic=True,
        enable_checkpointing=False,
        enable_model_summary=False,
        log_every_n_steps=1,
        enable_progress_bar=True,
        default_root_dir=str(Path.cwd() / "lightning_logs" / "tiny_listener"),
        callbacks=callbacks,
    )

    start = time.perf_counter()
    trainer.fit(module, train_dataloaders=loaders["train"], val_dataloaders=loaders["val"])
    val_metrics = trainer.validate(module, dataloaders=loaders["val"])
    test_metrics = trainer.test(module, dataloaders=loaders["test"])
    duration = time.perf_counter() - start

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    val_preds = copy.deepcopy(module.validation_predictions)
    test_preds = copy.deepcopy(module.test_predictions)
    confusion = compute_confusion_matrix(test_preds or val_preds, train_ds.num_classes)

    return {
        "config": asdict(config),
        "duration": duration,
        "train_history": history_cb.train_history,
        "val_history": history_cb.val_history,
        "learning_rates": history_cb.learning_rates,
        "val_metrics": sanitize_metrics(val_metrics[0] if val_metrics else {}),
        "test_metrics": sanitize_metrics(test_metrics[0] if test_metrics else {}),
        "confusion_matrix": confusion.tolist(),
        "per_class_accuracy": compute_per_class_accuracy(confusion),
        "misclassifications": extract_misclassifications(test_preds or val_preds, train_ds.words),
        "class_names": train_ds.words,
        "num_classes": train_ds.num_classes,
        "total_train_samples": len(train_ds),
        "total_val_samples": len(val_ds),
        "total_test_samples": len(test_ds),
        "timestamp": time.time(),
        "accelerator": config.accelerator,
    }


def run_training(config: TrainingConfig) -> Dict:
    try:
        return _run_training_once(config)
    except RuntimeError as exc:
        error_msg = str(exc).lower()
        if "no kernel image" in error_msg and config.accelerator != "cpu":
            cpu_config = replace(config, accelerator="cpu")
            result = _run_training_once(cpu_config)
            result["fallback_reason"] = "cuda_kernel_missing"
            return result
        raise


def _get_current_words() -> List[str]:
    try:
        config = load_master_dataset_config()
        selected = config.get("diccionario_seleccionado", {}) or {}
        words = selected.get("palabras") or []
        if isinstance(words, list) and words:
            return list(dict.fromkeys(words))
    except Exception:  # noqa: BLE001
        pass
    return get_default_words()


def render_dataset_tab(splits: Dict[str, AudioWordDataset], *, device: str) -> None:
    st.subheader("📚 Dataset de audio disponible")
    totals = {name: len(ds) for name, ds in splits.items()}
    vocab_size = splits["train"].num_classes

    metric_cols = st.columns(4)
    metric_cols[0].metric("Palabras", vocab_size)
    metric_cols[1].metric("Samples train", totals.get("train", 0))
    metric_cols[2].metric("Samples val", totals.get("val", 0))
    metric_cols[3].metric("Samples test", totals.get("test", 0))
    st.caption(f"Dispositivo activo: {device}")

    distribution = compute_distribution(splits)
    st.markdown("#### Distribución por palabra")
    st.dataframe(distribution, use_container_width=True)

    st.markdown("#### Escucha rápida de muestras (train)")
    words = distribution.index.tolist()
    if words:
        selected_word = st.selectbox("Palabra", options=words, index=0)
        if selected_word:
            previews = [s for s in splits["train"].samples if s.word == selected_word][:3]
            if previews:
                preview_cols = st.columns(len(previews))
                for col, sample in zip(preview_cols, previews):
                    with col:
                        st.caption(sample.metadata.get("tipo", "—"))
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                                if save_waveform_to_audio_file(sample.waveform, tmp_file.name, WAV2VEC_SR):
                                    with open(tmp_file.name, "rb") as audio_file:
                                        st.audio(audio_file.read(), format="audio/wav")
                        finally:
                            if "tmp_file" in locals() and os.path.exists(tmp_file.name):  # type: ignore[attr-defined]
                                os.unlink(tmp_file.name)
            else:
                st.info("No hay muestras en train para esta palabra.")

    st.markdown("#### Arquitectura TinySpeak adaptada")
    params = count_parameters(vocab_size)
    info_cols = st.columns(3)
    info_cols[0].metric("Parámetros totales", f"{params['total']:,}")
    info_cols[1].metric("LSTM", f"{params['lstm']:,}")
    info_cols[2].metric("Clasificador", f"{params['classifier']:,}")

    st.code(
        f"""
Audio → facebook/wav2vec2-base-es-voxpopuli-v2 (congelado)
Downsampling ×7 · salida {WAV2VEC_DIM} dim
TinySpeak LSTM(hidden=128, layers=2) → Linear(128→{vocab_size})
        """,
        language="text",
    )


def render_training_tab(class_names: List[str]) -> None:
    st.subheader("⚙️ Entrenamiento de TinyListener")
    st.write("Configura los hiperparámetros y entrena contra el dataset actual.")

    with st.form("tiny_listener_training"):
        col1, col2, col3 = st.columns(3)
        batch_size = col1.number_input("Batch size", min_value=4, max_value=128, value=16, step=4)
        learning_rate = col2.number_input("Learning rate", min_value=1e-6, max_value=1e-2, value=1e-3, format="%.1e")
        weight_decay = col3.number_input("Weight decay", min_value=0.0, max_value=1e-1, value=1e-4, format="%.1e")

        col4, col5, col6 = st.columns(3)
        max_epochs = int(col4.number_input("Épocas", min_value=1, max_value=200, value=20))
        num_workers = int(col5.selectbox("Workers", options=[0, 1, 2, 4, 8], index=0))
        seed = int(col6.number_input("Seed", min_value=0, max_value=10_000, value=DEFAULT_SEED))

        accelerator = st.selectbox(
            "Dispositivo",
            options=["auto", "cpu", "gpu"],
            index=0,
            help="Usa 'cpu' si tu GPU da errores de compatibilidad.",
        )

        freeze_wav2vec = st.toggle(
            "Congelar Wav2Vec2",
            value=True,
            help="Recomendado para entrenamientos rápidos. Desactiva para fine-tuning (muy pesado).",
        )

        submitted = st.form_submit_button("Entrenar TinyListener", type="primary")

    if submitted:
        if len(class_names) <= 1:
            st.error("Se requiere al menos 2 clases para entrenar TinyListener.")
            return
        config = TrainingConfig(
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            num_workers=num_workers,
            seed=seed,
            freeze_wav2vec=freeze_wav2vec,
            accelerator=accelerator,
        )
        with st.spinner("Ejecutando entrenamiento..."):
            try:
                result = run_training(config)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Entrenamiento falló: {exc}")
                return
        st.success("Entrenamiento completado")
        if result.get("fallback_reason") == "cuda_kernel_missing":
            st.warning(
                "Se detectó un error de CUDA. El entrenamiento se reintentó automáticamente en CPU."
            )
        st.caption(f"Acelerador usado: {result.get('accelerator', accelerator)}")
        st.session_state["listener_training_result"] = result
        st.json(result["config"])


def render_analytics_tab() -> None:
    result = st.session_state.get("listener_training_result")
    if not result:
        st.info("Aún no hay corridas de entrenamiento registradas.")
        return

    st.subheader("📈 Analíticas de TinyListener")
    duration = result.get("duration")
    if duration is not None:
        st.caption(f"Ejecución total: {duration:.1f} s")

    metrics_cols = st.columns(3)
    val_metrics = result.get("val_metrics", {})
    test_metrics = result.get("test_metrics", {})
    metrics_cols[0].metric("Val loss", f"{val_metrics.get('val_loss', float('nan')):.4f}" if val_metrics else "—")
    metrics_cols[1].metric(
        "Val top-1",
        f"{val_metrics.get('val_top1', float('nan')):.2f}%" if val_metrics else "—",
    )
    metrics_cols[2].metric(
        "Test top-1",
        f"{test_metrics.get('test_top1', float('nan')):.2f}%" if test_metrics else "—",
    )

    train_history = result.get("train_history", [])
    val_history = result.get("val_history", [])
    if train_history or val_history:
        st.markdown("#### Curvas de entrenamiento")
        history_frames = []
        if train_history:
            df_train = pd.DataFrame(train_history)
            df_train["split"] = "train"
            history_frames.append(df_train)
        if val_history:
            df_val = pd.DataFrame(val_history)
            df_val["split"] = "val"
            history_frames.append(df_val)
        history_df = pd.concat(history_frames, ignore_index=True)
        loss_cols = [col for col in history_df.columns if col.endswith("loss")]
        if loss_cols:
            loss_df = history_df.melt(
                id_vars=["epoch", "split"],
                value_vars=loss_cols,
                var_name="metric",
                value_name="value",
            ).dropna(subset=["value"])
            if not loss_df.empty:
                chart_data = loss_df.pivot_table(index="epoch", columns="metric", values="value")
                st.line_chart(chart_data)

        acc_cols = [col for col in history_df.columns if col.endswith("top1")]
        if acc_cols:
            acc_df = history_df.melt(
                id_vars=["epoch", "split"],
                value_vars=acc_cols,
                var_name="metric",
                value_name="value",
            ).dropna(subset=["value"])
            if not acc_df.empty:
                chart_data = acc_df.pivot_table(index="epoch", columns="metric", values="value")
                st.line_chart(chart_data)

    matrix = np.asarray(result.get("confusion_matrix", []))
    class_names = result.get("class_names", [])
    if matrix.size and class_names:
        st.markdown("#### Matriz de confusión")
        fig = plot_confusion(matrix, class_names)
        st.pyplot(fig)

        per_class = result.get("per_class_accuracy", [])
        if per_class:
            acc_df = pd.DataFrame(
                {
                    "palabra": class_names,
                    "accuracy": per_class,
                    "muestras": matrix.sum(axis=1).tolist(),
                }
            )
            acc_df["accuracy_pct"] = acc_df["accuracy"] * 100
            st.dataframe(acc_df, use_container_width=True)

    errors = result.get("misclassifications", [])
    if errors:
        st.markdown("#### Errores más confiados")
        st.table(pd.DataFrame(errors))

# ------------------------------------------------------------------
# Laboratorio interactivo (hereda funcionalidad previa)
# ------------------------------------------------------------------


@st.cache_resource
def load_audio_models():
    """Cargar TinySpeak/TinyListener con el vocabulario actual."""
    preferred_device = encontrar_device()
    device = preferred_device
    words = _get_current_words()

    wav2vec_model = TinyListenerLightning(class_names=words).wav2vec_model  # reuse weights download
    tiny_speak = TinySpeak(words=words, hidden_dim=128, num_layers=2, wav2vec_dim=WAV2VEC_DIM)
    tiny_listener = TinyListener(tiny_speak=tiny_speak, wav2vec_model=wav2vec_model)

    fallback_reason = None
    try:
        tiny_speak = tiny_speak.to(device)
        tiny_listener = tiny_listener.to(device)
        if hasattr(wav2vec_model, "to"):
            wav2vec_model = wav2vec_model.to(device)
    except RuntimeError as exc:
        error_message = str(exc).lower()
        is_cuda_device = isinstance(device, torch.device) and device.type == "cuda"
        if is_cuda_device and "no kernel image" in error_message:
            fallback_reason = "cuda_kernel_missing"
        elif is_cuda_device and "cuda" in error_message:
            fallback_reason = "cuda_move_failed"

        if fallback_reason:
            st.warning(
                "⚠️ No fue posible inicializar TinyListener en CUDA. Se realizará un fallback automático a CPU."
            )
            device = torch.device("cpu")
            tiny_speak = tiny_speak.cpu()
            tiny_listener = tiny_listener.cpu()
            if hasattr(wav2vec_model, "cpu"):
                wav2vec_model = wav2vec_model.cpu()
        else:
            raise

    return {
        "device": device,
        "preferred_device": preferred_device,
        "wav2vec_model": wav2vec_model,
        "tiny_speak": tiny_speak,
        "tiny_listener": tiny_listener,
        "words": words,
        "fallback_reason": fallback_reason,
    }


def _display_prediction_results(logits, models, waveforms=None):
    col1, col2 = st.columns([1, 1])

    with col1:
        predicted_idx = logits.argmax(dim=1).item()
        predicted_word = models["words"][predicted_idx]
        confidence = torch.softmax(logits, dim=1).max().item()
        st.metric("🎯 Predicción", predicted_word, help=f"Confianza: {confidence:.2%}")

        probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        top_indices = np.argsort(probabilities)[::-1][:5]
        st.markdown("**🏆 Top 5:**")
        for i, idx in enumerate(top_indices):
            word = models["words"][idx]
            prob = probabilities[idx]
            st.write(f"{i+1}. **{word}** ({prob:.2%})")

    with col2:
        fig = plot_logits_native(logits, models["words"], "Distribución de Predicciones")
        st.plotly_chart(fig, use_container_width=True)

    if waveforms is not None:
        fig = plot_waveform_native(waveforms, "Waveform analizado", sample_rate=WAV2VEC_SR)
        st.plotly_chart(fig, use_container_width=True)


def _analyze_audio_file(audio_bytes, models):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        waveform = load_waveform(tmp_path, target_sr=WAV2VEC_SR)
        os.unlink(tmp_path)

        if waveform is None:
            st.error("❌ Error al cargar el archivo de audio")
            return

        device = models["device"]
        waveform_device = waveform.unsqueeze(0).to(device)
        models["tiny_listener"].eval()
        with torch.no_grad():
            logits, _ = models["tiny_listener"]([waveform_device.squeeze(0)])
        _display_prediction_results(logits, models, waveform)
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ Error procesando audio: {exc}")


def _synthesize_and_analyze(text, rate, pitch, amplitude, models):
    try:
        waveform = synthesize_word(text, rate=rate, pitch=pitch, amplitude=amplitude)
        if waveform is None:
            st.error("❌ Error generando audio")
            return
        device = models["device"]
        waveform_device = waveform.to(device)
        models["tiny_listener"].eval()
        with torch.no_grad():
            logits, _ = models["tiny_listener"]([waveform_device])
        st.session_state["listener_synthesis_results"] = {
            "text": text,
            "waveform": waveform,
            "rate": rate,
            "pitch": pitch,
            "amplitude": amplitude,
            "logits": logits,
        }
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ Error en síntesis: {exc}")


def _render_synthesis_results(models):
    results = st.session_state.get("listener_synthesis_results")
    if not results:
        return

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            if save_waveform_to_audio_file(results["waveform"], tmp_file.name, WAV2VEC_SR):
                with open(tmp_file.name, "rb") as audio_file:
                    st.audio(audio_file.read(), format="audio/wav")
    finally:
        if "tmp_file" in locals() and os.path.exists(tmp_file.name):  # type: ignore[attr-defined]
            os.unlink(tmp_file.name)

    _display_prediction_results(results["logits"], models, results["waveform"])


def render_lab_tab(models: Dict) -> None:
    st.subheader("🧪 Laboratorio de TinyListener")
    tab1, tab2, tab3 = st.tabs(["📁 Subir Audio", "🎤 Grabar", "🔊 Sintetizar"])

    with tab1:
        st.markdown("#### 📁 Test con archivo")
        audio_file = st.file_uploader(
            "Sube un archivo de audio",
            type=["wav", "mp3", "flac", "m4a"],
        )
        if audio_file is not None:
            st.audio(audio_file)
            if st.button("🔍 Analizar archivo", key="analyze_upload", type="primary"):
                _analyze_audio_file(audio_file.read(), models)

    with tab2:
        st.markdown("#### 🎤 Test con grabación")
        recorded_audio = st.audio_input("Graba tu voz")
        if recorded_audio is not None:
            st.audio(recorded_audio)
            if st.button("🔍 Analizar grabación", key="analyze_record", type="primary"):
                _analyze_audio_file(recorded_audio.read(), models)

    with tab3:
        st.markdown("#### 🔊 Síntesis → Reconocimiento")
        word_options = ["Palabra personalizada"] + models["words"][:50]
        selected_option = st.selectbox("Selecciona una palabra", word_options)
        if selected_option == "Palabra personalizada":
            text_input = st.text_input("Escribe una palabra", value="hola")
        else:
            text_input = selected_option

        col_rate, col_pitch, col_amp = st.columns(3)
        with col_rate:
            rate = st.slider("Velocidad", 50, 200, 80)
        with col_pitch:
            pitch = st.slider("Tono", 0, 100, 50)
        with col_amp:
            amplitude = st.slider("Volumen", 50, 200, 120)

        if st.button("🎵 Sintetizar y analizar", type="primary", key="synthesize_and_run"):
            with st.spinner("Sintetizando y analizando..."):
                _synthesize_and_analyze(text_input, rate, pitch, amplitude, models)
                st.rerun()

        _render_synthesis_results(models)


def main() -> None:
    st.set_page_config(page_title="TinyListener", page_icon="🎵", layout="wide")
    display_modern_sidebar()
    st.title("🎵 TinyListener")

    device = str(encontrar_device())
    splits, error = load_audio_splits(seed=DEFAULT_SEED)
    if error or not splits:
        st.error(
            "No se pudo cargar el dataset de audio. Genera muestras desde `🎤 Audio Dataset Manager` y vuelve a intentarlo.\n\n"
            f"Detalle: {error}"
        )
        return

    st.session_state["listener_class_names"] = splits["train"].words

    dataset_tab, training_tab, analytics_tab, lab_tab = st.tabs(
        ["📚 Dataset & Modelo", "⚙️ Entrenamiento", "📈 Analíticas", "🧪 Laboratorio"]
    )

    with dataset_tab:
        render_dataset_tab(splits, device=device)
    with training_tab:
        render_training_tab(splits["train"].words)
    with analytics_tab:
        render_analytics_tab()
    with lab_tab:
        models = st.session_state.get("listener_models")
        if models is None:
            with st.spinner("Cargando modelos de audio..."):
                st.session_state["listener_models"] = load_audio_models()
            models = st.session_state["listener_models"]
        render_lab_tab(models)


if __name__ == "__main__":
    main()
