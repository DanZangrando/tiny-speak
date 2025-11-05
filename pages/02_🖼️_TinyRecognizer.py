"""
🖼️ TinyRecognizer - Entrenamiento y analítica sobre el dataset visual actual.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import streamlit as st
import torch
from PIL import Image
from matplotlib import pyplot as plt

from components.modern_sidebar import display_modern_sidebar
from models import TinyRecognizer
from training.visual_dataset import DEFAULT_SPLIT_RATIOS, VisualLetterDataset, build_visual_dataloaders
from training.visual_module import TinyRecognizerLightning
from utils import WAV2VEC_DIM, encontrar_device


DEFAULT_SEED = 42


@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    num_workers: int
    seed: int
    freeze_backbone: bool
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


def load_visual_splits(seed: int = DEFAULT_SEED) -> Tuple[Dict[str, VisualLetterDataset] | None, str | None]:
    ratios = DEFAULT_SPLIT_RATIOS
    try:
        splits = {
            name: VisualLetterDataset(split=name, augment=False, seed=seed, split_ratios=ratios)
            for name in ("train", "val", "test")
        }
        return splits, None
    except ValueError as exc:
        return None, str(exc)


def count_parameters(model: TinyRecognizer) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    classifier = sum(p.numel() for p in model.classifier.parameters())
    return total, classifier


def gather_preview_samples(dataset: VisualLetterDataset, max_images: int = 12) -> List:
    seen: set[str] = set()
    preview: List = []
    for sample in dataset.samples:
        if sample.letter not in seen:
            preview.append(sample)
            seen.add(sample.letter)
        if len(preview) >= max_images:
            break
    return preview


def compute_distribution(splits: Dict[str, VisualLetterDataset]) -> pd.DataFrame:
    letters = splits["train"].letters
    data = {}
    for split_name, dataset in splits.items():
        counts = {letter: 0 for letter in letters}
        for sample in dataset.samples:
            counts[sample.letter] += 1
        data[split_name] = [counts[letter] for letter in letters]
    df = pd.DataFrame(data, index=letters)
    df["total"] = df.sum(axis=1)
    return df


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
        letters = batch.get("letters", [])
        true_labels = batch.get("true_labels", [])
        top_indices = batch.get("top_indices", [])
        top_scores = batch.get("top_scores", [])
        for letter, y_true, preds, scores in zip(letters, true_labels, top_indices, top_scores):
            if not preds:
                continue
            predicted_idx = preds[0]
            if predicted_idx == y_true:
                continue
            examples.append(
                {
                    "letra": letter,
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


def _resolve_accelerator(accelerator: str) -> Tuple[str, int | str | None]:
    if accelerator == "cpu":
        return "cpu", 1
    if accelerator == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("No hay GPU disponible para entrenamiento.")
        return "gpu", 1
    # fallback to auto configuration
    return "auto", "auto"


def _run_training_once(config: TrainingConfig) -> Dict:
    pl.seed_everything(config.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    train_ds, val_ds, test_ds, loaders = build_visual_dataloaders(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        split_ratios=DEFAULT_SPLIT_RATIOS,
    )

    module = TinyRecognizerLightning(
        num_classes=train_ds.num_classes,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        freeze_backbone=config.freeze_backbone,
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
        default_root_dir=str(Path.cwd() / "lightning_logs" / "tiny_recognizer"),
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
        "misclassifications": extract_misclassifications(test_preds or val_preds, train_ds.letters),
        "class_names": train_ds.letters,
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


def render_dataset_tab(
    splits: Dict[str, VisualLetterDataset],
    *,
    device: str,
) -> None:
    st.subheader("📚 Dataset visual disponible")
    totals = {name: len(ds) for name, ds in splits.items()}
    num_classes = splits["train"].num_classes

    metric_cols = st.columns(4)
    metric_cols[0].metric("Clases", num_classes)
    metric_cols[1].metric("Imágenes train", totals.get("train", 0))
    metric_cols[2].metric("Imágenes val", totals.get("val", 0))
    metric_cols[3].metric("Imágenes test", totals.get("test", 0))
    st.caption(f"Dispositivo activo: {device}")

    distribution = compute_distribution(splits)
    st.markdown("#### Distribución por letra")
    st.dataframe(distribution, use_container_width=True)

    train_samples = gather_preview_samples(splits["train"])
    if train_samples:
        st.markdown("#### Vista rápida del split de entrenamiento")
        preview_cols = st.columns(4)
        for idx, sample in enumerate(train_samples):
            with preview_cols[idx % 4]:
                with Image.open(sample.path) as img:
                    st.image(
                        img,
                        caption=f"{sample.letter} · {sample.path.name}",
                        use_container_width=True,
                    )

    st.markdown("#### Arquitectura TinyRecognizer adaptada")
    model = TinyRecognizer(num_classes=num_classes)
    total_params, classifier_params = count_parameters(model)
    backbone_params = total_params - classifier_params
    info_cols = st.columns(3)
    info_cols[0].metric("Parámetros totales", f"{total_params:,}")
    info_cols[1].metric("Backbone", f"{backbone_params:,}")
    info_cols[2].metric("Clasificador", f"{classifier_params:,}")

    st.code(
        f"""
Input: 64×64×3 RGB
V1 → Conv(3→64) + ReLU + MaxPool
V2 → Conv(64→128) + ReLU + MaxPool
V4 → Conv(128→256) + ReLU + MaxPool
IT → Conv(256→512) + ReLU + MaxPool
Decoder → AdaptiveAvgPool → Flatten → Linear(512→1024) → ReLU → Linear(1024→{WAV2VEC_DIM})
Classifier → Linear({WAV2VEC_DIM}→{num_classes})
        """,
        language="text",
    )


def render_training_tab(num_classes: int) -> None:
    st.subheader("⚙️ Entrenamiento de TinyRecognizer")
    st.write("Configura los hiperparámetros y entrena contra el dataset actual.")

    with st.form("tiny_recognizer_training"):
        col1, col2, col3 = st.columns(3)
        batch_size = col1.number_input("Batch size", min_value=4, max_value=256, value=32, step=4)
        learning_rate = col2.number_input("Learning rate", min_value=1e-6, max_value=1e-1, value=1e-3, format="%.1e")
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

        freeze_backbone = st.toggle(
            "Congelar CORnet-Z",
            value=False,
            help="Congela el backbone y entrena solo el clasificador.",
        )

        submitted = st.form_submit_button("Entrenar TinyRecognizer", type="primary")

    if submitted:
        config = TrainingConfig(
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            num_workers=num_workers,
            seed=seed,
            freeze_backbone=freeze_backbone,
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
                "Se detectó un error de CUDA. El entrenamiento se reintentó automáticamente en CPU."  # noqa: E501
            )
        st.caption(f"Acelerador usado: {result.get('accelerator', accelerator)}")
        st.session_state["recognizer_training_result"] = result
        st.json(result["config"])


def render_analytics_tab() -> None:
    result = st.session_state.get("recognizer_training_result")
    if not result:
        st.info("Aún no hay corridas de entrenamiento registradas.")
        return

    st.subheader("📈 Analíticas de TinyRecognizer")
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
                    "letra": class_names,
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


def main() -> None:
    st.set_page_config(page_title="TinyRecognizer", page_icon="🖼️", layout="wide")
    display_modern_sidebar()
    st.title("🖼️ TinyRecognizer")

    device = str(encontrar_device())
    splits, error = load_visual_splits()
    if error or not splits:
        st.error(
            "No se pudo cargar el dataset visual. Genera imágenes desde `🖼️ Visual Dataset Manager` y vuelve a intentarlo.\n\n"
            f"Detalle: {error}"
        )
        return

    st.session_state["recognizer_class_names"] = splits["train"].letters

    dataset_tab, training_tab, analytics_tab = st.tabs(
        ["📚 Dataset & Modelo", "⚙️ Entrenamiento", "📈 Analíticas"]
    )

    with dataset_tab:
        render_dataset_tab(splits, device=device)
    with training_tab:
        render_training_tab(splits["train"].num_classes)
    with analytics_tab:
        render_analytics_tab()


if __name__ == "__main__":
    main()
