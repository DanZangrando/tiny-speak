import pytorch_lightning as pl
from pathlib import Path
import torch
import pandas as pd
from datetime import datetime
import time
import json
from typing import Dict, Any, List, Optional, Tuple

from training.audio_dataset import build_audio_dataloaders
from training.visual_dataset import build_visual_dataloaders
from training.audio_module import PhonologicalPathwayLightning
from training.visual_module import VisualPathwayLightning
from training.reader_module import TinyReaderLightning
from utils import save_model_metadata, RealTimePlotCallback, ReaderPredictionCallback

class TrainingHistoryCallback(pl.Callback):
    def __init__(self):
        self.history = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v 
                  for k, v in trainer.callback_metrics.items()}
        metrics['epoch'] = trainer.current_epoch
        self.history.append(metrics)

def train_listener(
    language: str, 
    config: Dict[str, Any], 
    progress_callback=None,
    plot_placeholders=None
) -> Tuple[str, List[Dict]]:
    """
    Entrena un PhonologicalPathway (Listener) para un idioma específico.
    Retorna: (path_checkpoint, historial_metricas)
    """
    epochs = config.get('epochs', 10)
    lr = config.get('lr', 1e-3)
    batch_size = config.get('batch_size', 32)
    
    # 1. Data
    train_ds, val_ds, test_ds, loaders = build_audio_dataloaders(
        batch_size=batch_size, 
        num_workers=0, 
        seed=42,
        target_language=language
    )
    words = train_ds.class_names
    
    if not words:
        raise ValueError(f"No hay palabras para el idioma {language}")

    # 2. Model
    model = PhonologicalPathwayLightning(
        class_names=words,
        learning_rate=lr
    )
    
    # 3. Trainer
    history_cb = TrainingHistoryCallback()
    callbacks = [history_cb]
    
    if plot_placeholders:
        callbacks.append(RealTimePlotCallback(*plot_placeholders))

    # Callbacks
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=config.get('min_delta', 0.0),
        patience=config.get('patience', 10),
        verbose=True,
        mode="min"
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"experiments/models/listener/{language}",
        filename="listener-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks + [early_stop_callback, checkpoint_callback],
        enable_progress_bar=False, # Desactivamos la barra de PL para usar la nuestra si es necesario
        default_root_dir=f"lightning_logs/experiment_listener_{language}"
    )
    
    # 4. Train
    if progress_callback:
        progress_callback(0, f"Iniciando entrenamiento Listener ({language})...")
        
    trainer.fit(model, train_dataloaders=loaders['train'], val_dataloaders=loaders['val'])
    
    # 5. Save
    save_dir = Path("experiments/models/listener")
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    final_path = save_dir / f"listener_{language}_{timestamp}.ckpt"
    trainer.save_checkpoint(final_path)
    
    # Metadata
    meta_config = {
        "epochs": epochs, "lr": lr, "batch_size": batch_size,
        "vocab": words, "language": language, "type": "listener"
    }
    final_metrics = history_cb.history[-1] if history_cb.history else {}
    save_model_metadata(final_path, meta_config, final_metrics)
    
    return str(final_path), history_cb.history

def train_recognizer(
    language: str, 
    config: Dict[str, Any],
    progress_callback=None,
    plot_placeholders=None
) -> Tuple[str, List[Dict]]:
    """
    Entrena un VisualPathway (Recognizer) para un idioma específico.
    """
    epochs = config.get('epochs', 10)
    lr = config.get('lr', 1e-3)
    batch_size = config.get('batch_size', 32)
    
    # 1. Data
    train_ds, val_ds, test_ds, loaders = build_visual_dataloaders(
        batch_size=batch_size, 
        num_workers=0, 
        seed=42,
        target_language=language
    )
    class_names = train_ds.letters
    
    # 2. Model
    model = VisualPathwayLightning(
        num_classes=len(class_names),
        learning_rate=lr
    )
    
    # 3. Trainer
    history_cb = TrainingHistoryCallback()
    callbacks = [history_cb]
    
    if plot_placeholders:
        callbacks.append(RealTimePlotCallback(*plot_placeholders))

    # Callbacks
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=config.get('min_delta', 0.0),
        patience=config.get('patience', 10),
        verbose=True,
        mode="min"
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"experiments/models/recognizer/{language}",
        filename="recognizer-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks + [early_stop_callback, checkpoint_callback],
        enable_progress_bar=False,
        default_root_dir=f"lightning_logs/experiment_recognizer_{language}"
    )
    
    # 4. Train
    if progress_callback:
        progress_callback(0, f"Iniciando entrenamiento Recognizer ({language})...")

    trainer.fit(model, train_dataloaders=loaders['train'], val_dataloaders=loaders['val'])
    
    # 5. Save
    save_dir = Path("experiments/models/recognizer")
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    final_path = save_dir / f"recognizer_{language}_{timestamp}.ckpt"
    trainer.save_checkpoint(final_path)
    
    # Metadata
    meta_config = {
        "epochs": epochs, "lr": lr, "batch_size": batch_size,
        "classes": class_names, "language": language, "type": "recognizer"
    }
    final_metrics = history_cb.history[-1] if history_cb.history else {}
    save_model_metadata(final_path, meta_config, final_metrics)
    
    return str(final_path), history_cb.history

def train_reader(
    language: str,
    listener_ckpt: str,
    recognizer_ckpt: str,
    config: Dict[str, Any],
    progress_callback=None,
    plot_placeholders=None,
    prediction_placeholder=None
) -> Tuple[str, List[Dict]]:
    """
    Entrena un TinyReader.
    """
    epochs = config.get('epochs', 20)
    lr = config.get('lr', 1e-3)
    batch_size = config.get('batch_size', 32)
    w_mse = config.get('w_mse', 1.0)
    w_cos = config.get('w_cos', 1.0)
    w_perceptual = config.get('w_perceptual', 0.1)
    
    # 1. Data (Audio dataloaders provide the words/concepts)
    train_ds, val_ds, test_ds, loaders = build_audio_dataloaders(
        batch_size=batch_size, 
        num_workers=0, 
        seed=42,
        target_language=language
    )
    words = train_ds.class_names
    
    # 2. Model
    model = TinyReaderLightning(
        class_names=words,
        listener_checkpoint_path=listener_ckpt,
        recognizer_checkpoint_path=recognizer_ckpt,
        learning_rate=lr,
        w_mse=w_mse,
        w_cos=w_cos,
        w_perceptual=w_perceptual
    )
    
    # 3. Trainer
    history_cb = TrainingHistoryCallback()
    callbacks = [history_cb]
    
    if plot_placeholders:
        callbacks.append(RealTimePlotCallback(*plot_placeholders))
        
    if prediction_placeholder:
        callbacks.append(ReaderPredictionCallback(loaders['val'], prediction_placeholder))

    # Callbacks
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=config.get('min_delta', 0.0),
        patience=config.get('patience', 10),
        verbose=True,
        mode="min"
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"experiments/models/reader/{language}",
        filename="reader-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks + [early_stop_callback, checkpoint_callback],
        enable_progress_bar=False,
        default_root_dir=f"lightning_logs/experiment_reader_{language}"
    )
    
    # 4. Train
    if progress_callback:
        progress_callback(0, f"Iniciando entrenamiento Reader ({language})...")
        
    trainer.fit(model, train_dataloaders=loaders['train'], val_dataloaders=loaders['val'])
    
    # 5. Save
    save_dir = Path("experiments/models/reader")
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    final_path = save_dir / f"reader_{language}_{timestamp}.ckpt"
    trainer.save_checkpoint(final_path)
    
    # Metadata
    meta_config = {
        "epochs": epochs, "lr": lr, "batch_size": batch_size,
        "weights": {"mse": w_mse, "cos": w_cos, "perceptual": w_perceptual},
        "listener_ckpt": listener_ckpt,
        "recognizer_ckpt": recognizer_ckpt,
        "language": language,
        "vocab": words,
        "type": "reader"
    }
    final_metrics = history_cb.history[-1] if history_cb.history else {}
    save_model_metadata(final_path, meta_config, final_metrics)
    
    return str(final_path), history_cb.history
