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
    use_phonemes = config.get('use_phonemes', False)
    
    # 1. Data
    train_ds, val_ds, test_ds, loaders = build_audio_dataloaders(
        batch_size=batch_size, 
        num_workers=0, 
        seed=42,
        target_language=language,
        use_phonemes=use_phonemes
    )
    words = train_ds.class_names
    
    if not words:
        msg = f"No hay {'fonemas' if use_phonemes else 'palabras'} para el idioma {language}"
        raise ValueError(msg)

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

    # Subdirectorio diferente para fonemas si se desea, o mismo con prefijo
    sub_dir = "listener_phonemes" if use_phonemes else "listener"
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"experiments/models/{sub_dir}/{language}",
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
        enable_progress_bar=False, 
        default_root_dir=f"lightning_logs/experiment_{sub_dir}_{language}"
    )
    
    # 4. Train
    if progress_callback:
        type_str = "Fonemas" if use_phonemes else "Palabras"
        progress_callback(0, f"Iniciando entrenamiento Listener ({language}) [{type_str}]...")
        
    trainer.fit(model, train_dataloaders=loaders['train'], val_dataloaders=loaders['val'])
    
    # 5. Save / Return Best
    best_path = checkpoint_callback.best_model_path
    if best_path and Path(best_path).exists():
        print(f"Usando mejor modelo Listener: {best_path}")
        final_path = best_path
        
        # Guardar metadata
        meta_config = {
            "epochs": epochs, "lr": lr, "batch_size": batch_size,
            "vocab": words, "language": language, "type": "listener",
            "use_phonemes": use_phonemes
        }
        final_metrics = history_cb.history[-1] if history_cb.history else {}
        save_model_metadata(final_path, meta_config, final_metrics)
        
        return str(final_path), history_cb.history
    else:
        # Fallback
        save_dir = Path(f"experiments/models/{sub_dir}")
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        final_path = save_dir / f"listener_{language}_{timestamp}.ckpt"
        trainer.save_checkpoint(final_path)
        
        # Metadata
        meta_config = {
            "epochs": epochs, "lr": lr, "batch_size": batch_size,
            "vocab": words, "language": language, "type": "listener",
            "use_phonemes": use_phonemes
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
    
    # 5. Save / Return Best
    # Si tenemos un mejor modelo guardado por el callback, lo usamos.
    # Si no (ej. 1 epoca), usamos el final.
    best_path = checkpoint_callback.best_model_path
    if best_path and Path(best_path).exists():
        print(f"Usando mejor modelo Recognizer: {best_path}")
        final_path = best_path
        
        # Copiar metadata al mejor modelo también si es necesario, 
        # pero por ahora retornamos el path del mejor.
        # Para consistencia con el experimento, podríamos copiarlo a final_path?
        # Mejor retornamos el best_path y guardamos metadata asociada a él.
        
        # Guardar metadata para el mejor modelo
        meta_config = {
            "epochs": epochs, "lr": lr, "batch_size": batch_size,
            "classes": class_names, "language": language, "type": "recognizer"
        }
        final_metrics = history_cb.history[-1] if history_cb.history else {}
        # Intentar buscar métricas del mejor epoch? Es complicado sin parsear.
        # Usamos las últimas disponibles como proxy o las del callback si pudiéramos.
        save_model_metadata(final_path, meta_config, final_metrics)
        
        return str(final_path), history_cb.history
    else:
        # Fallback: Guardar el actual
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
    Entrena un TinyReader. Soporta fases 'g2p', 'p2w', 'end_to_end'.
    """
    epochs = config.get('epochs', 20)
    lr = config.get('lr', 1e-3)
    batch_size = config.get('batch_size', 32)
    w_dtw = config.get('w_dtw', 1.0)
    w_perceptual = config.get('w_perceptual', 0.1)
    
    use_two_stage = config.get('use_two_stage', False)
    phoneme_listener_ckpt = config.get('phoneme_listener_ckpt', None)
    training_phase = config.get('training_phase', 'end_to_end')
    pretrained_speller_ckpt = config.get('pretrained_speller_ckpt', None)
    
    # 1. Data
    train_ds, val_ds, test_ds, loaders = build_audio_dataloaders(
        batch_size=batch_size, 
        num_workers=0, 
        seed=42,
        target_language=language,
        use_phonemes=False # Reader always trains on words (even if G2P phase uses phoneme targets internally generated)
    )
    words = train_ds.class_names
    
    # 2. Model
    model = TinyReaderLightning(
        class_names=words,
        listener_checkpoint_path=listener_ckpt,
        recognizer_checkpoint_path=recognizer_ckpt,
        learning_rate=lr,
        w_dtw=w_dtw,
        w_perceptual=w_perceptual,
        use_two_stage=use_two_stage,
        phoneme_listener_checkpoint_path=phoneme_listener_ckpt,
        training_phase=training_phase
    )
    
    # Cargar pesos de Speller si se proporciona (para fase P2W)
    if pretrained_speller_ckpt and Path(pretrained_speller_ckpt).exists():
        print(f"Cargando pesos de Speller desde {pretrained_speller_ckpt}...")
        try:
            speller_ckpt = torch.load(pretrained_speller_ckpt, map_location=model.device)
            state_dict = speller_ckpt['state_dict']
            # Filtrar solo las keys de reader_g2p
            g2p_weights = {k: v for k, v in state_dict.items() if "reader_g2p" in k}
            if g2p_weights:
                model.load_state_dict(g2p_weights, strict=False)
                print(f"✅ Pesos de G2P cargados ({len(g2p_weights)} keys).")
            
            # También cargar phoneme_listener si está en el checkpoint del speller
            ph_l_weights = {k.replace("phoneme_listener.", ""): v for k, v in state_dict.items() if "phoneme_listener" in k}
            if ph_l_weights and hasattr(model, 'phoneme_listener'):
                 model.phoneme_listener.load_state_dict(ph_l_weights)
                 print("✅ Phoneme Listener actualizado desde Speller.")
                 
        except Exception as e:
            print(f"⚠️ Error cargando pesos de Speller: {e}")

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

    # Nombre de archivo distintivo según fase
    phase_suffix = f"_{training_phase}" if training_phase != "end_to_end" else ""
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"experiments/models/reader{phase_suffix}/{language}",
        filename=f"reader{phase_suffix}-{{epoch:02d}}-{{val_loss:.2f}}",
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
        default_root_dir=f"lightning_logs/experiment_reader{phase_suffix}_{language}"
    )
    
    # 4. Train
    if progress_callback:
        progress_callback(0, f"Iniciando entrenamiento Reader ({language}) [{training_phase}]...")
        
    trainer.fit(model, train_dataloaders=loaders['train'], val_dataloaders=loaders['val'])
    
    # 5. Save / Return Best
    best_path = checkpoint_callback.best_model_path
    if best_path and Path(best_path).exists():
        print(f"Usando mejor modelo Reader: {best_path}")
        final_path = best_path
        
        # Metadata
        meta_config = {
            "epochs": epochs, "lr": lr, "batch_size": batch_size,
            "weights": {"dtw": w_dtw, "perceptual": w_perceptual},
            "listener_ckpt": listener_ckpt,
            "recognizer_ckpt": recognizer_ckpt,
            "language": language,
            "vocab": words,
            "type": "reader",
            "use_two_stage": use_two_stage,
            "phoneme_listener_ckpt": phoneme_listener_ckpt,
            "training_phase": training_phase,
            "pretrained_speller_ckpt": pretrained_speller_ckpt
        }
        final_metrics = history_cb.history[-1] if history_cb.history else {}
        save_model_metadata(final_path, meta_config, final_metrics)
        
        return str(final_path), history_cb.history
    else:
        # Fallback
        save_dir = Path(f"experiments/models/reader{phase_suffix}")
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        final_path = save_dir / f"reader{phase_suffix}_{language}_{timestamp}.ckpt"
        trainer.save_checkpoint(final_path)
        
        # Metadata
        meta_config = {
            "epochs": epochs, "lr": lr, "batch_size": batch_size,
            "weights": {"dtw": w_dtw, "perceptual": w_perceptual},
            "listener_ckpt": listener_ckpt,
            "recognizer_ckpt": recognizer_ckpt,
            "language": language,
            "vocab": words,
            "type": "reader",
            "use_two_stage": use_two_stage,
            "phoneme_listener_ckpt": phoneme_listener_ckpt,
            "training_phase": training_phase,
            "pretrained_speller_ckpt": pretrained_speller_ckpt
        }
        final_metrics = history_cb.history[-1] if history_cb.history else {}
        save_model_metadata(final_path, meta_config, final_metrics)
        
        return str(final_path), history_cb.history
