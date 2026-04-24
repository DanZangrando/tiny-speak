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
from training.audio_module import TinyEarsLightning
from training.visual_module import TinyEyesLightning
from training.reader_module import TinyReaderLightning
from training.callbacks import TrainingHistoryCallback, RealTimePlotCallback, ReaderPredictionCallback
from utils.checkpoints import save_model_metadata
from utils.graphemes import get_language_letters, get_phoneme_inventory



def train_listener(
    language: str, 
    config: Dict[str, Any], 
    progress_callback=None,
    plot_placeholders=None
) -> Tuple[str, List[Dict]]:
    """
    Entrena un TinyEars (Listener) para un idioma específico.
    Retorna: (path_checkpoint, historial_metricas)
    """
    epochs = config.get('epochs', 10)
    lr = config.get('lr', 1e-3)
    batch_size = config.get('batch_size', 32)
    use_phonemes = config.get('use_phonemes', False)
    
    # 1. Cargar configuración maestra para ratios
    from training.config import load_master_dataset_config
    master_conf = load_master_dataset_config()
    split_ratios = master_conf.get("experiment_config", {}).get("split_ratios")
    
    # 1.2 Forzar inventario si es fonemas
    forced_classes = None
    if use_phonemes:
        forced_classes = get_phoneme_inventory(language)
        if not forced_classes:
            st.error(f"⚠️ No se encontró inventario fonético para {language}. Se usará el automático.")

    # 1.3 Data
    train_ds, val_ds, test_ds, loaders = build_audio_dataloaders(
        batch_size=batch_size, 
        num_workers=0, 
        seed=42,
        split_ratios=split_ratios,
        target_language=language,
        use_phonemes=use_phonemes,
        class_names=forced_classes
    )
    words = train_ds.class_names
    
    if not words:
        msg = f"No hay {'fonemas' if use_phonemes else 'palabras'} para el idioma {language}"
        raise ValueError(msg)

    # Inyectar hparams definidos en experiment_config
    from training.config import load_master_dataset_config
    master_conf = load_master_dataset_config()
    arch_type = "tiny_ears_phonemes" if use_phonemes else "tiny_ears_words"
    hparams = master_conf.get("architectures", {}).get(arch_type, {})

    # Combinamos con config de entrenamiento (lr, etc) de modo que se guarden en metadata
    config.update(hparams)

    # 2. Model
    model = TinyEarsLightning(
        class_names=words,
        learning_rate=lr,
        **hparams
    )
    
    # 3. Trainer
    history_cb = TrainingHistoryCallback()
    callbacks = [history_cb]
    
    if plot_placeholders:
        callbacks.append(RealTimePlotCallback(*plot_placeholders, max_epochs=epochs))

    # Callbacks (eliminado EarlyStopping para forzar completitud de épocas)

    # Subdirectorio diferente para fonemas si se desea, o mismo con prefijo
    sub_dir = "tiny_ears_phonemes" if use_phonemes else "tiny_ears_words"
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"data/checkpoints/{sub_dir}/{language}",
        filename="best_model",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks + [checkpoint_callback],
        enable_progress_bar=False, 
        default_root_dir=f"experiments/logs/{sub_dir}_{language}"
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
            "classes": words, "language": language, "type": "listener",
            "use_phonemes": use_phonemes
        }
        final_metrics = history_cb.history[-1] if history_cb.history else {}
        save_model_metadata(final_path, meta_config, final_metrics, history=history_cb.history)
        
        try:
            device = trainer.strategy.root_device
            model.to(device)
            evaluate_and_save(model, loaders['val'], final_path, "listener")
        except Exception as e:
            print(f"Auto-eval falló: {e}")
            
        return str(final_path), history_cb.history
    else:
        # Fallback
        save_dir = Path(f"data/checkpoints/{sub_dir}/{language}")
        save_dir.mkdir(parents=True, exist_ok=True)
        final_path = save_dir / "best_model.ckpt"
        trainer.save_checkpoint(final_path)
        
        # Metadata
        meta_config = {
            "epochs": epochs, "lr": lr, "batch_size": batch_size,
            "classes": words, "language": language, "type": "listener",
            "use_phonemes": use_phonemes
        }
        final_metrics = history_cb.history[-1] if history_cb.history else {}
        save_model_metadata(final_path, meta_config, final_metrics, history=history_cb.history)
        
        try:
            device = trainer.strategy.root_device
            model.to(device)
            evaluate_and_save(model, loaders['val'], final_path, "listener")
        except Exception as e:
            print(f"Auto-eval falló: {e}")
            
        return str(final_path), history_cb.history

def train_recognizer(
    language: str, 
    config: Dict[str, Any],
    progress_callback=None,
    plot_placeholders=None
) -> Tuple[str, List[Dict]]:
    """
    Entrena un TinyEyes (Recognizer) para un idioma específico.
    """
    epochs = config.get('epochs', 10)
    lr = config.get('lr', 1e-3)
    batch_size = config.get('batch_size', 32)
    
    # 1. Cargar configuración maestra para ratios
    from training.config import load_master_dataset_config
    master_conf = load_master_dataset_config()
    split_ratios = master_conf.get("experiment_config", {}).get("split_ratios")
    
    # 1.2 Forzar alfabeto del idioma
    alphabet = get_language_letters(language)
    
    # 1.3 Data
    train_ds, val_ds, test_ds, loaders = build_visual_dataloaders(
        batch_size=batch_size, 
        num_workers=0, 
        seed=42,
        split_ratios=split_ratios,
        target_language=language,
        class_names=alphabet
    )
    class_names = train_ds.letters
    # Inyectar hparams definidos en experiment_config
    from training.config import load_master_dataset_config
    master_conf = load_master_dataset_config()
    hparams = master_conf.get("architectures", {}).get("tiny_eyes", {})

    # Combinamos con config de entrenamiento (lr, etc)
    config.update(hparams)

    # 2. Model
    model = TinyEyesLightning(
        class_names=class_names,
        learning_rate=lr,
        **hparams
    )
    
    # 3. Trainer
    history_cb = TrainingHistoryCallback()
    callbacks = [history_cb]
    
    if plot_placeholders:
        callbacks.append(RealTimePlotCallback(*plot_placeholders, max_epochs=epochs))

    # Callbacks (eliminado EarlyStopping para forzar completitud de épocas)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"data/checkpoints/tiny_eyes/{language}",
        filename="best_model",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks + [checkpoint_callback],
        enable_progress_bar=False,
        default_root_dir=f"experiments/logs/tiny_eyes_{language}"
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
        
        meta_config = {
            "epochs": epochs, "lr": lr, "batch_size": batch_size,
            "classes": class_names, "language": language, "type": "recognizer"
        }
        final_metrics = history_cb.history[-1] if history_cb.history else {}
        save_model_metadata(final_path, meta_config, final_metrics, history=history_cb.history)
        
        try:
            device = trainer.strategy.root_device
            model.to(device)
            evaluate_and_save(model, loaders['val'], final_path, "recognizer")
        except Exception as e:
            print(f"Auto-eval falló: {e}")
            
        return str(final_path), history_cb.history
    else:
        # Fallback: Guardar el actual
        save_dir = Path(f"data/checkpoints/tiny_eyes/{language}")
        save_dir.mkdir(parents=True, exist_ok=True)
        final_path = save_dir / "best_model.ckpt"
        trainer.save_checkpoint(final_path)
        
        # Metadata
        meta_config = {
            "epochs": epochs, "lr": lr, "batch_size": batch_size,
            "classes": class_names, "language": language, "type": "recognizer"
        }
        final_metrics = history_cb.history[-1] if history_cb.history else {}
        save_model_metadata(final_path, meta_config, final_metrics, history=history_cb.history)
        
        try:
            device = trainer.strategy.root_device
            model.to(device)
            evaluate_and_save(model, loaders['val'], final_path, "recognizer")
        except Exception as e:
            print(f"Auto-eval falló: {e}")
            
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
    w_perceptual = config.get('w_perceptual', 0.5) # Categorical
    w_mse = config.get('w_mse', 1.0)               # Structural
    
    use_two_stage = config.get('use_two_stage', False)
    phoneme_listener_ckpt = config.get('phoneme_listener_ckpt', None)
    training_phase = config.get('training_phase', 'end_to_end')
    pretrained_speller_ckpt = config.get('pretrained_speller_ckpt', None)
    
    # Auto-detección de phoneme_listener_ckpt si falta y es necesario
    if use_two_stage and not phoneme_listener_ckpt:
        if training_phase == "g2p":
            phoneme_listener_ckpt = listener_ckpt
        else:
            # Buscar el de fonemas por defecto
            ph_path = Path(f"data/checkpoints/tiny_ears_phonemes/{language}/best_model.ckpt")
            if ph_path.exists():
                phoneme_listener_ckpt = str(ph_path)
    # 1. Cargar configuración maestra para ratios
    from training.config import load_master_dataset_config
    master_conf = load_master_dataset_config()
    split_ratios = master_conf.get("experiment_config", {}).get("split_ratios")
    
    # Determinar modo (Atomic para G2P)
    is_g2p = training_phase == 'g2p'
    phoneme_to_idx = None
    
    if is_g2p:
        # Extraer el mapa de fonemas del listener para el modo atómico
        if phoneme_listener_ckpt:
             from training.audio_module import TinyEarsLightning
             import torch
             # Cargamos el listener maestro para obtener sus clases oficiales
             try:
                 ph_model = TinyEarsLightning.load_from_checkpoint(phoneme_listener_ckpt, map_location="cpu")
                 ph_classes = getattr(ph_model, "class_names", [])
                 phoneme_to_idx = {p: i for i, p in enumerate(ph_classes)}
                 print(f"✅ Mapa de fonemas sincronizado: {len(phoneme_to_idx)} clases detectadas.")
             except Exception as e:
                 print(f"⚠️ Error cargando mapa de fonemas desde {phoneme_listener_ckpt}: {e}")
                 # Dejar phoneme_to_idx en None para que build_reader_dataloaders falle de forma controlada
    
    from training.reader_dataset import build_reader_dataloaders
    train_ds, val_ds, test_ds, loaders = build_reader_dataloaders(
        batch_size=batch_size, 
        num_workers=0, 
        seed=42,
        split_ratios=split_ratios,
        target_language=language,
        training_phase=training_phase,
        phoneme_to_idx=phoneme_to_idx
    )
    
    if train_ds is None or loaders is None:
        raise RuntimeError(f"❌ Error crítico: No se pudieron construir los dataloaders para {language}. Verifica que existan imágenes visuales para este idioma.")
        
    words = train_ds.class_names if hasattr(train_ds, "class_names") else []
    
    # Inyectar hparams definidos en experiment_config
    reader_type = "tiny_speller" if is_g2p else "tiny_reader"
    hparams = master_conf.get("architectures", {}).get(reader_type, {})
    
    # Combinamos con config de entrenamiento
    config.update(hparams)

    # 2. Model
    model = TinyReaderLightning(
        class_names=words,
        listener_checkpoint_path=listener_ckpt,
        recognizer_checkpoint_path=recognizer_ckpt,
        learning_rate=lr,
        w_perceptual=w_perceptual,
        w_mse=w_mse,
        use_two_stage=use_two_stage,
        phoneme_listener_checkpoint_path=phoneme_listener_ckpt,
        training_phase=training_phase,
        target_language=language,
        **hparams
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
        callbacks.append(RealTimePlotCallback(*plot_placeholders, max_epochs=epochs))
        
    if prediction_placeholder:
        callbacks.append(ReaderPredictionCallback(loaders['val'], prediction_placeholder))

    # Callbacks (eliminado EarlyStopping para forzar completitud de épocas)

    # Nombre de archivo distintivo según fase
    sub_dir = "tiny_speller" if training_phase == "g2p" else "tiny_reader"
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"data/checkpoints/{sub_dir}/{language}",
        filename="best_model",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks + [checkpoint_callback],
        enable_progress_bar=False,
        default_root_dir=f"experiments/logs/{sub_dir}_{language}"
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
            "weights": {"structural_mse": w_perceptual},
            "listener_ckpt": listener_ckpt,
            "recognizer_ckpt": recognizer_ckpt,
            "language": language,
            "classes": words,
            "type": "reader",
            "use_two_stage": use_two_stage,
            "phoneme_listener_ckpt": phoneme_listener_ckpt,
            "training_phase": training_phase,
            "pretrained_speller_ckpt": pretrained_speller_ckpt
        }
        final_metrics = history_cb.history[-1] if history_cb.history else {}
        save_model_metadata(final_path, meta_config, final_metrics, history=history_cb.history)
        
        try:
            device = trainer.strategy.root_device
            model.to(device)
            evaluate_and_save(model, loaders['val'], final_path, "reader")
        except Exception as e:
            print(f"Auto-eval falló: {e}")
            
        return str(final_path), history_cb.history
    else:
        # Fallback
        save_dir = Path(f"data/checkpoints/{sub_dir}/{language}")
        save_dir.mkdir(parents=True, exist_ok=True)
        final_path = save_dir / "best_model.ckpt"
        trainer.save_checkpoint(final_path)
        
        # Metadata
        meta_config = {
            "epochs": epochs, "lr": lr, "batch_size": batch_size,
            "weights": {"structural_mse": w_mse, "categorical_ce": w_perceptual},
            "listener_ckpt": listener_ckpt,
            "recognizer_ckpt": recognizer_ckpt,
            "language": language,
            "classes": words,
            "type": "reader",
            "use_two_stage": use_two_stage,
            "phoneme_listener_ckpt": phoneme_listener_ckpt,
            "training_phase": training_phase,
            "pretrained_speller_ckpt": pretrained_speller_ckpt
        }
        final_metrics = history_cb.history[-1] if history_cb.history else {}
        save_model_metadata(final_path, meta_config, final_metrics, history=history_cb.history)
        
        try:
            device = trainer.strategy.root_device
            model.to(device)
            evaluate_and_save(model, loaders['val'], final_path, "reader")
        except Exception as e:
            print(f"Auto-eval falló: {e}")
            
        return str(final_path), history_cb.history

def evaluate_and_save(model, val_loader, final_path, model_type: str):
    import pickle
    device = next(model.parameters()).device
    model.eval()

    if isinstance(final_path, str):
        final_path = Path(final_path)

    eval_data = {}
    
    if "listener" in model_type:
        all_preds = []
        all_labels = []
        all_embeddings = []
        eval_list = []
        with torch.no_grad():
            from torch.nn.utils.rnn import pad_sequence
            for batch in val_loader:
                waveforms = [w.to(device) for w in batch["waveforms"]]
                labels = batch["label"].to(device)
                if isinstance(waveforms, list):
                    wave_input = pad_sequence(waveforms, batch_first=True).to(device)
                else:
                    wave_input = waveforms
                logits, embeddings = model.model(wave_input)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_embeddings.extend(embeddings.mean(dim=1).cpu().numpy())

                # Guardar muestras para visualización
                if len(eval_list) < 50:
                    for i in range(min(len(preds), 50 - len(eval_list))):
                        eval_list.append({
                            "prediction": model.class_names[preds[i]],
                            "target": model.class_names[labels[i]],
                            "confidence": torch.softmax(logits[i], dim=0).max().item()
                        })
        
        eval_data = {
            "samples": eval_list,
            "labels": all_labels,
            "embeddings": all_embeddings,
            "confusion": {
                "y_true": all_labels,
                "y_pred": all_preds,
                "class_names": model.class_names
            }
        }
        
    elif "recognizer" in model_type:
        all_preds = []
        all_labels = []
        all_embeddings = []
        eval_list = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits, embeddings = model.model(images)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_embeddings.extend(embeddings.cpu().numpy())

                # Guardar muestras
                if len(eval_list) < 50:
                    for i in range(min(len(preds), 50 - len(eval_list))):
                        eval_list.append({
                            "prediction": model.class_names[preds[i]],
                            "target": model.class_names[labels[i]],
                            "confidence": torch.softmax(logits[i], dim=0).max().item()
                        })
                
        eval_data = {
            "samples": eval_list,
            "labels": all_labels,
            "embeddings": all_embeddings,
            "confusion": {
                "y_true": all_labels,
                "y_pred": all_preds,
                "class_names": model.class_names
            }
        }
        
    elif "reader" in model_type:
        eval_list = []
        all_phoneme_preds = []
        all_phoneme_targets = []
        all_embeddings = []
        all_labels = []
        is_g2p = getattr(model, "training_phase", "end_to_end") == "g2p"
        
        with torch.no_grad():
            for batch in val_loader:
                if "waveforms" in batch:
                    batch["waveforms"] = [w.to(device) for w in batch["waveforms"]]
                if "waveform" in batch and hasattr(batch["waveform"], "to"):
                    batch["waveform"] = batch["waveform"].to(device)
                if "label" in batch and hasattr(batch["label"], "to"):
                    batch["label"] = batch["label"].to(device)

                try:
                    res = model.get_predictions(batch)
                    # Manejar retornos de 5 o 6 valores (compatibilidad)
                    if len(res) == 6:
                        words, preds, confs, targets, batch_embs, batch_labs = res
                    elif len(res) == 5:
                        words, preds, confs, targets, batch_embs = res
                        batch_labs = labels # Fallback a etiquetas del batch si no las devuelve el modelo
                    else:
                        continue
                        
                        # Recolectar para PCA (limite 1000)
                        if len(all_embeddings) < 1000:
                            all_embeddings.extend(batch_embs.cpu().numpy())
                            all_labels.extend(batch_labs.cpu().numpy())

                        if is_g2p:
                            # Para la matriz de confusión, recolectamos todos los fonemas individuales (como índices)
                            for p_str, t_str in zip(preds, targets):
                                p_list = p_str.split()
                                t_list = t_str.split()
                                for p, t in zip(p_list, t_list):
                                    if p in model.phoneme_to_idx and t in model.phoneme_to_idx:
                                        all_phoneme_preds.append(model.phoneme_to_idx[p])
                                        all_phoneme_targets.append(model.phoneme_to_idx[t])
                        else:
                            # Para el Reader P2W, la matriz de confusión es sobre palabras
                            # preds y confs ya vienen listos
                            # Las etiquetas reales están en batch_labs (que son los word_targets)
                            all_phoneme_preds.extend([model.class_to_idx[p] for p in preds])
                            all_phoneme_targets.extend(batch_labs.cpu().tolist())

                    # Guardar una muestra para la tabla de visualización (máx 100 palabras)
                    if len(eval_list) < 100:
                        for i in range(min(len(words), 100 - len(eval_list))):
                            input_key = "grapheme" if is_g2p else "word"
                            entry = {
                                input_key: words[i],
                                "prediction": preds[i],
                                "confidence": confs[i],
                            }
                            if targets:
                                entry["target"] = targets[i]
                            eval_list.append(entry)
                except Exception as e:
                    print(f"evaluate_and_save reader batch error: {e}")

        # Estructura estandarizada de eval_results.pkl
        eval_data = {
            "samples": eval_list,
            "labels": all_labels,
            "preds": all_phoneme_preds, # Reutilizamos esta clave para compatibilidad
            "embeddings": all_embeddings,
            "confusion": {
                "y_true": all_phoneme_targets,
                "y_pred": all_phoneme_preds,
                "class_names": getattr(model, "phoneme_class_names", []) if is_g2p else model.class_names
            }
        }

    eval_path = final_path.parent / "eval_results.pkl"
    with open(eval_path, "wb") as f:
        pickle.dump(eval_data, f)
