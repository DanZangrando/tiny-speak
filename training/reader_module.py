"""PyTorch Lightning module para entrenar TinyReader (Generación Top-Down)."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import random
from pathlib import Path
from PIL import Image

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torchvision import transforms

from models import TinyReader, PhonologicalPathway, VisualPathway, TinyReaderG2P, TinyReaderP2W
from utils import WAV2VEC_DIM, load_wav2vec_model, get_phonemes_from_word, SoftDTW
from training.config import load_master_dataset_config

class TinyReaderLightning(pl.LightningModule):
    """
    LightningModule para TinyReader.
    Aprende a generar embeddings de Wav2Vec2 a partir de secuencias de logits de letras (Spelling).
    Usa VisualPathway para "leer" las letras y PhonologicalPathway como "Oído Interno".
    Soporta modo Two-Stage: Grapheme -> Phoneme -> Word.
    Soporta Curriculum Training: 'g2p', 'p2w', 'end_to_end'.
    """

    def __init__(
        self,
        class_names: Sequence[str],
        listener_checkpoint_path: str,
        recognizer_checkpoint_path: str, # Nuevo: Path al reconocedor
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        hidden_dim: int = 256,
        num_layers: int = 2,
        w_dtw: float = 1.0,
        w_perceptual: float = 0.5,
        use_two_stage: bool = False,
        phoneme_listener_checkpoint_path: str = None,
        training_phase: str = "end_to_end", # 'g2p', 'p2w', 'end_to_end'
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.class_names = list(class_names)
        self.use_two_stage = use_two_stage
        self.training_phase = training_phase
        
        # Cargar configuración para buscar imágenes
        self.dataset_config = load_master_dataset_config()
        self.visual_config = self.dataset_config.get("visual_dataset", {}).get("generated_images", {})
        
        # Transformaciones para imágenes (deben coincidir con las del Recognizer)
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        # 1. Modelo Perceptivo (Listener - Oído Interno)
        self.listener = self._load_listener(listener_checkpoint_path)
        self.listener.eval()
        for p in self.listener.parameters():
            p.requires_grad = False
            
        # 1.1 Phoneme Listener (Solo si Two-Stage)
        if self.use_two_stage:
            if not phoneme_listener_checkpoint_path:
                raise ValueError("phoneme_listener_checkpoint_path es requerido para use_two_stage=True")
            self.phoneme_listener = self._load_listener(phoneme_listener_checkpoint_path)
            self.phoneme_listener.eval()
            for p in self.phoneme_listener.parameters():
                p.requires_grad = False
            
            # Crear mapa de fonemas
            self.phoneme_class_names = self.phoneme_listener.class_names if hasattr(self.phoneme_listener, 'class_names') else []
            self.phoneme_to_idx = {p: i for i, p in enumerate(self.phoneme_class_names)}
            
            # Inicializar banco de embeddings de fonemas (Canonical Phoneme Embeddings)
            self.register_buffer("phoneme_embeddings_bank", torch.zeros(len(self.phoneme_class_names), self.phoneme_listener.hidden_dim))
            self._init_phoneme_bank()
        
        # 2. Modelo Visual (Recognizer - Ojo)
        self.recognizer = self._load_recognizer(recognizer_checkpoint_path)
        self.recognizer.eval()
        for p in self.recognizer.parameters():
            p.requires_grad = False
            
        # Obtener dimensión de salida del recognizer (num_letters)
        if self.recognizer.classifier:
            input_dim = self.recognizer.classifier.out_features
        else:
            input_dim = 26 

        # 3. Modelo Generativo (Reader)
        if self.use_two_stage:
            # Stage 1: Grapheme -> Phoneme
            # Output dim debe coincidir con hidden_dim del PhonemeListener (256)
            self.reader_g2p = TinyReaderG2P(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=self.phoneme_listener.hidden_dim, 
                num_layers=num_layers
            )
            # Stage 2: Phoneme -> Word
            # Input dim = PhonemeListener hidden dim
            # Output dim = Word Listener hidden dim (WAV2VEC_DIM o 256)
            target_dim = self.listener.hidden_dim
            
            self.reader_p2w = TinyReaderP2W(
                input_dim=self.phoneme_listener.hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=target_dim,
                num_layers=num_layers
            )
            
            # Wrapper para facilitar acceso a parametros
            self.reader = nn.ModuleList([self.reader_g2p, self.reader_p2w])
            
            # Congelar capas según fase
            if self.training_phase == "g2p":
                print("❄️ Fase G2P: Congelando P2W")
                for p in self.reader_p2w.parameters():
                    p.requires_grad = False
            elif self.training_phase == "p2w":
                print("❄️ Fase P2W: Congelando G2P")
                for p in self.reader_g2p.parameters():
                    p.requires_grad = False
            
        else:
            # Single Stage: Grapheme -> Word
            target_dim = self.listener.hidden_dim # Usar hidden_dim del listener como target
            self.reader = TinyReader(
                input_dim=input_dim, 
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=target_dim 
            )
            
        # Pérdidas
        # Pérdidas
        self.perceptual_loss = nn.CrossEntropyLoss()
        self.soft_dtw = SoftDTW(gamma=0.1, normalize=True)

    def _init_phoneme_bank(self):
        """Inicializa el banco de embeddings de fonemas usando el PhonemeListener."""
        print("Inicializando banco de embeddings de fonemas...")
        from utils import load_waveform
        
        # Cargar configuración de samples
        phoneme_samples = self.dataset_config.get("phoneme_samples", {})
        if not phoneme_samples:
            print("Advertencia: No hay phoneme_samples en la configuración.")
            return

        device = self.device if hasattr(self, "device") else "cpu"
        # Mover listener a device temporalmente si es necesario
        self.phoneme_listener.to(device)
        
        repo_root = Path(self.dataset_config.get("repo_root", "."))
        
        with torch.no_grad():
            for i, phoneme in enumerate(self.phoneme_class_names):
                # Buscar samples para este fonema en todos los idiomas
                samples = []
                for lang_data in phoneme_samples.values():
                    if phoneme in lang_data:
                        samples.extend(lang_data[phoneme])
                
                if not samples:
                    continue
                    
                # Tomar hasta 5 samples para promediar
                selected_samples = samples[:5]
                embeddings = []
                
                for s in selected_samples:
                    try:
                        path = repo_root / s['file_path']
                        if path.exists():
                            waveform = load_waveform(str(path)).to(device)
                            # (1, Samples)
                            waveform = waveform.unsqueeze(0) 
                            
                            # Extraer embedding
                            # (Layers, Batch, Time, Dim) -> (1, 1, T, D)
                            emb = self.phoneme_listener.extract_hidden_activations(waveform)
                            
                            # Mean Pooling sobre el eje temporal (dim 2)
                            # (1, 1, D)
                            pooled = emb.mean(dim=2)
                            # (D)
                            pooled = pooled.squeeze()
                            embeddings.append(pooled)
                    except Exception as e:
                        print(f"Error procesando sample fonema {phoneme}: {e}")
                        
                if embeddings:
                    # Promediar
                    avg_emb = torch.stack(embeddings).mean(dim=0).squeeze(0)
                    self.phoneme_embeddings_bank[i] = avg_emb
                    
        print(f"Banco de fonemas inicializado. {len(self.phoneme_class_names)} fonemas.")

    def _load_recognizer(self, checkpoint_path: str) -> VisualPathway:
        """Carga el VisualPathway desde checkpoint para usarlo como ojo."""
        import importlib
        RecognizerPL = importlib.import_module("training.visual_module").VisualPathwayLightning
        
        print(f"Cargando VisualPathway (Ojo) desde {checkpoint_path}...")
        pl_module = RecognizerPL.load_from_checkpoint(
            checkpoint_path,
            map_location=self.device if hasattr(self, "device") else "cpu"
        )
        return pl_module.model

    def _get_word_images(self, word: str) -> torch.Tensor:
        """
        Obtiene una secuencia de imágenes para deletrear la palabra.
        Retorna: (L, C, H, W)
        """
        images = []
        repo_root = Path(self.dataset_config.get("repo_root", "."))
        
        # Obtener grafemas disponibles en el dataset visual
        available_graphemes = list(self.visual_config.keys())
        
        from utils import tokenize_graphemes
        tokens = tokenize_graphemes(word, available_graphemes)
        
        for char_key in tokens:
            entries = self.visual_config.get(char_key, [])
            if not entries:
                entries = self.visual_config.get(char_key.lower(), [])
            
            if entries:
                entry = random.choice(entries)
                rel_path = entry.get("file_path")
                full_path = repo_root / rel_path
                try:
                    img = Image.open(full_path).convert("RGB")
                    img_tensor = self.transform(img)
                except Exception:
                    img_tensor = torch.zeros(3, 64, 64)
            else:
                img_tensor = torch.zeros(3, 64, 64)
                
            images.append(img_tensor)
            
        if not images:
            return torch.zeros(1, 3, 64, 64)
            
        return torch.stack(images)

    def _load_listener(self, checkpoint_path: str) -> PhonologicalPathway:
        """Carga el PhonologicalPathway desde checkpoint para usarlo como juez."""
        from training.audio_module import PhonologicalPathwayLightning as ListenerPL
        import json
        
        print(f"Cargando PhonologicalPathway desde {checkpoint_path}...")
        
        # 1. Intentar obtener vocabulario correcto desde metadata
        meta_path = Path(checkpoint_path).with_suffix(".ckpt.meta.json")
        vocab = None
        
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                vocab = meta.get("config", {}).get("vocab", [])
                if not vocab:
                    vocab = meta.get("vocab", [])
            except Exception as e:
                print(f"Error leyendo metadata del listener: {e}")
        
        # 2. Cargar checkpoint
        try:
            if vocab:
                # Si tenemos vocabulario, lo usamos explícitamente
                pl_module = ListenerPL.load_from_checkpoint(
                    checkpoint_path,
                    class_names=vocab,
                    map_location=self.device if hasattr(self, "device") else "cpu"
                )
            else:
                # Si no, intentamos sin argumentos (si el checkpoint guardó hparams)
                pl_module = ListenerPL.load_from_checkpoint(
                    checkpoint_path,
                    map_location=self.device if hasattr(self, "device") else "cpu"
                )
        except Exception as e:
            print(f"Fallo carga automática/metadata: {e}")
            # Fallback legacy: Usar self.class_names (SOLO SI ES WORD LISTENER)
            # Si estamos cargando el phoneme listener y fallamos aquí, es probable que explote
            # si usamos self.class_names (palabras) para un modelo de fonemas.
            
            # Heurística: Si el error es de tamaño y estamos cargando un listener,
            # intentamos inferir si es fonema o palabra.
            
            # Por ahora, usamos self.class_names como último recurso, pero advertimos.
            print(f"Usando fallback self.class_names ({len(self.class_names)}) para cargar listener.")
            pl_module = ListenerPL.load_from_checkpoint(
                checkpoint_path,
                class_names=self.class_names, 
                map_location=self.device if hasattr(self, "device") else "cpu"
            )
            
        # Asegurar que el modelo tenga class_names
        if hasattr(pl_module, 'class_names'):
             pl_module.model.class_names = pl_module.class_names
             
        return pl_module.model

    def forward(self, x_seq, target_length=None):
        if self.use_two_stage:
            # En forward simple, asumimos paso completo
            # Stage 1
            phoneme_emb = self.reader_g2p(x_seq, target_length=target_length) # target_length es tricky aquí
            # Stage 2
            word_emb = self.reader_p2w(phoneme_emb, target_length=target_length)
            return word_emb
        else:
            return self.reader(x_seq, target_length)

    def _shared_step(self, batch: Dict, stage: str) -> torch.Tensor:
        # Datos reales
        waveforms = [w.to(self.device) for w in batch["waveforms"]]
        labels = batch["label"].to(self.device)
        batch_size = len(waveforms)
        
        # 1. Obtener Ground Truth Embeddings (Bottom-Up)
        with torch.no_grad():
            waveforms_padded = pad_sequence(waveforms, batch_first=True)
            real_embeddings = self.listener.extract_hidden_activations(waveforms_padded)
            real_embeddings, lengths = self.listener.mask_hidden_activations(real_embeddings)
            real_embeddings, lengths = self.listener.downsample_hidden_activations(real_embeddings, lengths, factor=7)
            real_embeddings = real_embeddings.squeeze(0)
            
        # 2. Generar Imaginación (Top-Down)
        words = [self.class_names[i] for i in labels]
        
        # A. Obtener secuencia de logits de letras (Spelling)
        logits_sequences = []
        for word in words:
            images = self._get_word_images(word).to(self.device)
            with torch.no_grad():
                res = self.recognizer(images)
                word_logits = res[0] if isinstance(res, tuple) else res
            logits_sequences.append(word_logits)
            
        padded_logits = pad_sequence(logits_sequences, batch_first=True, padding_value=0.0)
        max_len = real_embeddings.size(1)
        
        if self.use_two_stage:
            # --- STAGE 1: G2P ---
            
            # 1. Calcular targets (fonemas) para cada palabra
            phoneme_targets_list = []
            for word in words:
                phonemes = get_phonemes_from_word(word)
                # Mapear a índices. Si no existe, 0 (pero idealmente deberíamos tener un token UNK)
                idxs = [self.phoneme_to_idx.get(p, 0) for p in phonemes]
                if not idxs: idxs = [0] # Evitar listas vacías
                phoneme_targets_list.append(torch.tensor(idxs, device=self.device))

            # 2. Determinar longitud máxima para este batch
            # En G2P, queremos que el modelo genere tantos fonemas como sea necesario
            max_target_len = max(len(t) for t in phoneme_targets_list)
            
            # 3. Generar Embeddings (Imaginación) con la longitud correcta
            # Si estamos en fase P2W, no queremos gradientes en G2P
            if self.training_phase == "p2w":
                with torch.no_grad():
                    generated_phoneme_embeddings = self.reader_g2p(padded_logits, target_length=max_target_len)
            else:
                generated_phoneme_embeddings = self.reader_g2p(padded_logits, target_length=max_target_len)
            
            # 4. Padear targets para coincidir con max_target_len
            # Usamos -100 para que CrossEntropyLoss ignore el padding
            phoneme_targets_padded = pad_sequence(phoneme_targets_list, batch_first=True, padding_value=-100)
            
            # Si generated es más largo (raro si pasamos target_length), recortar targets? No, pad_sequence ya maneja max len.
            # Pero si generated es más corto? (Imposible si pasamos target_length)
            
            phoneme_targets_flat = phoneme_targets_padded.view(-1)
            
            # Si estamos en fase G2P, calculamos pérdida aquí y retornamos
            if self.training_phase == "g2p":
                # A. Perceptual Phoneme Loss
                phoneme_logits = self.phoneme_listener.classifier(generated_phoneme_embeddings)
                phoneme_logits_flat = phoneme_logits.view(-1, phoneme_logits.size(-1))
                
                # CrossEntropyLoss ignora -100 por defecto
                loss_perceptual_g2p = F.cross_entropy(phoneme_logits_flat, phoneme_targets_flat)
                
                # B. Soft-DTW Loss (usando Phoneme Bank)
                # SoftDTW necesita secuencias alineadas o maneja longitudes variables?
                # Nuestra implementación maneja (B, N, D) y (B, M, D).
                # Pero necesitamos los embeddings REALES de los fonemas target.
                # Construimos batch de embeddings reales
                # Para padding (-100), usamos un embedding dummy (ej. ceros)
                
                # Crear máscara para ignorar padding en DTW si fuera necesario, 
                # pero SoftDTW calcula distancia global.
                # Simplemente usamos los índices válidos.
                # Reemplazar -100 con 0 para lookup, luego enmascarar?
                # Simplificación: Ignorar DTW para padding o usar solo Perceptual Loss si DTW es inestable.
                # Vamos a intentar usar DTW solo en la parte válida es complejo.
                # Por ahora, usamos Perceptual Loss como principal y DTW con targets padeados (usando embedding 0 para padding)
                
                targets_for_dtw = phoneme_targets_padded.clone()
                targets_for_dtw[targets_for_dtw == -100] = 0 # Dummy index
                real_phoneme_emb = self.phoneme_embeddings_bank[targets_for_dtw]
                
                # Masking DTW is hard. Let's rely on Perceptual Loss mainly.
                # Reduce DTW weight or ignore it for now to simplify?
                # User asked to simplify. Let's keep DTW but maybe it adds noise if padding is involved.
                loss_dtw_g2p = self.soft_dtw(generated_phoneme_embeddings, real_phoneme_emb)
                
                loss_stage1 = (
                    self.hparams.w_perceptual * loss_perceptual_g2p +
                    self.hparams.w_dtw * loss_dtw_g2p 
                )
                
                self.log(f"{stage}_g2p_perceptual", loss_perceptual_g2p)
                self.log(f"{stage}_g2p_dtw", loss_dtw_g2p)
                self.log(f"{stage}_loss", loss_stage1)
                
                # Calcular Accuracy Fonemas (ignorando padding)
                preds = torch.argmax(phoneme_logits, dim=2)
                mask = phoneme_targets_padded != -100
                acc = (preds[mask] == phoneme_targets_padded[mask]).float().mean()
                self.log(f"{stage}_phoneme_acc", acc)
                
                return loss_stage1
            
            # --- STAGE 2: P2W ---
            generated_embeddings = self.reader_p2w(generated_phoneme_embeddings, target_length=max_len)
            
        else:
            generated_embeddings = self.reader(padded_logits, target_length=max_len)
        
        # 3. Calcular Pérdidas (Stage 2 o Single Stage)
        
        # A. Soft-DTW Loss (Reemplaza MSE+Cosine para secuencias)
        # No usamos mask aquí porque SoftDTW maneja secuencias de diferentes longitudes
        loss_dtw = self.soft_dtw(generated_embeddings, real_embeddings)
        
        # B. Perceptual Loss (Word Level)
        # Pooling sobre tiempo para clasificar
        # Usamos la máscara para el pooling si las longitudes son variables
        mask = torch.arange(max_len, device=self.device).expand(batch_size, max_len) < lengths.unsqueeze(1)
        mask_float = mask.float().unsqueeze(-1)
        pooled_gen = (generated_embeddings * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-9)
        
        listener_logits = self.listener.classifier(pooled_gen)
        loss_perceptual = self.perceptual_loss(listener_logits, labels)
        
        # Total Loss
        total_loss = (
            self.hparams.w_dtw * loss_dtw + 
            self.hparams.w_perceptual * loss_perceptual
        )
        
        # Metrics
        acc = (listener_logits.argmax(dim=1) == labels).float().mean()
        
        # Logging
        self.log(f"{stage}_loss", total_loss, prog_bar=True)
        self.log(f"{stage}_dtw", loss_dtw)
        self.log(f"{stage}_perceptual", loss_perceptual)
        self.log(f"{stage}_acc", acc)
        
        return total_loss

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def get_predictions(self, batch: Dict) -> Tuple[List[str], List[str], List[float]]:
        """
        Retorna las etiquetas reales, las predicciones (texto) y la confianza para un batch.
        """
        # Cambiar modo localmente
        modules_to_eval = [self.reader] if not isinstance(self.reader, nn.ModuleList) else self.reader
        was_training = [m.training for m in modules_to_eval]
        for m in modules_to_eval: m.eval()
            
        try:
            with torch.no_grad():
                waveforms = [w.to(self.device) for w in batch["waveforms"]]
                labels = batch["label"].to(self.device)
                
                # Ground Truth info
                waveforms_padded = pad_sequence(waveforms, batch_first=True)
                real_embeddings = self.listener.extract_hidden_activations(waveforms_padded)
                real_embeddings, lengths = self.listener.mask_hidden_activations(real_embeddings)
                real_embeddings, lengths = self.listener.downsample_hidden_activations(real_embeddings, lengths, factor=7)
                max_len = real_embeddings.size(1)
                
                words = [self.class_names[i] for i in labels]
                logits_sequences = []
                for word in words:
                    images = self._get_word_images(word).to(self.device)
                    res = self.recognizer(images)
                    word_logits = res[0] if isinstance(res, tuple) else res
                    logits_sequences.append(word_logits)
                
                padded_logits = pad_sequence(logits_sequences, batch_first=True, padding_value=0.0)
                
                if self.training_phase == "g2p":
                    # --- G2P Phase: Predecir Fonemas ---
                    
                    # 1. Calcular targets reales para determinar longitud
                    phoneme_targets_list = []
                    for word in words:
                        phonemes = get_phonemes_from_word(word)
                        idxs = [self.phoneme_to_idx.get(p, 0) for p in phonemes]
                        if not idxs: idxs = [0]
                        phoneme_targets_list.append(idxs)
                    
                    # 2. Determinar max len real
                    max_target_len = max(len(t) for t in phoneme_targets_list)
                    
                    # 3. Generar
                    generated_phoneme_embeddings = self.reader_g2p(padded_logits, target_length=max_target_len)
                    
                    # Clasificar fonemas
                    phoneme_logits = self.phoneme_listener.classifier(generated_phoneme_embeddings)
                    
                    # Decodificar
                    probs = torch.softmax(phoneme_logits, dim=-1)
                    top1_probs, top1_indices = torch.max(probs, dim=-1) # (B, T)
                    
                    predictions = []
                    confidences = []
                    for b in range(len(words)):
                        indices = top1_indices[b]
                        # Recortar a la longitud real del target de esta palabra?
                        # No sabemos la longitud real durante inferencia pura, pero aquí tenemos labels.
                        # Para visualización "honesta", mostramos todo lo que generó el modelo (hasta max_target_len).
                        # Pero max_target_len depende del batch.
                        
                        phonemes = [self.phoneme_class_names[idx] for idx in indices]
                        pred_str = " ".join(phonemes)
                        predictions.append(pred_str)
                        confidences.append(top1_probs[b].mean().item())
                        
                    return words, predictions, confidences

                else:
                    # --- P2W / End-to-End Phase: Predecir Palabras ---
                    if self.use_two_stage:
                        # Para P2W, necesitamos generar fonemas intermedios.
                        # Usamos una longitud fija razonable o basada en grafemas?
                        # En inferencia real P2W, no sabemos los fonemas target.
                        # Usamos heurística: len(grafemas) * 1.5? O simplemente len(grafemas).
                        # Por ahora usamos len(grafemas) como aproximación.
                        target_len_phonemes = padded_logits.size(1)
                        phoneme_emb = self.reader_g2p(padded_logits, target_length=target_len_phonemes)
                        generated_embeddings = self.reader_p2w(phoneme_emb, target_length=max_len)
                    else:
                        generated_embeddings = self.reader(padded_logits, target_length=max_len)
                    
                    # Clasificar palabras
                    mask = torch.arange(max_len, device=self.device).expand(len(waveforms), max_len) < lengths.unsqueeze(1)
                    mask_float = mask.float().unsqueeze(-1)
                    pooled_gen = (generated_embeddings * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-9)
                    
                    listener_logits = self.listener.classifier(pooled_gen)
                    
                    # Decodificar
                    probs = torch.softmax(listener_logits, dim=-1)
                    top1_probs, top1_indices = torch.max(probs, dim=-1)
                    
                    predictions = [self.class_names[idx] for idx in top1_indices]
                    confidences = [p.item() for p in top1_probs]
                    
                    return words, predictions, confidences
                    
        finally:
            for m, was in zip(modules_to_eval, was_training):
                m.train(was)

            
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch"
            }
        }
