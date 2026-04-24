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

from models import TinyReader, TinyEars, TinyEyes, TinySpeller, TinyReaderP2W
from utils.audio import AUDIO_EMBED_DIM
from utils.graphemes import get_phonemes_from_word, tokenize_graphemes, get_phoneme_inventory
from utils.plotting import SoftDTW
from training.config import load_master_dataset_config

class TinyReaderLightning(pl.LightningModule):
    """
    LightningModule para TinyReader.
    Aprende a generar representaciones auditivas a partir de secuencias de letras (Spelling).
    Usa TinyEyes para "leer" las letras y TinyEars como "Oído Interno".
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
        w_dtw: float = 0.0,      # Legacy
        w_perceptual: float = 0.5, # Categorical (Cross-Entropy)
        w_mse: float = 1.0,         # Structural (Neural Image Alignment)
        use_two_stage: bool = False,
        phoneme_listener_checkpoint_path: str = None,
        target_language: str | None = None,
        training_phase: str = "end_to_end", # 'g2p', 'p2w', 'end_to_end'
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.class_names = list(class_names)
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
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
            
        # --- NUEVO: Alineación de Dimensiones (Word Listener) ---
        num_word_classes = len(self.class_names)
        if self.listener.classifier.out_features != num_word_classes:
            print(f"⚠️ Desajuste en Word Listener: {self.listener.classifier.out_features} clases → {num_word_classes} (Actualizando...)")
            self.listener.update_num_classes(num_word_classes)

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
            
            # --- NUEVO: Alineación de Dimensiones (Phoneme Listener) ---
            num_phoneme_classes = len(self.phoneme_class_names)
            if self.phoneme_listener.classifier.out_features != num_phoneme_classes:
                print(f"⚠️ Desajuste en Phoneme Listener: {self.phoneme_listener.classifier.out_features} clases → {num_phoneme_classes} (Actualizando...)")
                self.phoneme_listener.update_num_classes(num_phoneme_classes)

            # Inicializar banco de embeddings de fonemas (Canonical Phoneme Embeddings)
            self.register_buffer("phoneme_embeddings_bank", torch.zeros(len(self.phoneme_class_names), self.phoneme_listener.hidden_dim))
            self._init_phoneme_bank()
            
            # --- Máscara de Aislamiento Lingüístico ---
            if self.hparams.target_language:
                inventory = set(get_phoneme_inventory(self.hparams.target_language))
                mask = torch.zeros(len(self.phoneme_class_names), dtype=torch.bool)
                for i, p in enumerate(self.phoneme_class_names):
                    if p in inventory:
                        mask[i] = True
                self.register_buffer("phoneme_mask", mask)
            else:
                self.register_buffer("phoneme_mask", torch.ones(len(self.phoneme_class_names), dtype=torch.bool))

            # --- NUEVO: Máscara de Palabras (Linguistic Isolation Stage 2) ---
            word_mask = torch.ones(len(self.class_names), dtype=torch.bool)
            # Para el Reader, el vocabulario ya está filtrado por idioma en el constructor (self.class_names)
            # Pero si el listener fuera global, esto nos protege.
            self.register_buffer("word_mask", word_mask)
        
        # 1.2 Word Embeddings Bank (Para la alineación estructural del Reader)
        self.register_buffer("word_embeddings_bank", torch.zeros(len(self.class_names), self.listener.hidden_dim))
        self._init_word_bank()
        
        # 2. Modelo Visual (Recognizer - Ojo)
        self.recognizer = self._load_recognizer(recognizer_checkpoint_path)
        self.recognizer.eval()
        for p in self.recognizer.parameters():
            p.requires_grad = False
            
        # Obtener dimensión del espacio latente del recognizer (Neural Image)
        if hasattr(self.recognizer, "hidden_dim"):
            input_dim = self.recognizer.hidden_dim
        else:
            # Fallback a logits si el modelo no tiene hidden_dim definido
            input_dim = self.recognizer.classifier.out_features if self.recognizer.classifier else 26

        # 3. Modelo Generativo (Reader)
        if self.use_two_stage:
            # Stage 1: Grapheme -> Phoneme
            # Output dim debe coincidir con hidden_dim del PhonemeListener (256)
            self.reader_g2p = TinySpeller(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=self.phoneme_listener.hidden_dim, 
                num_layers=num_layers
            )
            # Stage 2: Phoneme -> Word
            # Input dim = PhonemeListener hidden dim
            # Output dim = Word Listener hidden dim (AUDIO_EMBED_DIM o 256)
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
            target_dim = self.listener.hidden_dim
            # Usar TinyReaderP2W para consistencia (es un integrador secuencial)
            self.reader = TinyReaderP2W(
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
        from utils.audio import load_waveform
        
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

    def _init_word_bank(self):
        """Inicializa el banco de embeddings de palabras usando el listener congelado."""
        print("Inicializando banco de embeddings de palabras (Centroides)...")
        from utils.audio import load_waveform
        
        repo_root = Path(self.dataset_config.get("repo_root", "."))
        device = self.device if hasattr(self, "device") else "cpu"
        self.listener.to(device)
        
        with torch.no_grad():
            for i, word in enumerate(self.class_names):
                # Buscar directorio de la palabra en custom_dataset
                word_dir = repo_root / "data" / "audios" / "custom_dataset" / word
                if not word_dir.exists():
                    continue
                    
                samples = list(word_dir.glob("*.wav"))
                if not samples:
                    continue
                
                # Tomar hasta 5 samples para promediar el centroide
                selected_samples = samples[:5]
                embeddings = []
                
                for s_path in selected_samples:
                    try:
                        waveform = load_waveform(str(s_path)).to(device)
                        waveform = waveform.unsqueeze(0)
                        
                        # Extraer embedding del listener (Frozen)
                        # (Layers, Batch, Time, Dim) -> (1, 1, T, D)
                        emb = self.listener.extract_hidden_activations(waveform)
                        # Pooling temporal
                        pooled = emb.mean(dim=2).squeeze()
                        embeddings.append(pooled)
                    except Exception as e:
                        print(f"Error procesando word sample {word}: {e}")
                
                if embeddings:
                    avg_emb = torch.stack(embeddings).mean(dim=0)
                    self.word_embeddings_bank[i] = avg_emb
        
        print(f"Banco de palabras inicializado. {len(self.class_names)} palabras.")

    def _load_recognizer(self, checkpoint_path: str) -> TinyEyes:
        """Carga TinyEyes desde checkpoint para usarlo como ojo."""
        import importlib
        import json
        RecognizerPL = importlib.import_module("training.visual_module").TinyEyesLightning
        
        print(f"Cargando TinyEyes (Ojo) desde {checkpoint_path}...")
        
        # 1. Intentar obtener clases desde metadata
        meta_path = Path(checkpoint_path).with_suffix(".ckpt.meta.json")
        classes = None
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                classes = meta.get("config", {}).get("classes", [])
            except Exception as e:
                print(f"Error leyendo metadata del recognizer: {e}")

        # 2. Cargar checkpoint
        try:
            if classes:
                pl_module = RecognizerPL.load_from_checkpoint(
                    checkpoint_path,
                    class_names=classes,
                    map_location=self.device if hasattr(self, "device") else "cpu"
                )
            else:
                # Fallback sin argumentos si no hay metadata
                pl_module = RecognizerPL.load_from_checkpoint(
                    checkpoint_path,
                    map_location=self.device if hasattr(self, "device") else "cpu"
                )
        except Exception as e:
            # Si falla por falta de class_names, intentamos un fallback desesperado
            print(f"Fallo carga de recognizer: {e}. Reintentando con fallback...")
            # Aquí no tenemos una lista de grafemas a mano, 
            # pero el recognizer suele ser standard. 
            # El error reportado es justamente por falta de este argumento.
            raise e

        # Asegurar que el modelo tenga class_names
        if hasattr(pl_module, 'class_names'):
             pl_module.model.class_names = pl_module.class_names
             
        return pl_module.model

    def _get_word_images(self, word: str) -> torch.Tensor:
        """
        Obtiene una secuencia de imágenes para deletrear la palabra.
        Retorna: (L, C, H, W)
        """
        images = []
        repo_root = Path(self.dataset_config.get("repo_root", "."))
        
        # Obtener grafemas disponibles permitidos para el idioma
        all_graphemes = list(self.visual_config.keys())
        if self.hparams.target_language:
            from utils.graphemes import get_language_letters
            allowed = set(get_language_letters(self.hparams.target_language))
            available_graphemes = [g for g in all_graphemes if g.lower() in allowed]
        else:
            available_graphemes = all_graphemes
            
        from utils.graphemes import tokenize_graphemes
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

    def _load_listener(self, checkpoint_path: str) -> TinyEars:
        """Carga TinyEars desde checkpoint para usarlo como juez."""
        from training.audio_module import TinyEarsLightning as ListenerPL
        import json
        
        print(f"Cargando TinyEars desde {checkpoint_path}...")
        
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
        # Datos comunes
        labels = batch["label"].to(self.device)
        batch_size = labels.size(0)
        is_atomic = batch.get("is_atomic", False)

        # 1. Obtener Ground Truth Embeddings (Solo si no es atómico)
        if not is_atomic:
            waveforms = [w.to(self.device) for w in batch["waveforms"]]
            with torch.no_grad():
                waveforms_padded = pad_sequence(waveforms, batch_first=True)
                real_embeddings = self.listener.extract_hidden_activations(waveforms_padded)
                real_embeddings, lengths = self.listener.mask_hidden_activations(real_embeddings)
                real_embeddings, lengths = self.listener.downsample_hidden_activations(real_embeddings, lengths, factor=7)
                real_embeddings = real_embeddings.squeeze(0)
            
        # 2. Generar Imaginación (Top-Down)
        if "label_str" in batch:
            words = batch["label_str"]
        else:
            words = [self.class_names[i] for i in labels]
        
        # A. Obtener secuencia de representaciones visuales (Neural Images)
        visual_embeddings_sequences = []
        
        if is_atomic:
            # MODO ATÓMICO: Una imagen por muestra (B, 1, C, H, W)
            images_batch = batch["image"].to(self.device)
            # Ya viene con forma (B, 1, C, H, W) desde el collate
            for b in range(batch_size):
                with torch.no_grad():
                    res = self.recognizer(images_batch[b]) # procesar la secuencia L=1
                    word_emb = res[1] if isinstance(res, tuple) else res
                visual_embeddings_sequences.append(word_emb)
        elif "image" in batch:
            images_batch = batch["image"].to(self.device) # (B, L, C, H, W)
            for b in range(batch_size):
                word_images = images_batch[b]
                with torch.no_grad():
                    res = self.recognizer(word_images)
                    # Extraer embedding (segundo elemento)
                    word_emb = res[1] if isinstance(res, tuple) else res
                visual_embeddings_sequences.append(word_emb)
        else:
            for word in words:
                images = self._get_word_images(word).to(self.device)
                with torch.no_grad():
                    res = self.recognizer(images)
                    # Extraer embedding
                    word_emb = res[1] if isinstance(res, tuple) else res
                visual_embeddings_sequences.append(word_emb)
            
        padded_visual_embeddings = pad_sequence(visual_embeddings_sequences, batch_first=True, padding_value=0.0)
        
        if is_atomic:
            # En modo atómico, el entrenamiento es puramente G2P (Grapheme-to-Phoneme)
            # targets son directamente las labels
            phoneme_targets_padded = labels.unsqueeze(1) # (B, 1)
            
            # Generar Embeddings (Imaginación)
            generated_phoneme_embeddings = self.reader_g2p(padded_visual_embeddings) # (B, 1, D)
            
            phoneme_targets_flat = phoneme_targets_padded.reshape(-1)
            
            # --- CÁLCULO DE PÉRDIDAS ---
            # Asegurar que los índices sean válidos para el banco de embeddings (Blindaje CUDA)
            max_idx = self.phoneme_embeddings_bank.size(0) - 1
            labels_safe = labels.clamp(0, max_idx)
            
            # Detectar desajuste de datos
            if (labels > max_idx).any():
                print(f"⚠️ Alerta: Detectados índices de fonemas fuera de rango ({labels.max().item()} > {max_idx}). Clamping aplicado.")

            # 1. Pérdida Estructural
            target_embeddings = self.phoneme_embeddings_bank[labels_safe]
            loss_structural = F.mse_loss(generated_phoneme_embeddings.squeeze(1), target_embeddings)
            
            # 2. Pérdida Categórica
            phoneme_logits = self.phoneme_listener.classifier(generated_phoneme_embeddings)
            phoneme_logits = torch.where(self.phoneme_mask, phoneme_logits, phoneme_logits.new_tensor(-1e9))
            phoneme_logits_flat = phoneme_logits.view(-1, phoneme_logits.size(-1))
            
            # targets son directamente las labels
            phoneme_targets_flat = labels_safe
            loss_categorical = F.cross_entropy(phoneme_logits_flat, phoneme_targets_flat)
            
            total_loss = (self.hparams.w_mse * loss_structural) + (self.hparams.w_perceptual * loss_categorical)
            
            # Métricas
            with torch.no_grad():
                preds = torch.argmax(phoneme_logits_flat, dim=1)
                acc = (preds == phoneme_targets_flat).float().mean()
                self.log(f"{stage}_phoneme_acc", acc, prog_bar=True)
            
            self.log(f"{stage}_g2p_structural_mse", loss_structural)
            self.log(f"{stage}_g2p_categorical_ce", loss_categorical)
            self.log(f"{stage}_loss", total_loss)
            return total_loss

        # --- MODO SECUENCIAL (Reader/P2W) ---
        if self.use_two_stage:
            # --- STAGE 1: G2P ---
            
            # 1. Calcular targets (fonemas) para cada palabra
            phoneme_targets_list = []
            
            # Obtener inventario filtrado para tokenización precisa
            if self.hparams.target_language:
                from utils.graphemes import get_phoneme_inventory
                allowed_phonemes = set(get_phoneme_inventory(self.hparams.target_language))
                filtered_phoneme_names = [p for p in self.phoneme_class_names if p in allowed_phonemes]
            else:
                filtered_phoneme_names = self.phoneme_class_names

            for word in words:
                # Tokenizar usando el inventario filtrado para evitar crossovers lingüísticos
                tokens = tokenize_graphemes(word.lower(), filtered_phoneme_names)
                idxs = [self.phoneme_to_idx[t] for t in tokens if t in self.phoneme_to_idx]
                if not idxs:
                    # Fallback: tomar letra a letra pero comprobando la máscara
                    idxs = []
                    for c in word.lower():
                        if c in self.phoneme_to_idx:
                            idx = self.phoneme_to_idx[c]
                            if self.phoneme_mask[idx]:
                                idxs.append(idx)
                    if not idxs: idxs = [0]
                phoneme_targets_list.append(torch.tensor(idxs, device=self.device))

            # 2. Determinar longitud máxima para este batch
            # En G2P, queremos que el modelo genere tantos fonemas como sea necesario
            # 3. Generar Embeddings (Imaginación)
            # En la nueva arquitectura, reader_g2p es un proyector frame-wise.
            if self.training_phase == "p2w":
                with torch.no_grad():
                    generated_phoneme_embeddings = self.reader_g2p(padded_visual_embeddings)
            else:
                generated_phoneme_embeddings = self.reader_g2p(padded_visual_embeddings)
            
            # 4. Padear targets para coincidir con el input visual (padded_logits)
            # El projector mantiene la longitud de la secuencia de entrada.
            phoneme_targets_padded = pad_sequence(phoneme_targets_list, batch_first=True, padding_value=-100)
            
            # Asegurar coincidencia de longitudes si hay discrepancias de padding entre audio y visual
            L_gen = generated_phoneme_embeddings.size(1)
            L_tgt = phoneme_targets_padded.size(1)
            if L_gen > L_tgt:
                padding = torch.full((batch_size, L_gen - L_tgt), -100, device=self.device)
                phoneme_targets_padded = torch.cat([phoneme_targets_padded, padding], dim=1)
            elif L_tgt > L_gen:
                phoneme_targets_padded = phoneme_targets_padded[:, :L_gen]

            phoneme_targets_flat = phoneme_targets_padded.reshape(-1)
            
            # Si estamos en fase G2P, calculamos pérdida aquí y retornamos
            if self.training_phase == "g2p":
                # 1. Pérdida Estructural (MSE contra centroides)
                mask = (phoneme_targets_padded != -100)
                target_embeddings = self.phoneme_embeddings_bank[phoneme_targets_padded.clamp(min=0)]
                loss_structural = F.mse_loss(generated_phoneme_embeddings[mask], target_embeddings[mask])
                
                # 2. Pérdida Categórica (Cross-Entropy contra el clasificador)
                phoneme_logits = self.phoneme_listener.classifier(generated_phoneme_embeddings)
                
                # APLICAR MÁSCARA LINGÜÍSTICA: Invalidar fonemas fuera del idioma
                # Usamos torch.where porque soporta broadcasting (B, T, C) vs (C)
                # Usamos new_tensor para asegurar que coincida el dtype (fp16/fp32) y el device
                neg_inf = phoneme_logits.new_tensor(-1e9)
                phoneme_logits = torch.where(self.phoneme_mask, phoneme_logits, neg_inf)
                
                phoneme_logits_flat = phoneme_logits.view(-1, phoneme_logits.size(-1))
                loss_categorical = F.cross_entropy(phoneme_logits_flat, phoneme_targets_flat)
                
                # 3. Combinación Híbrida
                total_loss = (self.hparams.w_mse * loss_structural) + (self.hparams.w_perceptual * loss_categorical)
                
                # Métricas de Monitoreo
                with torch.no_grad():
                    valid_mask_flat = phoneme_targets_flat != -100
                    if valid_mask_flat.any():
                        # Asegurar que no hay índices fuera de rango antes de argmax
                        masked_logits = phoneme_logits_flat[valid_mask_flat]
                        if masked_logits.size(0) > 0:
                            preds = torch.argmax(masked_logits, dim=1)
                            acc = (preds == phoneme_targets_flat[valid_mask_flat]).float().mean()
                            self.log(f"{stage}_phoneme_acc", acc, prog_bar=True)

                self.log(f"{stage}_g2p_structural_mse", loss_structural)
                self.log(f"{stage}_g2p_categorical_ce", loss_categorical)
                self.log(f"{stage}_loss", total_loss)
                
                return total_loss
            
            # --- STAGE 2: P2W (Phoneme Sequence -> Word Vector) ---
            # reader_p2w toma la secuencia y devuelve un ÚNICO vector de palabra.
            generated_embeddings = self.reader_p2w(generated_phoneme_embeddings)
            
        else:
            # Single Stage: Grapheme -> Word
            # reader es un TinyReaderP2W (integrador secuencial)
            generated_embeddings = self.reader(padded_visual_embeddings)
        
        # A. Pérdida Estructural (MSE contra centroides de palabra)
        # Clamping defensivo para evitar CUDA device-side assert si un label está fuera de rango
        labels_safe = labels.clamp(0, self.word_embeddings_bank.size(0) - 1)
        target_word_embeddings = self.word_embeddings_bank[labels_safe]
        loss_structural = F.mse_loss(generated_embeddings, target_word_embeddings)
        
        # B. Pérdida Categórica (Cross-Entropy contra el clasificador de palabras)
        word_logits = self.listener.classifier(generated_embeddings)
        
        # Aplicar máscara de palabras (Linguistic Isolation)
        # Usar new_tensor para compatibilidad de precisión (fp16/fp32)
        neg_inf_word = word_logits.new_tensor(-1e9)
        word_logits = torch.where(self.word_mask, word_logits, neg_inf_word)
        
        loss_categorical = F.cross_entropy(word_logits, labels)
        
        # C. Combinación Híbrida
        total_loss = (self.hparams.w_mse * loss_structural) + (self.hparams.w_perceptual * loss_categorical)
        
        # Logging de métricas
        with torch.no_grad():
            acc_word = (torch.argmax(word_logits, dim=1) == labels).float().mean()
            self.log(f"{stage}_word_acc", acc_word, prog_bar=True)
            # Alias genérico para la interfaz
            self.log(f"{stage}_acc", acc_word, sync_dist=True)

        self.log(f"{stage}_word_structural_mse", loss_structural)
        self.log(f"{stage}_word_categorical_ce", loss_categorical)
        self.log(f"{stage}_loss", total_loss, prog_bar=True)
        
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
                is_atomic = batch.get("is_atomic", False)
                labels = batch["label"].to(self.device)
                
                if not is_atomic:
                    waveforms = [w.to(self.device) for w in batch["waveforms"]]
                    # Ground Truth info
                    waveforms_padded = pad_sequence(waveforms, batch_first=True)
                    real_embeddings = self.listener.extract_hidden_activations(waveforms_padded)
                    real_embeddings, lengths = self.listener.mask_hidden_activations(real_embeddings)
                    real_embeddings, lengths = self.listener.downsample_hidden_activations(real_embeddings, lengths, factor=7)
                    max_len = real_embeddings.size(1)
                else:
                    max_len = 1
                
                # A. Obtener secuencia de representaciones visuales
                visual_embeddings_sequences = []
                
                if is_atomic:
                    images_batch = batch["image"].to(self.device)
                    # Procesar cada imagen de la secuencia L=1
                    for b in range(len(batch["label_str"])):
                        res = self.recognizer(images_batch[b])
                        word_emb = res[1] if isinstance(res, tuple) else res
                        visual_embeddings_sequences.append(word_emb)
                    words = list(batch["label_str"])
                else:
                    # Modo secuencial
                    if "label_str" in batch:
                        words = list(batch["label_str"])
                    else:
                        words = [self.class_names[int(i)] for i in labels]
                        
                    for word in words:
                        images = self._get_word_images(word).to(self.device)
                        res = self.recognizer(images)
                        word_emb = res[1] if isinstance(res, tuple) else res
                        visual_embeddings_sequences.append(word_emb)
                
                padded_visual_embeddings = pad_sequence(visual_embeddings_sequences, batch_first=True, padding_value=0.0)
                
                if self.training_phase == "g2p":
                    # --- G2P Phase: Predecir Fonemas ---
                    
                    # 1. Calcular targets reales para determinar longitud
                    phoneme_targets_list = []
                    
                    # Filtramos el inventario para que coincida con el idioma actual
                    if self.hparams.target_language:
                        from utils.graphemes import get_phoneme_inventory
                        allowed_phonemes = set(get_phoneme_inventory(self.hparams.target_language))
                        filtered_phoneme_names = [p for p in self.phoneme_class_names if p in allowed_phonemes]
                    else:
                        filtered_phoneme_names = self.phoneme_class_names

                    if is_atomic:
                        # En modo atómico, el "word" es el grafema, y el target es el fonema único
                        # words[b] ya es el token (ej. "sh")
                        for word in words:
                            idx = self.phoneme_to_idx.get(word, 0)
                            phoneme_targets_list.append([idx])
                    else:
                        # Modo secuencial: Tokenizar palabras en fonemas
                        for word in words:
                            tokens = tokenize_graphemes(word.lower(), filtered_phoneme_names)
                            idxs = [self.phoneme_to_idx[t] for t in tokens if t in self.phoneme_to_idx]
                            if not idxs:
                                idxs = []
                                for c in word.lower():
                                    if c in self.phoneme_to_idx:
                                        idx = self.phoneme_to_idx[c]
                                        # Si tenemos máscara, comprobarla
                                        if hasattr(self, 'phoneme_mask'):
                                            if self.phoneme_mask[idx]: idxs.append(idx)
                                        else:
                                            idxs.append(idx)
                                    if not idxs: idxs = [0]
                            phoneme_targets_list.append(idxs)
                    
                    # 2. Determinar max len real
                    max_target_len = max(len(t) for t in phoneme_targets_list)
                    
                    # 3. Generar
                    # En la nueva arquitectura, reader_g2p no usa target_length
                    generated_phoneme_embeddings = self.reader_g2p(padded_visual_embeddings)
                    
                    # Clasificar fonemas
                    phoneme_logits = self.phoneme_listener.classifier(generated_phoneme_embeddings)
                    
                    # APLICAR MÁSCARA LINGÜÍSTICA en la inferencia usando where (broadcasting support)
                    if hasattr(self, 'phoneme_mask'):
                        neg_inf = phoneme_logits.new_tensor(-1e9)
                        phoneme_logits = torch.where(self.phoneme_mask, phoneme_logits, neg_inf)
                    
                    # Decodificar
                    probs = torch.softmax(phoneme_logits, dim=-1)
                    top1_probs, top1_indices = torch.max(probs, dim=-1) # (B, T)
                    
                    predictions = []
                    targets = []
                    confidences = []
                    for b in range(len(words)):
                        indices = top1_indices[b]
                        target_indices = phoneme_targets_list[b]
                        
                        # Recortar predicciones a la longitud del target para visualización coherente
                        # o mostrar todo? Los targets aquí son la referencia.
                        L_target = len(target_indices)
                        pred_indices = indices[:L_target]
                        
                        pred_phonemes = [str(self.phoneme_class_names[int(idx)]) for idx in pred_indices]
                        target_phonemes = [str(self.phoneme_class_names[int(idx)]) for idx in target_indices]
                        
                        pred_str = " ".join(pred_phonemes)
                        target_str = " ".join(target_phonemes)
                        
                        predictions.append(pred_str)
                        targets.append(target_str)
                        confidences.append(top1_probs[b][:L_target].mean().item())
                        
                    # Retornamos (Input, Predictions, Confidences, Targets, Embeddings, Labels)
                    # Aplanamos embeddings y targets para el PCA de fonemas
                    all_embs = []
                    all_labs = []
                    for b in range(len(words)):
                        L = len(phoneme_targets_list[b])
                        all_embs.append(generated_phoneme_embeddings[b, :L].detach())
                        all_labs.append(torch.tensor(phoneme_targets_list[b]))
                    
                    final_embeddings = torch.cat(all_embs, dim=0) if all_embs else torch.empty(0)
                    final_labels = torch.cat(all_labs, dim=0) if all_labs else torch.empty(0)

                    return words, predictions, confidences, targets, final_embeddings, final_labels

                else:
                    # --- P2W / End-to-End Phase: Predecir Palabras ---
                    if self.use_two_stage:
                        # Generar fonemas (Pointwise)
                        phoneme_emb = self.reader_g2p(padded_visual_embeddings)
                        # Ensamblar palabra (Sequential Assembly) -> retorna un único vector
                        generated_embeddings = self.reader_p2w(phoneme_emb)
                    else:
                        # Integrador directo
                        generated_embeddings = self.reader(padded_visual_embeddings)
                    
                    # Clasificar palabras
                    word_logits = self.listener.classifier(generated_embeddings)

                    # Aplicar máscara de palabras si existe (aislamiento lingüístico)
                    if hasattr(self, 'word_mask'):
                        neg_inf_word = word_logits.new_tensor(-1e9)
                        word_logits = torch.where(self.word_mask, word_logits, neg_inf_word)
                    
                    # Decodificar palabra
                    probs = torch.softmax(word_logits, dim=-1)
                    top1_probs, top1_indices = torch.max(probs, dim=-1)
                    
                    predictions = [self.class_names[int(idx)] for idx in top1_indices]
                    confidences = top1_probs.tolist()

                    # Retornamos (Words, Predictions, Confidences, Targets (None), Embeddings, Labels)
                    word_targets = batch["label"].cpu()
                    return words, predictions, confidences, None, generated_embeddings.detach(), word_targets

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
