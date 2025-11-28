"""
Definición de modelos para TinySpeak
"""
import torch
import torch.nn.functional as F
from torch.nn import Module, LSTM, Linear, Sequential, ReLU, AdaptiveAvgPool2d, Conv2d, MaxPool2d
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_sequence
from transformers import Wav2Vec2Model, Wav2Vec2Config
from collections import OrderedDict
import torch.nn as nn

# ==========================================
# CUSTOM TINY LISTENER (Mini-Wav2Vec2)
# ==========================================

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # El Transformer espera [Batch, SeqLen, Dim] si batch_first=True
        # Nuestra implementación usa batch_first=True en el encoder layer
        # Pero la implementación estándar de PE suele esperar [SeqLen, Batch, Dim]
        # Ajustamos para soportar batch_first=True
        
        # x: [Batch, SeqLen, Dim]
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

class PhonologicalPathway(nn.Module):
    """
    Arquitectura personalizada "PhonologicalPathway" entrenada desde cero.
    Combina:
    1. MelSpectrogram: Convierte waveform -> Time-Frequency representation.
    2. Feature Extractor (CNN): Procesa el espectrograma.
    3. Positional Encoding: Añade información temporal.
    4. Context Encoder (Transformer): Procesa dependencias temporales.
    5. Classifier Head: Predice la palabra.
    """
    def __init__(
        self, 
        num_classes: int,
        hidden_dim: int = 256, 
        num_conv_layers: int = 3,
        num_transformer_layers: int = 2,
        nhead: int = 4,
        sample_rate: int = 16000,
        n_mels: int = 80
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 0. Audio Transform (Waveform -> MelSpectrogram)
        # Usamos torchaudio para calcular el espectrograma en la GPU
        try:
            import torchaudio
            self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_mels=n_mels,
                n_fft=400,
                hop_length=160
            )
        except ImportError:
            raise ImportError("torchaudio es necesario para PhonologicalPathway. Instálalo con pip install torchaudio")

        # 1. Feature Extractor (CNN 1D)
        # Entrada: (B, n_mels, T_spec) -> Salida: (B, hidden_dim, T')
        # Tratamos las bandas de frecuencia (n_mels) como canales de entrada
        layers = []
        in_channels = n_mels
        
        for i in range(num_conv_layers):
            out_channels = hidden_dim if i == num_conv_layers - 1 else 64 * (2**i)
            # Kernel size más pequeño y stride menor porque el espectrograma ya está "comprimido" temporalmente
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2))
            layers.append(nn.GroupNorm(out_channels // 8 if out_channels > 8 else 1, out_channels))
            layers.append(nn.GELU())
            in_channels = out_channels
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # Proyección para asegurar dimensión correcta para Transformer
        self.post_extract_proj = nn.Linear(in_channels, hidden_dim)

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=0.1)

        # 3. Context Encoder (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # 4. Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Para compatibilidad con Reader (target layer)
        self.target_layer = num_transformer_layers - 1

    def extract_features(self, waveforms):
        # waveforms: (B, T)
        
        # 0. Mel Spectrogram
        # (B, T) -> (B, n_mels, T_spec)
        x = self.mel_spectrogram(waveforms)
        
        # Log-Mel Spectrogram (estabilidad numérica y mejor rango dinámico)
        x = torch.log(x + 1e-9)
        
        # 1. CNN Feature Extractor
        # Entrada: (B, n_mels, T_spec)
        features = self.feature_extractor(x) # (B, C, T')
        
        return features.transpose(1, 2) # (B, T', C)

    def extract_hidden_activations(self, waveforms):
        """
        Interfaz requerida por TinyReader.
        Retorna los hidden states del Transformer.
        """
        features = self.extract_features(waveforms)
        features = self.post_extract_proj(features)
        
        # Add Positional Encoding
        features = self.pos_encoder(features)
        
        # Transformer espera (B, T, D)
        encoded = self.transformer(features)
        
        # Reader espera (Layers, B, T, D). Hacemos fake stack
        return encoded.unsqueeze(0) 

    def mask_hidden_activations(self, hidden_activations):
        # hidden_activations: (1, B, T, D) -> Tomamos el único layer
        hidden = hidden_activations[0] 
        B, T, D = hidden.shape
        lengths = torch.full((B,), T, dtype=torch.long, device=hidden.device)
        return hidden.unsqueeze(0), lengths

    def downsample_hidden_activations(self, hidden_activations, lengths, factor=1):
        # No necesitamos downsample extra si la CNN ya redujo bastante
        return hidden_activations, lengths

    def forward(self, waveforms):
        # 1. Features (incluye MelSpectrogram)
        features = self.extract_features(waveforms) # (B, T', C)
        features = self.post_extract_proj(features) # (B, T', D)
        
        # 2. Positional Encoding
        features = self.pos_encoder(features)
        
        # 3. Transformer
        encoded = self.transformer(features) # (B, T', D)
        
        # 4. Classification (Mean Pooling)
        pooled = encoded.mean(dim=1) # (B, D)
        logits = self.classifier(pooled)
        
        return logits, encoded


# ==========================================
# VISUAL PATHWAY (formerly TinyRecognizer)
# ==========================================

class VisualPathway(nn.Module):
    """
    Arquitectura personalizada "VisualPathway" (antes TinyRecognizer).
    Inspirada en CORnet-Z pero simplificada.
    """
    def __init__(self, num_classes: int, hidden_dim: int = 512):
        super().__init__()
        
        def conv_block(in_c, out_c, k=3, s=1, p=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, k, s, p),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )
            
        self.features = nn.Sequential(
            conv_block(3, 64),    # 64 -> 32
            conv_block(64, 128),  # 32 -> 16
            conv_block(128, 256), # 16 -> 8
            conv_block(256, hidden_dim), # 8 -> 4
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.hidden_dim = hidden_dim

    def update_num_classes(self, num_classes: int) -> None:
        device = next(self.parameters()).device
        self.classifier = nn.Linear(self.hidden_dim, num_classes).to(device)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        
        # Retornamos logits, embeddings (x) para compatibilidad con Reader
        return logits, x







class TinyReader(Module):
    """
    Modelo Generativo (Top-Down): Secuencia de Letras (Logits) -> Imaginación Auditiva (Wav2Vec2 Embeddings).
    Arquitectura Seq2Seq: Encoder (Lee letras) -> Decoder (Imagina audio).
    """
    def __init__(
        self, 
        input_dim: int, # Dimensión de los logits de entrada (ej. 26 letras)
        hidden_dim: int = 256, 
        wav2vec_dim: int = 768, 
        num_layers: int = 2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.wav2vec_dim = wav2vec_dim
        
        # Encoder: Procesa la secuencia de logits de las letras
        # Input: (B, L_text, input_dim)
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1, # Encoder simple
            batch_first=True
        )
        
        # Decoder: Genera la secuencia temporal de audio
        # Input: (B, L_audio, hidden_dim) - Inicializado con el estado del encoder
        self.decoder = nn.LSTM(
            input_size=hidden_dim, # Entrada en cada paso (contexto del encoder repetido o autoregresivo)
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Proyección de salida: Latente -> Embedding Wav2Vec2
        self.output_projection = nn.Linear(hidden_dim, wav2vec_dim)

    def forward(self, x_seq, target_length=None):
        """
        x_seq: (B, L_text, input_dim) - Secuencia de logits de letras (del TinyRecognizer)
        target_length: int - Longitud de la secuencia de audio a generar.
        """
        B = x_seq.size(0)
        
        # 1. Encoder (Leer el texto)
        # encoder_out: (B, L_text, hidden_dim)
        # (h_n, c_n): Estado final del encoder -> Contexto para el decoder
        _, (h_n, c_n) = self.encoder(x_seq)
        
        # Usamos el último estado oculto como representación del concepto global
        # h_n: (num_layers, B, hidden_dim). Tomamos el último layer si num_layers > 1
        context_vector = h_n[-1] # (B, hidden_dim)
        
        # 2. Preparar entrada para el Decoder (Imaginación)
        # Repetimos el contexto para cada paso de tiempo (como un "bias" constante)
        if target_length is None:
            target_length = 100 
            
        # (B, 1, hidden_dim) -> (B, L_audio, hidden_dim)
        decoder_input = context_vector.unsqueeze(1).expand(-1, target_length, -1)
        
        # 3. Decoder (Generar audio)
        # Pasamos el estado del encoder como estado inicial del decoder?
        # Para simplificar y conectar ambos, podemos inicializar el decoder con ceros 
        # y darle el contexto como entrada en cada paso (hecho arriba).
        # O inicializar con el estado del encoder. Vamos a hacer ambas cosas para máximo flujo.
        
        # Expandir estado del encoder para matchear num_layers del decoder si son diferentes
        # Aquí asumimos que queremos inicializar el decoder.
        # Si decoder tiene más layers, repetimos o rellenamos.
        # Por simplicidad, dejamos que el decoder arranque de 0 pero reciba el contexto fuerte en la entrada.
        
        decoder_out, _ = self.decoder(decoder_input)
        
        # 4. Proyectar a espacio Wav2Vec2
        # (B, L_audio, 768)
        generated_embeddings = self.output_projection(decoder_out)
        
        return generated_embeddings
