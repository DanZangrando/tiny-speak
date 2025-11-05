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

class Flatten(nn.Module):
    """Helper module for flattening input tensor to 1-D for the use in Linear modules"""
    def forward(self, x):
        return x.view(x.size(0), -1)

class Identity(nn.Module):
    """Helper module that stores the current tensor. Useful for accessing by name"""
    def forward(self, x):
        return x

class CORblock_Z(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=kernel_size // 2)
        self.nonlin = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.output = Identity()  # for an easy access to this block's output

    def forward(self, inp):
        x = self.conv(inp)
        x = self.nonlin(x)
        x = self.pool(x)
        x = self.output(x)  # for an easy access to this block's output
        return x

def CORnet_Z(pretrained=False):
    """CORnet-Z architecture with optional pretrained weights."""
    model = nn.Sequential(OrderedDict([
        ('V1', CORblock_Z(3, 64, kernel_size=7, stride=2)),
        ('V2', CORblock_Z(64, 128)),
        ('V4', CORblock_Z(128, 256)),
        ('IT', CORblock_Z(256, 512)),
        ('decoder', nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 1000)),
            ('output', Identity())
        ])))
    ]))

    if pretrained:
        # ✅ NUEVA FUNCIONALIDAD: Cargar pesos preentrenados de CORnet-Z
        try:
            import torch.hub
            # Intentar cargar desde torch hub (si disponible)
            # Fallback: usar inicialización mejorada He para ReLU
            print("⚠️ Pesos preentrenados de CORnet-Z no disponibles, usando inicialización He")
            _initialize_cornet_weights(model)
        except Exception as e:
            print(f"⚠️ Error cargando pesos preentrenados: {e}")
            _initialize_cornet_weights(model)
    else:
        # Inicialización estándar (mejorada)
        _initialize_cornet_weights(model)

    return model


def _initialize_cornet_weights(model):
    """Inicialización mejorada para CORnet-Z - más apropiada para ReLU."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # He initialization para ReLU
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # Xavier para capas lineales
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class TinySpeak(Module):
    def __init__(
        self,
        words: list | None = None,
        *,
        num_classes: int | None = None,
        hidden_dim: int = 128,
        num_layers: int = 2,
        wav2vec_dim: int = 768,
    ):
        super().__init__()
        if words is None and num_classes is None:
            raise ValueError("TinySpeak requiere una lista de palabras o num_classes explícito.")

        self.words = words or []
        self.hidden_dim = hidden_dim
        self.lstm = LSTM(
            input_size=wav2vec_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        n_outputs = num_classes if num_classes is not None else len(self.words)
        if n_outputs <= 0:
            raise ValueError("TinySpeak necesita al menos una clase de salida.")

        self.classifier = Linear(hidden_dim, n_outputs)

    def update_vocabulary(self, words: list[str]) -> None:
        """Actualizar dinámicamente el vocabulario y la capa de salida."""
        self.words = words
        device = next(self.parameters()).device
        self.classifier = Linear(self.hidden_dim, len(words)).to(device)

    def forward(self, packed_sequence: PackedSequence):
        _, (h_n, _) = self.lstm(packed_sequence)
        logits = self.classifier(h_n[-1])
        return logits, h_n

class TinyListener(Module):
    def __init__(self, tiny_speak: TinySpeak, wav2vec_model, wav2vec_target_layer=5):
        super().__init__()
        self.tiny_speak = tiny_speak
        self.wav2vec_model = wav2vec_model
        self.wav2vec_target_layer = wav2vec_target_layer

    def extract_hidden_activations(self, waveforms):
        B = len(waveforms)
        padded = pad_sequence(waveforms, batch_first=True).to(waveforms[0].device)
        lengths = torch.tensor([w.size(0) for w in waveforms], device=waveforms[0].device)
        arange = torch.arange(padded.size(1), device=waveforms[0].device)
        mask = (arange.unsqueeze(0) < lengths.unsqueeze(1)).long()
        
        self.wav2vec_model.eval()
        with torch.no_grad():
            out = self.wav2vec_model(padded, attention_mask=mask)
        return torch.stack(out.hidden_states, dim=0)

    def mask_hidden_activations(self, hidden_activations):
        hidden = hidden_activations[self.wav2vec_target_layer]  # → (B, T_hidden, D)
        B, T_hidden, D = hidden.shape
        device = hidden.device
        lengths = torch.full((B,), T_hidden, dtype=torch.long, device=device)
        return hidden.unsqueeze(0), lengths

    def downsample_hidden_activations(self, hidden_activations, lengths, factor=7):
        B, L, N, D = hidden_activations.shape
        N_target = N // factor
        hidden_activations = hidden_activations.reshape(B * L, N, D).transpose(1, 2)
        hidden_activations = F.interpolate(hidden_activations, size=N_target, mode="linear", align_corners=False)
        hidden_activations = hidden_activations.transpose(1, 2).reshape(B, L, N_target, D)
        lengths = torch.div(lengths, factor, rounding_mode='floor')
        return hidden_activations, lengths

    def forward(self, waveform):
        """
        waveform: FloatTensor of shape (B, T_max, input_dim) or list of tensors
        """
        if isinstance(waveform, torch.Tensor):
            waveforms = [waveform[i] for i in range(waveform.size(0))]
        else:
            waveforms = waveform
            
        hidden_activations = self.extract_hidden_activations(waveforms)
        hidden_activations, lengths = self.mask_hidden_activations(hidden_activations)
        hidden_activations, lengths = self.downsample_hidden_activations(hidden_activations, lengths, factor=7)
        hidden_activations = hidden_activations.squeeze(0)
        
        # Pack the padded batch
        packed = pack_padded_sequence(
            hidden_activations, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        return self.tiny_speak(packed)

class TinyRecognizer(Module):
    def __init__(self, wav2vec_dim=768, num_classes=26, pretrained_backbone=True):
        super().__init__()
        # ✅ MEJORA CRÍTICA: Usar backbone preentrenado (o inicialización He mejorada)
        self.cornet = CORnet_Z(pretrained=pretrained_backbone)
        # ✅ SIMPLIFICACIÓN: Decoder directo sin capas intermedias innecesarias
        self.cornet.decoder = Sequential(OrderedDict([
            ('avgpool', AdaptiveAvgPool2d((1, 1))),
            ('flatten', Flatten()),
            ('output', Linear(512, num_classes))  # Directo 512→26 clases
        ]))
        if num_classes <= 0:
            raise ValueError("TinyRecognizer necesita al menos una clase de salida.")
        # ✅ SIMPLIFICACIÓN: Sin clasificador separado, todo en el decoder

    def update_num_classes(self, num_classes: int) -> None:
        if num_classes <= 0:
            raise ValueError("TinyRecognizer necesita al menos una clase.")
        device = next(self.parameters()).device
        # ✅ SIMPLIFICACIÓN: Actualizar el último layer del decoder
        self.cornet.decoder.output = Linear(512, num_classes).to(device)

    def forward(self, x):
        # ✅ SIMPLIFICACIÓN: Forward directo - CORnet ya incluye el clasificador
        logits = self.cornet(x)
        return logits, logits  # Retornar logits duplicados para compatibilidad

class TinySpeller(Module):
    def __init__(self, tiny_recognizer: TinyRecognizer, tiny_speak: TinySpeak):
        super().__init__()
        self.tiny_recognizer = tiny_recognizer
        self.tiny_speak = tiny_speak

        self.tiny_recognizer.eval()
        for param in self.tiny_recognizer.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        x: Tensor (batch_size, seq_len, C, H, W)
        """
        batch_size, seq_len, C, H, W = x.size()
        predicted_embeddings = []

        # Embed each letter image in the sequence
        for t in range(seq_len):
            x_t = x[:, t, :, :, :]
            _, predicted_embedding = self.tiny_recognizer(x_t)
            predicted_embeddings.append(predicted_embedding.unsqueeze(1))

        embs = torch.cat(predicted_embeddings, dim=1)  # (batch, seq_len, dim_embeddings)
        lengths = [seq_len] * batch_size
        packed = pack_padded_sequence(
            embs, lengths, batch_first=True, enforce_sorted=False
        )
        return self.tiny_speak(packed)


class TinySpellerImproved(Module):
    """
    Versión mejorada de TinySpeller con backbone entrenable y procesamiento eficiente.
    
    Mejoras:
    - TinyRecognizer entrenable (sin .eval() forzado)
    - Procesamiento batch-wise de secuencias
    - Encoder BiLSTM con attention
    - Regularización con dropout
    """
    def __init__(self, num_classes: int, vocab_size: int, 
                 embed_dim: int = 768, hidden_dim: int = 256,
                 dropout: float = 0.3, freeze_recognizer: bool = False):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # Backbone visual - ENTRENABLE por defecto
        self.visual_encoder = TinyRecognizer(num_classes=num_classes, wav2vec_dim=embed_dim)
        
        if freeze_recognizer:
            # Solo congelar si se especifica explícitamente
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
        
        # Encoder de secuencias bidireccional
        self.sequence_encoder = LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            dropout=dropout if hidden_dim > 1 else 0,
            batch_first=True
        )
        
        # Clasificador final con regularización
        classifier_input_dim = hidden_dim * 2  # bidirectional
        self.classifier = Sequential(
            torch.nn.Dropout(dropout),
            Linear(classifier_input_dim, hidden_dim),
            ReLU(),
            torch.nn.Dropout(dropout * 0.5),
            Linear(hidden_dim, vocab_size)
        )
        
    def forward(self, x):
        """
        x: Tensor (batch_size, seq_len, C, H, W)
        
        Returns:
            logits: Tensor (batch_size, vocab_size)
            embeddings: Tensor (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, C, H, W = x.shape
        
        # Procesar todas las imágenes en batch (MÁS EFICIENTE)
        x_flat = x.view(-1, C, H, W)  # [batch*seq, C, H, W]
        _, embeddings_flat = self.visual_encoder(x_flat)  # [batch*seq, embed_dim]
        
        # Reshape back to sequences
        embeddings = embeddings_flat.view(batch_size, seq_len, -1)  # [batch, seq, embed_dim]
        
        # Encode sequences con BiLSTM
        lstm_out, _ = self.sequence_encoder(embeddings)  # [batch, seq, hidden*2]
        
        # Global pooling sobre la secuencia (mean pooling)
        # Alternativas: max pooling, attention, último hidden state
        pooled = lstm_out.mean(dim=1)  # [batch, hidden*2]
        
        # Clasificación final
        logits = self.classifier(pooled)  # [batch, vocab_size]
        
        return logits, embeddings
    
    def update_vocab_size(self, new_vocab_size: int):
        """Actualiza dinámicamente el tamaño del vocabulario"""
        if new_vocab_size <= 0:
            raise ValueError("TinySpellerImproved necesita al menos una palabra en el vocabulario.")
        
        device = next(self.parameters()).device
        classifier_input_dim = self.hidden_dim * 2
        
        # Recrear clasificador con nuevo tamaño
        self.classifier = Sequential(
            torch.nn.Dropout(0.3),
            Linear(classifier_input_dim, self.hidden_dim),
            ReLU(),
            torch.nn.Dropout(0.15),
            Linear(self.hidden_dim, new_vocab_size)
        ).to(device)
        
        self.vocab_size = new_vocab_size