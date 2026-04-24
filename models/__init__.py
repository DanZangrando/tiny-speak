"""
models/ — Arquitecturas neurales de TinyLearner.

Re-exports públicos para mantener compatibilidad con imports existentes:

    from models import PhonologicalPathway, VisualPathway, TinyReader, ...

Submódulos:
    tiny_ears    → TinyEars, PositionalEncoding
    tiny_eyes    → TinyEyes
    tiny_speller → TinySpeller
    tiny_reader  → TinyReader, TinyReaderP2W
"""

from models.tiny_ears import TinyEars, PositionalEncoding
from models.tiny_eyes import TinyEyes
from models.tiny_speller import TinySpeller, TinyReaderG2P
from models.tiny_reader import TinyReader, TinyReaderP2W

__all__ = [
    "TinyEars",
    "PositionalEncoding",
    "TinyEyes",
    "TinySpeller",
    "TinyReader",
    "TinyReaderG2P",
    "TinyReaderP2W",
]
