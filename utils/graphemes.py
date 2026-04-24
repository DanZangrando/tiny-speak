"""
utils/graphemes.py — Utilidades de tokenización, fonemas y vocabulario.

Funciones:
    tokenize_graphemes        Tokeniza una palabra en grafemas (greedy matching).
    get_language_letters      Alfabeto de un idioma.
    get_phoneme_inventory     Carga el inventario de fonemas desde JSON.
    get_phonemes_from_word    Convierte una palabra en lista de fonemas.
    get_default_words         Vocabulario por defecto del sistema.
"""

from __future__ import annotations

import os
import string
from pathlib import Path
from typing import Sequence


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

LETTERS = list(string.ascii_lowercase)


# ---------------------------------------------------------------------------
# Tokenización de grafemas
# ---------------------------------------------------------------------------

def tokenize_graphemes(word: str, available_graphemes: Sequence[str]) -> list[str]:
    """Tokeniza una palabra en grafemas priorizando los más largos (greedy matching).

    Ejemplo:
        tokenize_graphemes("chancho", ["ch", "a", "n", "o"]) → ["ch", "a", "n", "ch", "o"]
    """
    if not available_graphemes:
        return list(word)

    sorted_graphemes = sorted(available_graphemes, key=len, reverse=True)

    tokens: list[str] = []
    i = 0
    while i < len(word):
        match_found = False
        for grapheme in sorted_graphemes:
            if word.startswith(grapheme, i):
                tokens.append(grapheme)
                i += len(grapheme)
                match_found = True
                break
        if not match_found:
            tokens.append(word[i])
            i += 1

    return tokens


# ---------------------------------------------------------------------------
# Alfabetos por idioma
# ---------------------------------------------------------------------------

def get_language_letters(language: str = "es") -> list[str]:
    """Retorna el alfabeto para el idioma dado."""
    alphabets: dict[str, list[str]] = {
        "es": list("abcdefghijklmnñopqrstuvwxyz") + ["ch", "ll", "rr"],
        "en": list("abcdefghijklmnopqrstuvwxyz") + ["sh", "ch", "th", "ph", "wh", "ng", "igh", "tch", "ea", "ee", "oo", "ai", "ay", "oi", "oy"],
        "fr": list("abcdefghijklmnopqrstuvwxyzàáâäèéêëìíîïòóôöùúûü") + ["ch", "gn", "ou", "oi", "an", "in", "on", "un", "eau", "au", "eu", "ai", "ei", "ain", "ein", "oin", "ieu", "ill"],
        "de": list("abcdefghijklmnopqrstuvwxyzäöüß"),
    }
    return alphabets.get(language, alphabets["es"])


# ---------------------------------------------------------------------------
# Inventario de fonemas
# ---------------------------------------------------------------------------

def get_phoneme_inventory(language: str = "es") -> list[str]:
    """Carga el inventario de fonemas para un idioma desde ``data/fonemas/phonemes.json``."""
    import json

    base_path = Path(__file__).parent.parent  # repo root
    json_path = base_path / "data" / "fonemas" / "phonemes.json"

    if not json_path.exists():
        json_path = Path("data/fonemas/phonemes.json")

    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get(language, [])
    except Exception as exc:
        print(f"Error cargando inventario de fonemas: {exc}")
        return []


def get_phonemes_from_word(word: str, language: str = "es") -> list[str]:
    """Convierte una palabra en lista de fonemas (greedy matching sobre el inventario).

    Prioriza digrafos (ej. 'ch', 'll') sobre letras individuales.
    """
    inventory = get_phoneme_inventory(language)
    inventory = sorted(inventory, key=len, reverse=True)  # digrafos primero

    phonemes: list[str] = []
    i = 0
    word = word.lower()
    while i < len(word):
        match = False
        for p in inventory:
            if word[i:].startswith(p):
                phonemes.append(p)
                i += len(p)
                match = True
                break
        if not match:
            i += 1  # saltar carácter desconocido

    return phonemes


# ---------------------------------------------------------------------------
# Vocabulario por defecto
# ---------------------------------------------------------------------------

def get_default_words() -> list[str]:
    """Retorna el vocabulario por defecto — busca en data/tiny-kalulu-200/val primero."""
    try:
        data_path = Path(__file__).parent.parent / "data"
        kalulu_path = data_path / "tiny-kalulu-200" / "val"

        if kalulu_path.exists():
            words = [
                d
                for d in sorted(os.listdir(kalulu_path))
                if not d.startswith(".") and os.path.isdir(kalulu_path / d)
            ]
            if words:
                return words
    except Exception:
        pass

    # Fallback: lista estática
    return [
        "agua", "amor", "azul", "bailar", "barco", "blanco", "bosque", "cama", "campo",
        "cantar", "casa", "cielo", "color", "correr", "dormir", "escuela", "estrella",
        "familia", "feliz", "flor", "fuego", "grande", "hermano", "historia", "hombre",
        "jardín", "juego", "leche", "libro", "lluvia", "lugar", "luna", "madre", "mar",
        "mesa", "montaña", "música", "niño", "noche", "número", "ojo", "papel", "palabra",
        "parecer", "parte", "pequeño", "perro", "persona", "piedra", "puerta", "río",
        "rojo", "señor", "sol", "tiempo", "tierra", "trabajar", "verde", "vida", "viento",
    ]
