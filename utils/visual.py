"""
utils/visual.py — Utilidades para la generación y gestión del dataset visual.
"""

import os
import io
import base64
import string
import numpy as np
import random
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

# Definir fuentes del sistema disponibles y robustas
SYSTEM_FONTS = {
    "DejaVu Sans": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "DejaVu Serif": "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "Noto Sans Mono": "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
    "Liberation Sans": "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "Ubuntu": "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf"
}

def get_font_path(font_name: str) -> str:
    """Obtiene el path real de la fuente o un fallback seguro."""
    if font_name in SYSTEM_FONTS:
        path = Path(SYSTEM_FONTS[font_name])
        if path.exists(): return str(path)
    
    fallback = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if fallback.exists(): return str(fallback)
    
    return "arial.ttf"

# get_language_letters se ha movido a utils/graphemes.py para evitar duplicidad.

def generate_letter_image(
    letter: str, 
    font_size: int = 32, 
    rotation: float = 0, 
    noise_level: float = 0.0, 
    font_name: str = "DejaVu Sans"
) -> Image.Image | None:
    """Genera una imagen de una letra con parámetros específicos."""
    try:
        img_size = (64, 64)
        img = Image.new('L', img_size, color=255)  # Fondo blanco
        draw = ImageDraw.Draw(img)
        
        try:
            font_path = get_font_path(font_name)
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()
        
        text = letter.upper()
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Auto-escalado para digrafos o letras anchas
        if text_width > img_size[0] - 4:
            scale_factor = (img_size[0] - 4) / text_width
            new_font_size = int(font_size * scale_factor)
            try:
                font = ImageFont.truetype(font_path, new_font_size)
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except:
                pass
        
        x = (img_size[0] - text_width) // 2
        y = (img_size[1] - text_height) // 2
        
        draw.text((x, y), text, fill=0, font=font)
        
        if rotation != 0:
            img = img.rotate(rotation, fillcolor=255)
        
        if noise_level > 0:
            img_array = np.array(img)
            noise = np.random.normal(0, noise_level * 255, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)
        
        return img
    except Exception:
        return None

def save_image_to_file(
    image: Image.Image, 
    letter: str, 
    params: dict, 
    dataset_dir: str = "data/visual",
    base_path_root: Path | None = None
) -> dict | None:
    """Guarda imagen como archivo JPEG y retorna metadatos."""
    try:
        root = base_path_root if base_path_root else Path(__file__).parent.parent
        letter_dir = root / dataset_dir / letter.lower()
        letter_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        font_clean = params.get('font', 'dejavu').replace(' ', '_').replace('.ttf', '').lower()
        filename = f"{letter.lower()}_{font_clean}_fs{params.get('font_size', 32)}_r{params.get('rotation', 0):.1f}_n{params.get('noise_level', 0):.3f}_{timestamp}.jpg"
        
        image_path = letter_dir / filename
        
        if image.mode != 'RGB':
            bg = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'L':
                bg.paste(image)
            else:
                image = image.convert('RGBA')
                bg.paste(image, mask=image.split()[-1])
            image = bg
        
        image.save(image_path, format='JPEG', quality=85, optimize=True)
        
        return {
            'file_path': str(image_path.relative_to(root)),
            'filename': filename,
            'letter': letter.upper(),
            'params': params.copy(),
            'created': datetime.now().isoformat(),
            'size': list(image.size),
            'format': 'JPEG'
        }
    except Exception:
        return None
