import streamlit as st
import json
import os
import base64
import io
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import string
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from components.modern_sidebar import display_modern_sidebar

st.set_page_config(
    page_title="📊 Visual Dataset Manager - TinySpeak",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Aplicar estilos CSS modernos
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 2rem;
}

.modern-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
    border: none;
    text-align: center;
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    border-left: 4px solid #00d4ff;
    margin-bottom: 0.5rem;
}

.status-success {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: bold;
}

.status-warning {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: bold;
}

.sidebar-header {
    background: linear-gradient(90deg, #667eea, #764ba2);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Definir fuentes del sistema disponibles y robustas
SYSTEM_FONTS = {
    "DejaVu Sans": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "DejaVu Serif": "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "Noto Sans Mono": "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
    "Liberation Sans": "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", # Common fallback
    "Ubuntu": "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf"
}

def get_font_path(font_name):
    """Obtiene el path real de la fuente o un fallback seguro"""
    # 1. Buscar en mapa de fuentes del sistema
    if font_name in SYSTEM_FONTS:
        path = Path(SYSTEM_FONTS[font_name])
        if path.exists(): return str(path)
    
    # 2. Fallback a DejaVu Sans (muy robusta para unicode)
    fallback = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if fallback.exists(): return str(fallback)
    
    # 3. Último recurso
    return "arial.ttf" 

def load_master_config():
    """Cargar configuración desde master_dataset_config.json PRESERVANDO todo el contenido existente"""
    # Usar ruta absoluta basada en la ubicación del archivo actual
    current_dir = Path(__file__).parent.parent
    config_file = current_dir / "master_dataset_config.json"
    
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Asegurar que existe la sección visual SIN sobrescribir
            if 'visual_dataset' not in config:
                config['visual_dataset'] = {}
            
            # Solo agregar campos que no existen, preservando los existentes
            visual_defaults = {
                "name": "Visual Dataset TinySpeak",
                "letters": list(string.ascii_lowercase),
                "vocabulary": config.get('diccionario_seleccionado', config.get('master_config_reference', 'casa_familia')),
                "fonts": ["arial.ttf", "times.ttf", "calibri.ttf"],
                "font_sizes": [20, 24, 28, 32, 36, 40],
                "rotation_range": 15,
                "noise_levels": [0.0, 0.1, 0.2, 0.3],
                "image_size": [64, 64],
                "generated_images": {},
                "version": "2.0"
            }
            
            config_changed = False
            for key, default_value in visual_defaults.items():
                if key not in config['visual_dataset']:
                    config['visual_dataset'][key] = default_value
                    config_changed = True
            
            # Solo guardar si hubo cambios
            if config_changed:
                st.info("🔧 Agregando configuración visual faltante (preservando datos existentes)")
                save_master_config(config)
            
            return config
        
        except Exception as e:
            st.error(f"Error cargando configuración maestra: {e}")
            return create_default_master_config()
    else:
        return create_default_master_config()

def create_default_master_config():
    """Crea configuración maestra por defecto SOLO si no existe archivo"""
    st.warning("⚠️ No se encontró master_dataset_config.json. Creando configuración básica.")
    st.info("💡 Recomendación: Configura primero el diccionario y parámetros desde las otras páginas.")
    
    config = {
        "master_config_reference": "casa_familia",
        "diccionario_seleccionado": "casa_familia",
        "visual_dataset": {
            "name": "Visual Dataset TinySpeak", 
            "letters": list(string.ascii_lowercase),
            "vocabulary": "casa_familia",
            "fonts": ["arial.ttf", "times.ttf", "calibri.ttf"],
            "font_sizes": [20, 24, 28, 32, 36, 40],
            "rotation_range": 15,
            "noise_levels": [0.0, 0.1, 0.2, 0.3],
            "image_size": [64, 64],
            "generated_images": {},
            "version": "2.0",
            "created": datetime.now().isoformat()
        }
    }
    
    save_master_config(config)
    return config

def clean_config_for_json(obj):
    """Limpia recursivamente el config para que sea serializable a JSON"""
    if isinstance(obj, dict):
        cleaned = {}
        for key, value in obj.items():
            # Evitar guardar objetos Image y otros no serializables
            if key in ['image_objects', 'preview_images']:
                continue
            cleaned[key] = clean_config_for_json(value)
        return cleaned
    elif isinstance(obj, list):
        return [clean_config_for_json(item) for item in obj]
    elif hasattr(obj, '__module__') and 'PIL' in str(type(obj)):
        # Si es un objeto PIL Image, no lo incluimos
        return None
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    else:
        # Para cualquier otro tipo, intentar convertir a string
        try:
            return str(obj) if obj is not None else None
        except:
            return None

def save_master_config(config):
    """Guardar configuración a master_dataset_config.json"""
    try:
        # Limpiar config antes de guardar
        clean_config = clean_config_for_json(config)
        
        # Usar ruta absoluta basada en la ubicación del archivo actual
        current_dir = Path(__file__).parent.parent
        config_file = current_dir / "master_dataset_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(clean_config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Error guardando configuración: {e}")
        return False

def get_unique_letters_from_dataset():
    """Detecta letras únicas del experimento o dataset actual (Soporte Multi-idioma)"""
    config = load_master_config()
    unique_letters = set()
    
    # 1. Estrategia Principal: Configuración de Experimento (Multi-idioma)
    exp_config = config.get('experiment_config', {})
    if exp_config and 'languages' in exp_config:
        langs = exp_config['languages']
        
        # Usar el alfabeto completo de cada idioma configurado
        # Esto asegura que se incluyan letras como 'w' (EN) o 'ç' (FR) 
        # aunque no estén presentes en las palabras del diccionario actual.
        for lang in langs:
            lang_letters = get_language_letters(lang)
            unique_letters.update(lang_letters)
            
        if unique_letters:
            return sorted(list(unique_letters))

    # 2. Estrategia Secundaria: Diccionario Seleccionado (Legacy/Single Lang)
    diccionario_info = config.get('diccionario_seleccionado', {})
    if diccionario_info and 'palabras' in diccionario_info:
        palabras = diccionario_info['palabras']
        for palabra in palabras:
            for char in palabra.lower():
                if char.isalpha():
                    unique_letters.add(char)
        if unique_letters:
            return sorted(list(unique_letters))

    # 3. Fallback: Muestras Generadas (con aplanado robusto para evitar AttributeError)
    audio_samples = config.get('generated_samples', {})
    if audio_samples:
        for key, value in audio_samples.items():
            if isinstance(value, list):
                # Estructura plana: key es la palabra
                for char in key.lower():
                    if char.isalpha(): unique_letters.add(char)
            elif isinstance(value, dict):
                # Estructura anidada: key es idioma, value es dict de palabras
                for word in value.keys():
                    for char in word.lower():
                        if char.isalpha(): unique_letters.add(char)
    
    return sorted(list(unique_letters))

def get_language_letters(language='es'):
    """Obtiene todas las letras de un idioma específico"""
    language_alphabets = {
        'es': list('abcdefghijklmnñopqrstuvwxyz'),  # Español
        'en': list('abcdefghijklmnopqrstuvwxyz'),   # Inglés
        'fr': list('abcdefghijklmnopqrstuvwxyzàáâäèéêëìíîïòóôöùúûü'),  # Francés básico
        'de': list('abcdefghijklmnopqrstuvwxyzäöüß'),  # Alemán básico
    }
    return language_alphabets.get(language, language_alphabets['es'])

def create_language_based_dataset_name(language, method="manual"):
    """Crea nombre de dataset basado en idioma y método"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"dataset_visual_{language}_{method}_{timestamp}"

def update_visual_vocabulary():
    """Actualiza el vocabulario visual para sincronizar con el master config"""
    config = load_master_config()
    
    # Sincronizar vocabulario visual con master_config_reference
    master_vocab = config.get('master_config_reference', 'casa_familia')
    config['visual_dataset']['vocabulary'] = master_vocab
    
    # Obtener letras únicas del vocabulario master
    if master_vocab in config.get('generated_samples', {}):
        vocab_letters = set()
        for sample in config['generated_samples'][master_vocab]:
            for char in sample.get('word', '').lower():
                if char.isalpha():
                    vocab_letters.add(char)
        
        config['visual_dataset']['letters'] = sorted(list(vocab_letters))
    
    save_master_config(config)
    return config

def generate_letter_image(letter, font_size=32, rotation=0, noise_level=0.0, font_name="arial.ttf"):
    """Genera una imagen de una letra con parámetros específicos"""
    try:
        # Crear imagen base
        img_size = (64, 64)
        img = Image.new('L', img_size, color=255)  # Fondo blanco
        draw = ImageDraw.Draw(img)
        
        # Intentar cargar font (usar font por defecto si falla)
        try:
            # Resolver path real de la fuente
            font_path = get_font_path(font_name)
            font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            # st.warning(f"Font error {font_name}: {e}")
            font = ImageFont.load_default()
        
        # Calcular posición centrada
        bbox = draw.textbbox((0, 0), letter.upper(), font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (img_size[0] - text_width) // 2
        y = (img_size[1] - text_height) // 2
        
        # Dibujar letra
        draw.text((x, y), letter.upper(), fill=0, font=font)  # Negro sobre blanco
        
        # Aplicar rotación si se especifica
        if rotation != 0:
            img = img.rotate(rotation, fillcolor=255)
        
        # Aplicar ruido si se especifica
        if noise_level > 0:
            img_array = np.array(img)
            noise = np.random.normal(0, noise_level * 255, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)
        
        return img
    
    except Exception as e:
        st.error(f"Error generando imagen para '{letter}': {e}")
        return None

def save_image_to_file(image, letter, params, dataset_dir="data/visual"):
    """Guarda imagen como archivo PNG y retorna metadatos"""
    try:
        # Crear estructura de carpetas: data/visual/letra/
        current_dir = Path(__file__).parent.parent
        letter_dir = current_dir / dataset_dir / letter.lower()
        letter_dir.mkdir(parents=True, exist_ok=True)
        
        # Generar nombre único basado en parámetros y timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        font_name = params.get('font', 'arial').replace(' ', '_').replace('.ttf', '').lower()
        filename = f"{letter.lower()}_{font_name}_fs{params.get('font_size', 32)}_r{params.get('rotation', 0):.1f}_n{params.get('noise_level', 0):.3f}_{timestamp}.jpg"
        
        image_path = letter_dir / filename
        
        # Convertir a RGB si es necesario (JPEG no soporta transparencia)
        if image.mode in ('RGBA', 'LA', 'P'):
            # Crear fondo blanco
            bg = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            bg.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            image = bg
        
        # Guardar imagen como JPEG optimizado para AI
        image.save(image_path, format='JPEG', quality=85, optimize=True)
        
        # Retornar metadatos
        return {
            'file_path': str(image_path.relative_to(current_dir)),  # Ruta relativa para portabilidad
            'filename': filename,
            'letter': letter.upper(),
            'params': params.copy(),  # Hacer copia para evitar referencias
            'created': datetime.now().isoformat(),
            'size': list(image.size),  # Convertir tuple a list para JSON
            'format': 'JPEG'
        }
        
    except Exception as e:
        st.error(f"Error guardando imagen: {e}")
        return None

def load_image_from_metadata(metadata, base_dir=None):
    """Carga imagen desde metadatos"""
    try:
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        
        image_path = base_dir / metadata['file_path']
        if image_path.exists():
            return Image.open(image_path)
        else:
            st.warning(f"Archivo de imagen no encontrado: {image_path}")
            return None
    except Exception as e:
        st.error(f"Error cargando imagen: {e}")
        return None

def create_pytorch_dataset_structure(config, output_base_dir="visual_dataset"):
    """Crear estructura de directorios compatible con PyTorch Dataset"""
    base_path = Path(output_base_dir)
    base_path.mkdir(exist_ok=True)
    
    # Crear directorios por letra
    classes = []
    for letter in config['alphabet'].keys():
        letter_dir = base_path / letter
        letter_dir.mkdir(exist_ok=True)
        classes.append(letter)
    
    # Crear archivo de clases para PyTorch
    classes_file = base_path / "classes.txt"
    with open(classes_file, 'w', encoding='utf-8') as f:
        for cls in sorted(classes):
            f.write(f"{cls}\n")
    
    # Crear directorios de splits (train/val/test) si no existen
    for split in ['train', 'val', 'test']:
        split_dir = base_path / split
        split_dir.mkdir(exist_ok=True)
        for letter in classes:
            (split_dir / letter).mkdir(exist_ok=True)
    
    return base_path

def migrate_base64_to_files(config, output_dir="visual_dataset"):
    """Migrar imágenes existentes de base64 a archivos de forma segura"""
    migrated_count = 0
    errors = []
    base_path = Path(output_dir)
    
    # Crear copia de seguridad del config antes de la migración
    backup_path = Path("master_dataset_config_pre_migration.json")
    try:
        import shutil
        shutil.copy2("master_dataset_config.json", backup_path)
        st.info(f"📋 Backup creado en: {backup_path}")
    except Exception as e:
        st.warning(f"⚠️ No se pudo crear backup: {str(e)}")
    
    # Buscar imágenes en diferentes secciones del config
    sections_to_check = [
        ('alphabet', 'variaciones'),
        ('generated_samples', None),
        ('visual_dataset.generated_images', None)
    ]
    
    for section_path, sub_key in sections_to_check:
        try:
            # Navegar a la sección del config
            current_data = config
            for key in section_path.split('.'):
                if key in current_data:
                    current_data = current_data[key]
                else:
                    current_data = None
                    break
            
            if not current_data:
                continue
                
            # Procesar datos según el tipo
            if isinstance(current_data, dict):
                for letter, letter_data in current_data.items():
                    if sub_key and sub_key in letter_data:
                        # Caso: alphabet.variaciones
                        images_list = letter_data[sub_key]
                    elif isinstance(letter_data, list):
                        # Caso: generated_samples o visual_dataset.generated_images
                        images_list = letter_data
                    else:
                        continue
                    
                    if not images_list:
                        continue
                        
                    # Crear directorio para la letra
                    letter_dir = base_path / letter
                    letter_dir.mkdir(exist_ok=True)
                    
                    # Procesar cada imagen
                    for idx, img_data in enumerate(images_list):
                        if not isinstance(img_data, dict):
                            continue
                            
                        if 'image' in img_data and img_data['image'] and 'file_path' not in img_data:
                            try:
                                # Validar que no sea una cadena vacía
                                if not img_data['image'].strip():
                                    continue
                                    
                                # Decodificar imagen base64
                                img_bytes = base64.b64decode(img_data['image'])
                                img = Image.open(io.BytesIO(img_bytes))
                                
                                # Generar nombre de archivo único
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                                filename = f"{letter}_{timestamp}_{idx:03d}.jpg"
                                file_path = letter_dir / filename
                                
                                # Convertir a RGB si es necesario (JPEG no soporta transparencia)
                                if img.mode in ('RGBA', 'LA', 'P'):
                                    bg = Image.new('RGB', img.size, (255, 255, 255))
                                    if img.mode == 'P':
                                        img = img.convert('RGBA')
                                    bg.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                                    img = bg
                                
                                # Guardar imagen como JPEG optimizado para AI
                                img.save(file_path, format='JPEG', quality=85, optimize=True)
                                
                                # Actualizar metadatos de forma segura
                                img_data['file_path'] = str(file_path)
                                img_data['created_at'] = timestamp
                                img_data['migrated_from'] = 'base64'
                                
                                # Verificar que el archivo se guardó correctamente
                                if file_path.exists() and file_path.stat().st_size > 0:
                                    # Solo eliminar base64 si el archivo se guardó exitosamente
                                    del img_data['image']
                                    migrated_count += 1
                                else:
                                    errors.append(f"Error: archivo {file_path} no se guardó correctamente")
                                    
                            except Exception as e:
                                error_msg = f"Error migrando {section_path}.{letter}[{idx}]: {str(e)}"
                                errors.append(error_msg)
                                st.error(error_msg)
                                
        except Exception as e:
            error_msg = f"Error procesando sección {section_path}: {str(e)}"
            errors.append(error_msg)
            st.error(error_msg)
    
    # Mostrar resumen
    if errors:
        st.warning(f"⚠️ {len(errors)} errores durante la migración")
        with st.expander("Ver errores"):
            for error in errors:
                st.text(error)
    
    return migrated_count

def main():
    # Mostrar sidebar moderna
    display_modern_sidebar("visual_dataset")
    
    # Header moderno
    st.markdown('<h1 class="main-header">🖼️ Visual Dataset Manager</h1>', unsafe_allow_html=True)
    
    # Cargar configuración
    config = load_master_config()
    visual_config = config['visual_dataset']
    
    # === SECCIÓN 1: DETECCIÓN Y CONFIGURACIÓN DE LETRAS ===
    st.markdown("## 🔍 Análisis del Dataset")
    
    # Mostrar información de la fuente de datos
    exp_config = config.get('experiment_config', {})
    diccionario_info = config.get('diccionario_seleccionado', {})
    
    if exp_config and 'languages' in exp_config:
        langs = exp_config.get('languages', [])
        base_dict = exp_config.get('base_dictionary', 'Desconocido')
        st.info(f"🧪 **Experimento Activo**: {base_dict} - Idiomas: {', '.join([l.upper() for l in langs])}")
    elif diccionario_info:
        st.info(f"📚 **Diccionario activo**: {diccionario_info.get('nombre', 'Sin nombre')} - "
                f"{len(diccionario_info.get('palabras', []))} palabras")
    else:
        st.warning("⚠️ No hay configuración de experimento ni diccionario seleccionado.")
    
    # Detectar letras únicas del dataset actual
    dataset_letters = get_unique_letters_from_dataset()
    
    # Botón de recarga por si hay problemas de cache
    if st.button("🔄 Recargar Análisis", help="Actualiza el análisis de letras del diccionario"):
        st.rerun()
    
    analysis_col1, analysis_col2 = st.columns(2)
    
    with analysis_col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 Letras Únicas en Dataset</h4>
            <p><strong>{len(dataset_letters)} letras detectadas</strong></p>
            <p style="font-size: 1rem; margin-top: 0.5rem; font-family: monospace; letter-spacing: 2px;">
                {' '.join(dataset_letters) if dataset_letters else 'Ninguna letra detectada'}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with analysis_col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>�️ Estado del Dataset Visual</h4>
            <p><strong>{len(visual_config.get('generated_images', {}))} letras generadas</strong></p>
            <p style="font-size: 0.8rem; margin-top: 0.5rem;">
                {sum(len(imgs) for imgs in visual_config.get('generated_images', {}).values())} imágenes totales
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # === SECCIÓN 2: SELECCIÓN DE ESTRATEGIA DE GENERACIÓN ===
    st.markdown("## 🎯 Configuración de Generación")
    
    strategy_col1, strategy_col2 = st.columns(2)
    
    with strategy_col1:
        generation_strategy = st.radio(
            "🎯 **Estrategia de Generación**",
            options=["dataset_letters", "language_complete"],
            format_func=lambda x: {
                "dataset_letters": f"🔤 Solo letras del dataset ({len(dataset_letters)} letras)",
                "language_complete": "🌐 Alfabeto completo del idioma"
            }[x],
            help="Elige si generar solo las letras presentes en tu dataset o todo el alfabeto del idioma"
        )
    
    with strategy_col2:
        if generation_strategy == "language_complete":
            target_language = st.selectbox(
                "🌍 **Idioma Objetivo**",
                options=["es", "en", "fr", "de"],
                format_func=lambda x: {
                    "es": "🇪🇸 Español (27 letras)",
                    "en": "🇺🇸 Inglés (26 letras)", 
                    "fr": "🇫🇷 Francés (42 letras)",
                    "de": "🇩🇪 Alemán (30 letras)"
                }[x],
                help="Selecciona el idioma para generar el alfabeto completo"
            )
            target_letters = get_language_letters(target_language)
        else:
            target_language = "dataset"
            target_letters = dataset_letters
    
    # Mostrar letras objetivo
    st.markdown(f"""
    <div style="background: rgba(103, 58, 183, 0.1); padding: 1rem; border-radius: 8px; margin: 1rem 0;">
        <h4>🎯 Letras Objetivo: {len(target_letters)} letras</h4>
        <p style="font-family: monospace; font-size: 1.2rem; letter-spacing: 2px;">
            {' '.join(target_letters)}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # === SECCIÓN 3: CONFIGURACIÓN DE PARÁMETROS ===
    st.markdown("## ⚙️ Parámetros de Generación")
    
    params_col1, params_col2, params_col3 = st.columns(3)
    
    with params_col1:
        st.markdown("**🎨 Parámetros Visuales**")
        variations_per_letter = st.number_input(
            "Variaciones por letra",
            min_value=1,
            max_value=50,
            value=visual_config.get('image_params', {}).get('variations_per_letter', 10),
            help="Número de imágenes diferentes por cada letra"
        )
        
        font_size_min = st.number_input("Tamaño mínimo de fuente", min_value=10, max_value=30, value=16)
        font_size_max = st.number_input("Tamaño máximo de fuente", min_value=31, max_value=60, value=48)
    
    with params_col2:
        st.markdown("**🔄 Transformaciones**")
        rotation_min = st.number_input("Rotación mínima (°)", min_value=-45, max_value=-1, value=-30)
        rotation_max = st.number_input("Rotación máxima (°)", min_value=1, max_value=45, value=30)
        
        noise_min = st.number_input("Ruido mínimo", min_value=0.0, max_value=0.1, value=0.0, step=0.01)
        noise_max = st.number_input("Ruido máximo", min_value=0.1, max_value=0.3, value=0.2, step=0.01)
    
    with params_col3:
        st.markdown("**📝 Fuentes**")
        available_fonts = list(SYSTEM_FONTS.keys())
        selected_fonts = st.multiselect(
            "Fuentes a usar",
            available_fonts,
            default=visual_config.get('image_params', {}).get('fonts', ["DejaVu Sans"]),
            help="Selecciona las fuentes para generar variedad"
        )
    
    # === SECCIÓN 4: GESTIÓN DE DATASET ===
    st.markdown("---")
    st.markdown("## 📦 Gestión del Dataset")
    
    dataset_col1, dataset_col2 = st.columns(2)
    
    with dataset_col1:
        # Generar nombre automático del dataset
        dataset_name = create_language_based_dataset_name(target_language, "visual_generation")
        
        st.text_input(
            "📁 **Nombre del Dataset**",
            value=dataset_name,
            help="Nombre automático basado en idioma y método",
            disabled=True
        )
        
        clear_existing = st.checkbox(
            "🗑️ **Borrar dataset preexistente**",
            value=True,
            help="Recomendado: Limpia dataset anterior del mismo idioma"
        )
    
    with dataset_col2:
        st.markdown("**🎯 Selección de Letras**")
        if len(target_letters) <= 10:
            letters_to_generate = st.multiselect(
                "Letras a generar",
                options=target_letters,
                default=target_letters,
                help="Selecciona las letras específicas a generar"
            )
        else:
            generate_all = st.checkbox("Generar todas las letras", value=True)
            if not generate_all:
                letters_to_generate = st.multiselect(
                    "Letras específicas",
                    options=target_letters,
                    default=target_letters[:5],
                    help="Demasiadas letras, selecciona las específicas"
                )
            else:
                letters_to_generate = target_letters
    
    # === SECCIÓN 5: GENERACIÓN Y VISUALIZACIÓN ===
    st.markdown("---")
    generation_tab1, generation_tab2 = st.tabs(["🚀 Generar Dataset", "👁️ Visualizar"])
    
    with generation_tab1:
        st.markdown("### 🚀 Generación del Dataset Visual")
        
        # Resumen de configuración
        st.markdown(f"""
        <div style="background: rgba(76, 175, 80, 0.1); padding: 1rem; border-radius: 8px;">
            <h4>📋 Resumen de Configuración</h4>
            <ul>
                <li><strong>Estrategia:</strong> {'Alfabeto completo' if generation_strategy == 'language_complete' else 'Solo letras del dataset'}</li>
                <li><strong>Idioma:</strong> {target_language.upper()}</li>
                <li><strong>Letras a generar:</strong> {len(letters_to_generate)} de {len(target_letters)}</li>
                <li><strong>Variaciones por letra:</strong> {variations_per_letter}</li>
                <li><strong>Total de imágenes:</strong> {len(letters_to_generate) * variations_per_letter}</li>
                <li><strong>Limpiar dataset previo:</strong> {'Sí' if clear_existing else 'No'}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Botón de generación
        if st.button("🚀 **Generar Dataset Visual**", type="primary", width='stretch'):
            if not letters_to_generate:
                st.error("⚠️ Selecciona al menos una letra para generar")
            elif not selected_fonts:
                st.error("⚠️ Selecciona al menos una fuente")
            else:
                # Aquí iría la lógica de generación
                generate_visual_dataset_new(
                    letters_to_generate, variations_per_letter,
                    font_size_min, font_size_max, rotation_min, rotation_max,
                    noise_min, noise_max, selected_fonts, target_language, clear_existing
                )
    
    with generation_tab2:
        display_visual_dataset_preview(visual_config)

def generate_visual_dataset_new(letters, variations_per_letter, font_size_min, font_size_max, 
                               rotation_min, rotation_max, noise_min, noise_max, fonts, 
                               language, clear_existing):
    """Nueva función de generación mejorada del dataset visual"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Cargar configuración
    master_config = load_master_config()
    
    # Limpiar dataset previo si se solicita
    if clear_existing:
        if 'visual_dataset' not in master_config:
            master_config['visual_dataset'] = {}
        master_config['visual_dataset']['generated_images'] = {}
        status_text.text("🗑️ Limpiando dataset previo...")
    
    # Asegurar que existe la estructura
    if 'generated_images' not in master_config['visual_dataset']:
        master_config['visual_dataset']['generated_images'] = {}
    
    generated_images = master_config['visual_dataset']['generated_images']
    total_operations = len(letters) * variations_per_letter
    current_operation = 0
    
    try:
        for letter_idx, letter in enumerate(letters):
            status_text.text(f"🖼️ Generando imágenes para la letra '{letter.upper()}'...")
            
            # Inicializar lista para esta letra
            if letter not in generated_images:
                generated_images[letter] = []
            
            letter_images = []
            
            for var_idx in range(variations_per_letter):
                # Parámetros aleatorios para cada variación
                font = random.choice(fonts)
                font_size = random.randint(font_size_min, font_size_max)
                rotation = random.uniform(rotation_min, rotation_max)
                noise_level = random.uniform(noise_min, noise_max)
                
                # Generar imagen usando la función correcta
                img = generate_letter_image(
                    letter, font_size, rotation, noise_level, font
                )
                
                if img:
                    # Preparar parámetros para guardar
                    image_params = {
                        "font": font,
                        "font_size": font_size,
                        "rotation": rotation,
                        "noise_level": noise_level,
                        "size": [64, 64]
                    }
                    
                    # Guardar imagen como archivo y obtener metadatos
                    image_metadata = save_image_to_file(img, letter, image_params)
                    
                    if image_metadata:
                        # Agregar información adicional de generación
                        image_metadata.update({
                            "letter": letter.upper(),
                            "created": datetime.now().isoformat(),
                            "language": language,
                            "generation_method": "file_based_v3"  # Nuevo método
                        })
                        letter_images.append(image_metadata)
                
                current_operation += 1
                progress_bar.progress(current_operation / total_operations)
            
            # Agregar imágenes de esta letra al dataset
            generated_images[letter].extend(letter_images)
        
        # Actualizar metadatos del dataset
        master_config['visual_dataset'].update({
            'letters': letters,
            'language': language,
            'generation_params': {
                'font_size_range': [font_size_min, font_size_max],
                'rotation_range': [rotation_min, rotation_max],
                'noise_range': [noise_min, noise_max],
                'fonts': fonts,
                'variations_per_letter': variations_per_letter
            },
            'last_generated': datetime.now().isoformat(),
            'total_images': sum(len(imgs) for imgs in generated_images.values())
        })
        
        # Guardar configuración
        save_master_config(master_config)
        
        status_text.text("✅ ¡Generación completada!")
        st.success(f"🎉 ¡Dataset generado exitosamente! "
                  f"Se generaron {len(letters) * variations_per_letter} imágenes "
                  f"para {len(letters)} letras en {language.upper()}")
        
        # Mostrar estadísticas finales
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Letras generadas", len(letters))
        with col2:
            st.metric("Imágenes por letra", variations_per_letter)
        with col3:
            st.metric("Total de imágenes", len(letters) * variations_per_letter)
            
    except Exception as e:
        st.error(f"❌ Error durante la generación: {str(e)}")
    
    progress_bar.empty()
    status_text.empty()

def display_visual_dataset_preview(visual_config):
    """Muestra una vista previa del dataset visual generado"""
    
    st.markdown("### 👁️ Vista Previa del Dataset")
    
    # Cargar master config para obtener las imágenes del lugar correcto
    master_config = load_master_config()
    generated_images = master_config.get('visual_dataset', {}).get('generated_images', {})
    
    if not generated_images:
        st.info("📭 No hay imágenes generadas aún. Ve a la pestaña 'Generar Dataset' para crear el dataset.")
        return
    
    # Estadísticas rápidas
    total_images = sum(len(imgs) for imgs in generated_images.values())
    letters_count = len(generated_images)
    
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    with stats_col1:
        st.metric("🔤 Letras", letters_count)
    with stats_col2:
        st.metric("🖼️ Total Imágenes", total_images)
    with stats_col3:
        avg_per_letter = total_images // letters_count if letters_count > 0 else 0
        st.metric("📊 Promedio/Letra", avg_per_letter)
    
    # Selector de letra para previsualizar
    available_letters = list(generated_images.keys())
    selected_letter = st.selectbox(
        "Selecciona una letra para previsualizar",
        available_letters,
        help="Elige una letra para ver sus imágenes generadas"
    )
    
    if selected_letter and selected_letter in generated_images:
        images_data = generated_images[selected_letter]
        
        st.markdown(f"#### 📝 Letra '{selected_letter.upper()}' - {len(images_data)} imágenes")
        
        # Mostrar hasta 12 imágenes en una grilla
        images_to_show = images_data[:12]
        
        cols = st.columns(4)
        for idx, img_data in enumerate(images_to_show):
            col_idx = idx % 4
            
            with cols[col_idx]:
                try:
                    img = None
                    
                    # Nuevo sistema: cargar desde archivo (PRIORITARIO)
                    if 'file_path' in img_data and img_data['file_path']:
                        img = load_image_from_metadata(img_data)
                        if img:
                            st.image(img, caption=f"Var {idx + 1}", width=100)
                            params = img_data.get('params', {})
                            st.caption(f"F:{params.get('font_size', 'N/A')} "
                                     f"R:{params.get('rotation', 0):.1f}° "
                                     f"N:{params.get('noise_level', 0):.2f}")
                            st.caption(f"📁 {Path(img_data['file_path']).name}")
                        else:
                            st.error(f"❌ Archivo no encontrado: {img_data.get('file_path', 'N/A')}")
                    
                    # Sistema legacy: cargar desde base64 (SOLO SI NO HAY ARCHIVO)
                    elif 'image' in img_data and img_data['image'] and img_data['image'] != None:
                        try:
                            img_bytes = base64.b64decode(img_data['image'])
                            img = Image.open(io.BytesIO(img_bytes))
                            st.image(img, caption=f"Var {idx + 1} (legacy)", width=100)
                            params = img_data.get('params', {})
                            st.caption(f"F:{params.get('font_size', 'N/A')} "
                                     f"R:{params.get('rotation', 0):.1f}° "
                                     f"N:{params.get('noise_level', 0):.2f}")
                            st.caption("🔄 Base64 (migrar a archivos)")
                        except Exception as e:
                            st.error(f"❌ Error decodificando base64: {str(e)}")
                    
                    # Sin datos de imagen válidos
                    else:
                        st.warning("⚠️ Datos de imagen no disponibles")
                        st.caption("Generar nuevamente o migrar desde base64")
                        
                except Exception as e:
                    st.error(f"Error procesando imagen {idx + 1}: {str(e)}")
        
        if len(images_data) > 12:
            st.info(f"📝 Mostrando las primeras 12 de {len(images_data)} imágenes disponibles")

def configuration_tab(visual_config, master_config):
    """Tab de configuración de parámetros visuales"""
    st.header("🎛️ Configuración de Parámetros Visuales")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Configuración Básica")
        
        # Nombre del dataset
        new_name = st.text_input(
            "Nombre del dataset visual",
            value=visual_config.get('name', 'Visual Dataset TinySpeak'),
            key="visual_dataset_name"
        )
        
        # Letras objetivo (solo mostrar, se sincronizan con master)
        letters_display = ', '.join(visual_config.get('letters', [])[:10])
        if len(visual_config.get('letters', [])) > 10:
            letters_display += f" ... (+{len(visual_config['letters']) - 10} más)"
        
        st.text_input(
            "Letras objetivo (sincronizadas con Master Config)",
            value=letters_display,
            disabled=True,
            help="Las letras se sincronizan automáticamente con el vocabulario maestro"
        )
        
        # Tamaño de imagen
        img_width = st.number_input("Ancho de imagen (px)", min_value=32, max_value=128, value=visual_config.get('image_size', [64, 64])[0])
        img_height = st.number_input("Alto de imagen (px)", min_value=32, max_value=128, value=visual_config.get('image_size', [64, 64])[1])
        
        # Versión
        version = st.text_input("Versión del dataset", value=visual_config.get('version', '2.0'))
    
    with col2:
        st.subheader("🎨 Parámetros de Generación")
        
        # Tamaños de fuente
        font_sizes = st.multiselect(
            "Tamaños de fuente disponibles",
            options=[16, 20, 24, 28, 32, 36, 40, 44, 48],
            default=visual_config.get('font_sizes', [20, 24, 28, 32, 36, 40]),
            key="font_sizes_config"
        )
        
        # Rango de rotación
        rotation_range = st.slider(
            "Rango de rotación (±grados)",
            min_value=0,
            max_value=45,
            value=visual_config.get('rotation_range', 15),
            help="Las imágenes se rotarán aleatoriamente dentro de este rango"
        )
        
        # Niveles de ruido
        noise_levels = st.multiselect(
            "Niveles de ruido",
            options=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
            default=visual_config.get('noise_levels', [0.0, 0.1, 0.2, 0.3]),
            format_func=lambda x: f"{x:.2f}" if x > 0 else "Sin ruido",
            key="noise_levels_config"
        )
        
        # Fuentes disponibles
        available_fonts = ["arial.ttf", "times.ttf", "calibri.ttf", "courier.ttf", "georgia.ttf"]
        selected_fonts = st.multiselect(
            "Fuentes a utilizar",
            options=available_fonts,
            default=visual_config.get('fonts', ["arial.ttf", "times.ttf", "calibri.ttf"]),
            key="fonts_config"
        )
    
    # Guardar configuración
    st.markdown("---")
    if st.button("💾 Guardar Configuración Visual", type="primary"):
        # Actualizar configuración visual en master config
        master_config['visual_dataset']['name'] = new_name
        master_config['visual_dataset']['image_size'] = [img_width, img_height]
        master_config['visual_dataset']['version'] = version
        master_config['visual_dataset']['font_sizes'] = font_sizes
        master_config['visual_dataset']['rotation_range'] = rotation_range
        master_config['visual_dataset']['noise_levels'] = noise_levels
        master_config['visual_dataset']['fonts'] = selected_fonts
        
        if save_master_config(master_config):
            st.success("✅ Configuración visual guardada en master config!")
        else:
            st.error("❌ Error guardando configuración")

def generation_tab(visual_config, master_config):
    """Tab de generación de imágenes"""
    st.header("🖼️ Generación de Imágenes de Letras")
    
    # Información del vocabulario actual
    st.info(f"🎯 Generando imágenes para el vocabulario: **{visual_config.get('vocabulary', 'N/A')}**")
    
    generation_col1, generation_col2 = st.columns(2)
    
    with generation_col1:
        st.subheader("🔤 Selección de Letras")
        
        # Mostrar letras disponibles
        available_letters = visual_config.get('letters', [])
        
        if available_letters:
            # Opción de generar todas las letras
            generate_all = st.checkbox("Generar todas las letras", key="generate_all_letters")
            
            if not generate_all:
                # Selección manual de letras
                selected_letters = st.multiselect(
                    "Seleccionar letras específicas",
                    options=available_letters,
                    default=available_letters[:5] if len(available_letters) >= 5 else available_letters,
                    key="selected_letters_manual"
                )
            else:
                selected_letters = available_letters
                st.write(f"✅ Se generarán imágenes para todas las {len(available_letters)} letras")
        else:
            st.warning("⚠️ No hay letras disponibles. Sincroniza con el Master Config primero.")
            selected_letters = []
    
    with generation_col2:
        st.subheader("⚙️ Parámetros de Generación")
        
        # Cantidad de variaciones por letra
        variations_per_letter = st.number_input(
            "Variaciones por letra",
            min_value=1,
            max_value=50,
            value=10,
            help="Número de imágenes diferentes que se generarán para cada letra"
        )
        
        # Parámetros específicos de esta generación
        use_random_params = st.checkbox(
            "Usar parámetros aleatorios",
            value=True,
            help="Si se activa, cada imagen usará parámetros aleatorios dentro de los rangos configurados"
        )
        
        if not use_random_params:
            # Parámetros fijos
            fixed_font_size = st.selectbox("Tamaño de fuente fijo", visual_config.get('font_sizes', [32]))
            fixed_rotation = st.slider("Rotación fija", -45, 45, 0)
            fixed_noise = st.selectbox("Nivel de ruido fijo", visual_config.get('noise_levels', [0.0]))
            fixed_font = st.selectbox("Fuente fija", visual_config.get('fonts', ['arial.ttf']))
    
    # Botón de generación
    st.markdown("---")
    
    if selected_letters:
        if st.button("🚀 Generar Imágenes", type="primary", key="generate_images_btn"):
            generate_images_for_letters(
                selected_letters, 
                variations_per_letter, 
                visual_config, 
                master_config,
                use_random_params,
                {
                    'font_size': fixed_font_size if not use_random_params else None,
                    'rotation': fixed_rotation if not use_random_params else None,
                    'noise_level': fixed_noise if not use_random_params else None,
                    'font': fixed_font if not use_random_params else None
                }
            )
    else:
        st.warning("⚠️ Selecciona al menos una letra para generar imágenes")
    
    # Vista previa de parámetros
    if selected_letters:
        st.markdown("---")
        st.subheader("👀 Vista Previa")
        
        preview_letter = st.selectbox("Letra para vista previa", selected_letters, key="preview_letter")
        
        if preview_letter:
            preview_col1, preview_col2, preview_col3 = st.columns(3)
            
            # Generar 3 imágenes de muestra
            sample_params = [
                {'font_size': 24, 'rotation': -10, 'noise_level': 0.0, 'font': 'arial.ttf'},
                {'font_size': 32, 'rotation': 0, 'noise_level': 0.1, 'font': 'times.ttf'},
                {'font_size': 36, 'rotation': 15, 'noise_level': 0.2, 'font': 'calibri.ttf'}
            ]
            
            for i, (col, params) in enumerate(zip([preview_col1, preview_col2, preview_col3], sample_params)):
                with col:
                    st.write(f"**Muestra {i+1}:**")
                    try:
                        preview_img = generate_letter_image(
                            preview_letter,
                            params['font_size'],
                            params['rotation'],
                            params['noise_level'],
                            params['font']
                        )
                        
                        if preview_img:
                            st.image(preview_img, width=100)
                            st.caption(f"Font: {params['font_size']}, Rot: {params['rotation']}°, Ruido: {params['noise_level']}")
                        
                    except Exception as e:
                        st.error(f"Error en preview: {e}")

def generate_images_for_letters(letters, variations_per_letter, visual_config, master_config, use_random_params, fixed_params):
    """Genera imágenes para las letras especificadas"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_operations = len(letters) * variations_per_letter
    current_operation = 0
    
    # Asegurar que existe la estructura de imágenes generadas
    if 'generated_images' not in master_config['visual_dataset']:
        master_config['visual_dataset']['generated_images'] = {}
    
    for letter_idx, letter in enumerate(letters):
        status_text.text(f"Generando imágenes para la letra '{letter.upper()}'...")
        
        # Inicializar lista para esta letra si no existe
        if letter not in master_config['visual_dataset']['generated_images']:
            master_config['visual_dataset']['generated_images'][letter] = []
        
        letter_images = []
        
        for variation in range(variations_per_letter):
            try:
                if use_random_params:
                    # Parámetros aleatorios
                    font_size = random.choice(visual_config.get('font_sizes', [32]))
                    rotation = random.uniform(-visual_config.get('rotation_range', 15), visual_config.get('rotation_range', 15))
                    noise_level = random.choice(visual_config.get('noise_levels', [0.0]))
                    font = random.choice(visual_config.get('fonts', ['arial.ttf']))
                else:
                    # Parámetros fijos
                    font_size = fixed_params['font_size']
                    rotation = fixed_params['rotation']
                    noise_level = fixed_params['noise_level']
                    font = fixed_params['font']
                
                # Generar imagen
                img = generate_letter_image(letter, font_size, rotation, noise_level, font)
                
                if img:
                    # Preparar parámetros para guardar
                    image_params = {
                        'font_size': font_size,
                        'rotation': round(rotation, 2),
                        'noise_level': noise_level,
                        'font': font
                    }
                    
                    # Guardar imagen como archivo y obtener metadatos
                    image_metadata = save_image_to_file(img, letter, image_params)
                    
                    if image_metadata:
                        letter_images.append(image_metadata)
                
                current_operation += 1
                progress_bar.progress(current_operation / total_operations)
                
            except Exception as e:
                st.warning(f"Error generando variación {variation + 1} para letra '{letter}': {e}")
                current_operation += 1
                progress_bar.progress(current_operation / total_operations)
        
        # Agregar nuevas imágenes a las existentes
        master_config['visual_dataset']['generated_images'][letter].extend(letter_images)
        
        status_text.text(f"✅ Completada letra '{letter.upper()}' - {len(letter_images)} imágenes generadas")
    
    # Guardar configuración actualizada
    if save_master_config(master_config):
        st.success(f"🎉 ¡Generación completada! Se generaron {sum(len(master_config['visual_dataset']['generated_images'].get(l, [])) for l in letters)} imágenes para {len(letters)} letras")
        progress_bar.progress(1.0)
        status_text.text("✅ Todas las imágenes guardadas exitosamente")
    else:
        st.error("❌ Error guardando las imágenes generadas")

def clean_corrupted_image_data(config):
    """Limpia datos de imagen corruptos (con image: None)"""
    cleaned_count = 0
    
    if 'visual_dataset' in config and 'generated_images' in config['visual_dataset']:
        generated_images = config['visual_dataset']['generated_images']
        
        for letter, images in generated_images.items():
            # Filtrar imágenes corruptas (sin file_path y con image: None)
            valid_images = []
            for img_data in images:
                if 'file_path' in img_data and img_data['file_path']:
                    valid_images.append(img_data)  # Imagen con archivo válido
                elif 'image' in img_data and img_data['image'] and img_data['image'] != None:
                    valid_images.append(img_data)  # Base64 válido (legacy)
                else:
                    cleaned_count += 1  # Imagen corrupta
            
            generated_images[letter] = valid_images
    
    return cleaned_count

def management_tab(visual_config, master_config):
    """Tab de gestión del dataset visual"""
    st.header("📁 Gestión del Dataset Visual")
    
    # Sección de migración y PyTorch
    st.subheader("🔄 Migración a Sistema de Archivos")
    
    col_migrate1, col_migrate2 = st.columns(2)
    
    with col_migrate1:
        st.markdown("**📂 Nuevo Sistema de Archivos**")
        st.info("""
        El nuevo sistema almacena imágenes como archivos PNG individuales 
        en lugar de base64 en JSON. Esto es más eficiente para entrenamiento 
        con PyTorch y reduce el tamaño del archivo de configuración.
        """)
        
        # Verificar si hay imágenes base64 para migrar
        has_base64_images = False
        base64_count = 0
        
        # Buscar en múltiples secciones
        sections_to_check = [
            ('alphabet', 'variaciones'),
            ('generated_samples', None),
            ('visual_dataset', 'generated_images')
        ]
        
        for section_path, sub_key in sections_to_check:
            try:
                current_data = master_config
                for key in section_path.split('.') if '.' in section_path else [section_path]:
                    if key in current_data:
                        current_data = current_data[key]
                    else:
                        current_data = None
                        break
                
                if not current_data:
                    continue
                    
                if isinstance(current_data, dict):
                    for letter_data in current_data.values():
                        images_list = []
                        
                        if sub_key and isinstance(letter_data, dict) and sub_key in letter_data:
                            images_list = letter_data[sub_key]
                        elif isinstance(letter_data, list):
                            images_list = letter_data
                        
                        for img_data in images_list:
                            if isinstance(img_data, dict) and 'image' in img_data and img_data.get('image', '').strip() and 'file_path' not in img_data:
                                has_base64_images = True
                                base64_count += 1
                                
            except Exception as e:
                st.warning(f"Error verificando sección {section_path}: {str(e)}")
        
        if has_base64_images:
            st.warning(f"📊 Encontradas {base64_count} imágenes en formato base64 para migrar")
            
            output_dir = st.text_input(
                "Directorio de salida:", 
                value="visual_dataset",
                help="Directorio donde se guardarán las imágenes"
            )
            
            if st.button("🔄 Migrar a Archivos", key="migrate_to_files"):
                with st.spinner("Migrando imágenes..."):
                    # Crear estructura de directorios
                    create_pytorch_dataset_structure(master_config, output_dir)
                    
                    # Migrar imágenes
                    migrated = migrate_base64_to_files(master_config, output_dir)
                    
                    if migrated > 0:
                        # Guardar configuración actualizada
                        if save_master_config(master_config):
                            st.success(f"✅ {migrated} imágenes migradas exitosamente a '{output_dir}'")
                            st.info("📁 Estructura de directorios PyTorch creada")
                            st.rerun()
                        else:
                            st.error("❌ Error guardando configuración actualizada")
                    else:
                        st.warning("⚠️ No se migraron imágenes")
        else:
            st.success("✅ Todas las imágenes ya están en formato de archivo")
    
    with col_migrate2:
        st.markdown("**🧠 Configuración PyTorch**")
        
        # Mostrar estadísticas del dataset actual
        file_count = 0
        
        # Contar archivos en diferentes secciones
        for section_path, sub_key in sections_to_check:
            try:
                current_data = master_config
                for key in section_path.split('.') if '.' in section_path else [section_path]:
                    if key in current_data:
                        current_data = current_data[key]
                    else:
                        current_data = None
                        break
                
                if not current_data or not isinstance(current_data, dict):
                    continue
                    
                for letter_data in current_data.values():
                    images_list = []
                    
                    if sub_key and isinstance(letter_data, dict) and sub_key in letter_data:
                        images_list = letter_data[sub_key]
                    elif isinstance(letter_data, list):
                        images_list = letter_data
                    
                    for img_data in images_list:
                        if isinstance(img_data, dict) and 'file_path' in img_data:
                            file_count += 1
                            
            except Exception as e:
                st.warning(f"Error contando archivos en {section_path}: {str(e)}")
        
        st.metric("Imágenes en archivos", file_count)
        
        dataset_path = st.text_input(
            "Ruta del dataset:", 
            value="visual_dataset",
            help="Directorio base del dataset PyTorch"
        )
        
        if st.button("🏗️ Crear Estructura PyTorch", key="create_pytorch_structure"):
            if master_config:
                base_path = create_pytorch_dataset_structure(master_config, dataset_path)
                st.success(f"✅ Estructura PyTorch creada en: {base_path}")
                
                # Mostrar estructura creada
                classes_file = base_path / "classes.txt"
                if classes_file.exists():
                    with open(classes_file, 'r', encoding='utf-8') as f:
                        classes = f.read().strip().split('\n')
                    st.info(f"📝 Archivo classes.txt creado con {len(classes)} clases")
            else:
                st.error("❌ Error cargando configuración")
    
    st.divider()
    
    management_col1, management_col2 = st.columns(2)
    
    with management_col1:
        st.subheader("🗂️ Limpieza y Mantenimiento")
        
        # Mostrar estadísticas actuales
        generated_images = master_config['visual_dataset'].get('generated_images', {})
        total_images = sum(len(samples) for samples in generated_images.values())
        letters_with_images = len([l for l in generated_images.keys() if len(generated_images[l]) > 0])
        
        st.markdown(f"""
        **📊 Estado Actual:**
        - 🔤 Letras con imágenes: {letters_with_images}
        - 🖼️ Total de imágenes: {total_images}
        - 💾 Tamaño estimado: ~{total_images * 2} KB
        """)
        
        # Opciones de limpieza
        st.markdown("**🧹 Opciones de Limpieza:**")
        
        # Limpiar datos corruptos
        if st.button("🔧 Limpiar Datos Corruptos", key="clean_corrupted_data"):
            cleaned_count = clean_corrupted_image_data(master_config)
            if cleaned_count > 0:
                if save_master_config(master_config):
                    st.success(f"✅ {cleaned_count} imágenes corruptas eliminadas")
                    st.rerun()
                else:
                    st.error("❌ Error guardando después de limpieza")
            else:
                st.info("✨ No se encontraron datos corruptos")
        
        if st.button("🗑️ Limpiar Todas las Imágenes", key="clear_all_images"):
            if st.session_state.get('confirm_clear_all', False):
                master_config['visual_dataset']['generated_images'] = {}
                if save_master_config(master_config):
                    st.success("✅ Todas las imágenes han sido eliminadas")
                    st.rerun()
                else:
                    st.error("❌ Error eliminando imágenes")
                st.session_state['confirm_clear_all'] = False
            else:
                st.session_state['confirm_clear_all'] = True
                st.warning("⚠️ Haz clic nuevamente para confirmar la eliminación de todas las imágenes")
        
        # Limpiar por letra específica
        if generated_images:
            letter_to_clear = st.selectbox(
                "Seleccionar letra para limpiar",
                options=list(generated_images.keys()),
                key="letter_to_clear"
            )
            
            if letter_to_clear and st.button(f"🗑️ Limpiar letra '{letter_to_clear.upper()}'"):
                master_config['visual_dataset']['generated_images'][letter_to_clear] = []
                if save_master_config(master_config):
                    st.success(f"✅ Imágenes de la letra '{letter_to_clear.upper()}' eliminadas")
                    st.rerun()
    
    with management_col2:
        st.subheader("📤 Exportar/Importar")
        
        # Exportar configuración
        if total_images > 0:
            st.markdown("**💾 Exportar Dataset:**")
            
            export_format = st.selectbox(
                "Formato de exportación",
                ["Configuración completa (JSON)", "Solo configuración (sin imágenes)", "Estadísticas resumidas"]
            )
            
            if st.button("📤 Exportar Dataset Visual"):
                export_visual_dataset(master_config['visual_dataset'], export_format)
        
        # Importar configuración
        st.markdown("**📥 Importar Dataset:**")
        
        uploaded_file = st.file_uploader(
            "Cargar archivo de configuración visual",
            type=['json'],
            help="Archivo JSON con configuración visual compatible"
        )
        
        if uploaded_file is not None:
            try:
                imported_config = json.load(uploaded_file)
                
                # Validar estructura
                if 'generated_images' in imported_config or 'name' in imported_config:
                    st.success("✅ Archivo válido detectado")
                    
                    # Mostrar preview
                    imported_images = sum(len(samples) for samples in imported_config.get('generated_images', {}).values())
                    st.info(f"📊 El archivo contiene {imported_images} imágenes")
                    
                    if st.button("📥 Importar Configuración Visual"):
                        # Merge con configuración existente
                        if 'generated_images' in imported_config:
                            for letter, images in imported_config['generated_images'].items():
                                if letter not in master_config['visual_dataset']['generated_images']:
                                    master_config['visual_dataset']['generated_images'][letter] = []
                                master_config['visual_dataset']['generated_images'][letter].extend(images)
                        
                        if save_master_config(master_config):
                            st.success("✅ Dataset visual importado exitosamente!")
                            st.rerun()
                        else:
                            st.error("❌ Error importando dataset")
                
                else:
                    st.error("❌ Archivo no válido - falta estructura requerida")
            
            except Exception as e:
                st.error(f"❌ Error procesando archivo: {e}")

def dataset_configuration_tab(visual_config, master_config):
    """Tab de configuración avanzada del dataset"""
    st.header("📋 Configuración Avanzada del Dataset")
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.subheader("🔗 Sincronización con Master Config")
        
        # Mostrar estado de sincronización
        master_vocab = master_config.get('master_config_reference', 'N/A')
        visual_vocab = visual_config.get('vocabulary', 'N/A')
        
        if master_vocab == visual_vocab:
            st.success(f"✅ Sincronizado con: **{master_vocab}**")
        else:
            st.warning(f"⚠️ Desincronizado: Master={master_vocab}, Visual={visual_vocab}")
            
            if st.button("🔄 Forzar Sincronización"):
                update_visual_vocabulary()
                st.success("✅ Sincronización forzada completada")
                st.rerun()
        
        # Información del vocabulario maestro
        if master_vocab in master_config.get('generated_samples', {}):
            vocab_samples = master_config['generated_samples'][master_vocab]
            st.info(f"📚 Vocabulario maestro contiene {len(vocab_samples)} muestras de audio")
            
            # Mostrar palabras del vocabulario
            words = [sample.get('word', '') for sample in vocab_samples[:10]]
            st.write("**Palabras de muestra:**", ', '.join(words))
    
    with config_col2:
        st.subheader("⚙️ Configuración Técnica")
        
        # Configuraciones avanzadas
        batch_size = st.number_input(
            "Tamaño de lote para generación",
            min_value=1,
            max_value=100,
            value=10,
            help="Número de imágenes a procesar simultáneamente"
        )
        
        max_images_per_letter = st.number_input(
            "Máximo de imágenes por letra",
            min_value=10,
            max_value=1000,
            value=100,
            help="Límite máximo de imágenes almacenadas por letra"
        )
        
        auto_cleanup = st.checkbox(
            "Auto-limpieza al exceder límite",
            value=False,
            help="Eliminar automáticamente imágenes más antiguas al superar el límite"
        )
        
        # Configuración de calidad
        st.markdown("**🎨 Configuración de Calidad:**")
        
        image_quality = st.slider(
            "Calidad de compresión (%)",
            min_value=50,
            max_value=100,
            value=90
        )
        
        use_antialiasing = st.checkbox(
            "Usar anti-aliasing",
            value=True,
            help="Mejora la calidad visual de las letras"
        )
        
        # Guardar configuraciones avanzadas
        if st.button("💾 Guardar Configuración Avanzada"):
            master_config['visual_dataset']['advanced_config'] = {
                'batch_size': batch_size,
                'max_images_per_letter': max_images_per_letter,
                'auto_cleanup': auto_cleanup,
                'image_quality': image_quality,
                'use_antialiasing': use_antialiasing
            }
            
            if save_master_config(master_config):
                st.success("✅ Configuración avanzada guardada!")
            else:
                st.error("❌ Error guardando configuración")
    
    # Información del sistema
    st.markdown("---")
    st.subheader("🔍 Información del Sistema")
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.markdown("**📁 Archivos:**")
        st.write(f"- Master config: ✅ Existe")
        st.write(f"- Visual config: 🔗 Integrado")
        
    with info_col2:
        st.markdown("**📊 Estadísticas:**")
        generated_images = master_config['visual_dataset'].get('generated_images', {})
        st.write(f"- Letras: {len(generated_images)}")
        st.write(f"- Imágenes: {sum(len(samples) for samples in generated_images.values())}")
        
    with info_col3:
        st.markdown("**🕐 Timestamps:**")
        created = visual_config.get('created', 'N/A')
        if created != 'N/A':
            try:
                fecha = datetime.fromisoformat(created)
                st.write(f"- Creado: {fecha.strftime('%d/%m/%Y')}")
            except:
                st.write(f"- Creado: {created[:10]}")
        st.write(f"- Versión: {visual_config.get('version', 'N/A')}")

def export_visual_dataset(visual_config, export_format):
    """Exporta el dataset visual en el formato especificado"""
    
    if export_format == "Solo configuración (sin imágenes)":
        # Exportar solo configuración sin las imágenes
        config_export = visual_config.copy()
        config_export['generated_images'] = {}
        
        config_json = json.dumps(config_export, indent=2, ensure_ascii=False)
        st.download_button(
            label="💾 Descargar Configuración Visual",
            data=config_json,
            file_name=f"visual_dataset_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    elif export_format == "Estadísticas resumidas":
        # Exportar solo estadísticas
        generated_images = visual_config.get('generated_images', {})
        stats = {
            'name': visual_config.get('name', 'Visual Dataset'),
            'letters_count': len(generated_images),
            'total_images': sum(len(samples) for samples in generated_images.values()),
            'letters_with_images': [letter for letter, images in generated_images.items() if len(images) > 0],
            'export_timestamp': datetime.now().isoformat(),
            'version': visual_config.get('version', '1.0')
        }
        
        stats_json = json.dumps(stats, indent=2, ensure_ascii=False)
        st.download_button(
            label="📊 Descargar Estadísticas",
            data=stats_json,
            file_name=f"visual_dataset_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    else:  # Configuración completa (JSON)
        # Exportar configuración completa con todas las imágenes
        config_json = json.dumps(visual_config, indent=2, ensure_ascii=False, default=str)
        st.download_button(
            label="💾 Descargar Dataset Completo",
            data=config_json,
            file_name=f"visual_dataset_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()