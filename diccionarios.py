import os
from pathlib import Path

# Definición de metadatos de los diccionarios
# Las palabras se cargan desde archivos .txt en data/diccionarios/
DICCIONARIOS_METADATA = {
    "tiny_kalulu_original": {
        "nombre": "TinySpeak Original - Kalulu (200 palabras)",
        "descripcion": "Vocabulario original del proyecto TinySpeak basado en el dataset Kalulu",
        "archivo": "tiny_kalulu_original.txt"
    },
    "basico_español": {
        "nombre": "Básico Español (50 palabras)",
        "descripcion": "Vocabulario básico en español para principiantes",
        "archivo": "basico_español.txt"
    },
    "colores_numeros": {
        "nombre": "Colores y Números (30 palabras)",
        "descripcion": "Vocabulario de colores básicos y números del 1 al 20",
        "archivo": "colores_numeros.txt"
    },
    "animales": {
        "nombre": "Animales (40 palabras)",
        "descripcion": "Vocabulario de animales comunes",
        "archivo": "animales.txt"
    },
    "casa_familia": {
        "nombre": "Casa y Familia (35 palabras)",
        "descripcion": "Vocabulario del hogar y relaciones familiares",
        "archivo": "casa_familia.txt"
    },
    "acciones_verbos": {
        "nombre": "Acciones y Verbos (45 palabras)",
        "descripcion": "Verbos y acciones comunes",
        "archivo": "acciones_verbos.txt"
    },
    "tiny_phones_original": {
        "nombre": "TinySpeak Original - Phones (200 palabras)",
        "descripcion": "Vocabulario original del proyecto TinySpeak basado en dataset de phonemas",
        "archivo": "tiny_phones_original.txt"
    },
    "comida_bebida": {
        "nombre": "Comida y Bebida (40 palabras)",
        "descripcion": "Vocabulario de alimentos y bebidas",
        "archivo": "comida_bebida.txt"
    }
}

def get_diccionario_predefinido(nombre_diccionario, idioma='es'):
    """
    Obtiene un diccionario predefinido por su nombre e idioma.
    
    Args:
        nombre_diccionario: Clave del diccionario (ej: 'animales')
        idioma: Código de idioma ('es', 'en', 'fr'). Default 'es'.
    """
    metadata = DICCIONARIOS_METADATA.get(nombre_diccionario)
    if not metadata:
        return None
    
    # Construir nombre de archivo según idioma
    filename = metadata["archivo"]
    if idioma and idioma != 'es':
        # Asumimos que los archivos traducidos tienen sufijo _en.txt, _fr.txt
        # ej: animales.txt -> animales_en.txt
        filename = filename.replace(".txt", f"_{idioma}.txt")
        
    # Construir ruta al archivo
    base_dir = Path(__file__).parent
    archivo_path = base_dir / "data" / "diccionarios" / filename
    
    palabras = []
    try:
        if archivo_path.exists():
            with open(archivo_path, 'r', encoding='utf-8') as f:
                palabras = [line.strip() for line in f if line.strip()]
        else:
            # Si no existe la traducción, devolver lista vacía o fallback?
            # Por ahora lista vacía para indicar que no hay datos para ese idioma
            print(f"Advertencia: No se encontró el archivo de diccionario {archivo_path}")
            return None
    except Exception as e:
        print(f"Error leyendo diccionario {nombre_diccionario} ({idioma}): {e}")
        return None
        
    return {
        "nombre": f"{metadata['nombre']} ({idioma.upper()})",
        "descripcion": metadata["descripcion"],
        "palabras": palabras,
        "idioma": idioma
    }

def get_nombres_diccionarios():
    """Obtiene la lista de nombres de diccionarios disponibles"""
    return list(DICCIONARIOS_METADATA.keys())

def get_info_diccionarios():
    """Obtiene información resumida de todos los diccionarios"""
    info = {}
    for key in DICCIONARIOS_METADATA.keys():
        dic_data = get_diccionario_predefinido(key)
        if dic_data:
            info[key] = {
                "nombre": dic_data["nombre"],
                "descripcion": dic_data["descripcion"],
                "cantidad_palabras": len(dic_data["palabras"])
            }
    return info

# Construir DICCIONARIOS_PREDEFINIDOS para compatibilidad hacia atrás
# Esto carga todos los diccionarios en memoria al importar el módulo
DICCIONARIOS_PREDEFINIDOS = {}
for key in DICCIONARIOS_METADATA.keys():
    dic_data = get_diccionario_predefinido(key)
    if dic_data:
        DICCIONARIOS_PREDEFINIDOS[key] = dic_data