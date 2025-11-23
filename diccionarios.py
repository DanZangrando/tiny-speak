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

def get_diccionario_predefinido(nombre_diccionario):
    """Obtiene un diccionario predefinido por su nombre, cargando las palabras desde el archivo"""
    metadata = DICCIONARIOS_METADATA.get(nombre_diccionario)
    if not metadata:
        return None
    
    # Construir ruta al archivo
    base_dir = Path(__file__).parent
    archivo_path = base_dir / "data" / "diccionarios" / metadata["archivo"]
    
    palabras = []
    try:
        if archivo_path.exists():
            with open(archivo_path, 'r', encoding='utf-8') as f:
                palabras = [line.strip() for line in f if line.strip()]
        else:
            print(f"Advertencia: No se encontró el archivo de diccionario {archivo_path}")
    except Exception as e:
        print(f"Error leyendo diccionario {nombre_diccionario}: {e}")
        
    return {
        "nombre": metadata["nombre"],
        "descripcion": metadata["descripcion"],
        "palabras": palabras
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