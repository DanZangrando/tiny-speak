import json
from pathlib import Path
from datetime import datetime
import os

EXPERIMENTS_DIR = Path("experiments")
EXPERIMENTS_DIR.mkdir(exist_ok=True)

def get_opacity_index(language):
    """
    Retorna el índice de opacidad ortográfica teórico (aproximado) para un idioma.
    0.0 = Transparente (1:1 grafema-fonema)
    1.0 = Opaco (Relación compleja/irregular)
    """
    indices = {
        'es': 0.1,  # Español: Muy transparente
        'it': 0.15, # Italiano: Muy transparente
        'de': 0.3,  # Alemán: Moderadamente transparente
        'pt': 0.35, # Portugués: Moderado
        'fr': 0.7,  # Francés: Opaco (muchas letras mudas, digrafos)
        'en': 0.9   # Inglés: Muy opaco (profunda ortografía)
    }
    return indices.get(language, 0.5)

def save_experiment(experiment_id, data):
    """Guarda los datos de un experimento en un archivo JSON."""
    file_path = EXPERIMENTS_DIR / f"{experiment_id}.json"
    
    # Asegurar que data tenga timestamp
    if "timestamp" not in data:
        data["timestamp"] = datetime.now().isoformat()
        
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return str(file_path)

def load_experiment(experiment_id):
    """Carga los datos de un experimento."""
    file_path = EXPERIMENTS_DIR / f"{experiment_id}.json"
    if not file_path.exists():
        return None
    
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def list_experiments():
    """Lista todos los experimentos disponibles."""
    files = sorted(list(EXPERIMENTS_DIR.glob("*.json")), key=os.path.getmtime, reverse=True)
    experiments = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)
                experiments.append({
                    "id": f.stem,
                    "name": data.get("name", f.stem),
                    "timestamp": data.get("timestamp", ""),
                    "languages": data.get("languages", [])
                })
        except:
            continue
    return experiments

def check_dataset_status(language):
    """Verifica si existen datos para un idioma."""
    audio_path = Path(f"data/audios/{language}")
    visual_path = Path(f"data/visual/{language}") # Aunque visual suele ser compartido o generado on-the-fly
    
    # Para visual, a veces se generan, así que verificamos si hay configuración o carpeta
    # Asumimos que si hay audio, podemos entrenar.
    
    return {
        "audio": audio_path.exists() and any(audio_path.iterdir()),
        "visual": True # Por ahora asumimos True ya que se generan sintéticamente o se descargan
    }
