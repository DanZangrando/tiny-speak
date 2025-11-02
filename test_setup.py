"""
Script de prueba para verificar que los componentes básicos funcionen
"""
import sys
from pathlib import Path

def test_imports():
    """Prueba que todas las importaciones funcionen"""
    print("🔍 Probando importaciones...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        import torchaudio
        print(f"✅ TorchAudio: {torchaudio.__version__}")
    except ImportError as e:
        print(f"❌ TorchAudio: {e}")
        return False
        
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers: {e}")
        return False
        
    try:
        import streamlit
        print(f"✅ Streamlit: {streamlit.__version__}")
    except ImportError as e:
        print(f"❌ Streamlit: {e}")
        return False
    
    return True

def test_device():
    """Prueba la detección de dispositivo"""
    print("\n🖥️  Probando dispositivo...")
    
    try:
        from utils import encontrar_device
        device = encontrar_device()
        print(f"✅ Dispositivo detectado: {device}")
        return True
    except Exception as e:
        print(f"❌ Error detectando dispositivo: {e}")
        return False

def test_espeak():
    """Prueba que espeak funcione"""
    print("\n🔊 Probando espeak...")
    
    try:
        from utils import synthesize_word
        waveform = synthesize_word("prueba")
        if waveform is not None:
            print(f"✅ Espeak funcionando, audio generado: {waveform.shape}")
            return True
        else:
            print("❌ Espeak no generó audio")
            return False
    except Exception as e:
        print(f"❌ Error con espeak: {e}")
        return False

def test_models():
    """Prueba que los modelos se puedan cargar"""
    print("\n🧠 Probando modelos...")
    
    try:
        from models import TinySpeak
        from utils import get_default_words
        
        words = get_default_words()
        print(f"✅ Palabras cargadas: {len(words)} palabras")
        
        model = TinySpeak(words=words[:5])  # Solo las primeras 5 para prueba rápida
        print(f"✅ TinySpeak inicializado: {sum(p.numel() for p in model.parameters())} parámetros")
        
        return True
    except Exception as e:
        print(f"❌ Error con modelos: {e}")
        return False

def main():
    """Ejecuta todas las pruebas"""
    print("🚀 TinySpeak - Prueba de Componentes\n")
    
    tests = [
        test_imports,
        test_device,
        test_espeak,
        test_models
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Resultados: {passed}/{len(tests)} pruebas pasaron")
    
    if passed == len(tests):
        print("🎉 ¡Todos los componentes funcionan correctamente!")
        print("🚀 Puedes ejecutar la aplicación con: streamlit run app.py")
        return True
    else:
        print("⚠️  Algunos componentes necesitan atención")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)