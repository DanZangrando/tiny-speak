#!/usr/bin/env python3
"""
Script de verificación completa post-instalación
Ejecutar después de instalar requirements.txt
"""

import os
import sys
from pathlib import Path

def run_installation_check():
    """Ejecuta verificación de instalación"""
    print("🔍 Ejecutando verificación de instalación...")
    os.system("python3 check_installation.py")

def run_project_checks():
    """Ejecuta verificaciones del proyecto"""
    print("\n🔍 Ejecutando verificaciones del proyecto...")
    
    # Verificación básica
    print("\n1. Verificación básica:")
    os.system("python3 check_basic.py")
    
    # Verificación avanzada
    print("\n2. Verificación avanzada:")
    os.system("python3 check_advanced.py")
    
    # Verificación final
    print("\n3. Verificación final:")
    os.system("python3 check_final.py")

def test_streamlit_basic():
    """Test básico de Streamlit"""
    print("\n🚀 PRUEBA BÁSICA DE STREAMLIT")
    print("=" * 50)
    
    try:
        import streamlit as st
        print("✅ Streamlit importado correctamente")
        
        # Verificar que app.py existe y es válido
        if Path("app.py").exists():
            print("✅ app.py encontrado")
            
            # Intentar importar componentes del proyecto
            sys.path.append(".")
            try:
                import utils
                print("✅ utils.py importado")
            except Exception as e:
                print(f"⚠️ Error importando utils: {e}")
            
            try:
                import models
                print("✅ models.py importado")
            except Exception as e:
                print(f"⚠️ Error importando models: {e}")
            
            try:
                import diccionarios
                print("✅ diccionarios.py importado")
            except Exception as e:
                print(f"⚠️ Error importando diccionarios: {e}")
                
        else:
            print("❌ app.py no encontrado")
            
    except ImportError as e:
        print(f"❌ No se pudo importar Streamlit: {e}")
        return False
    
    return True

def main():
    """Función principal"""
    print("🎯 VERIFICACIÓN COMPLETA POST-INSTALACIÓN")
    print("🚀 TinySpeak - RTX 5090 + CUDA 12.9")
    print("=" * 60)
    
    # 1. Verificar instalación de dependencias
    run_installation_check()
    
    # 2. Verificar proyecto
    run_project_checks()
    
    # 3. Test básico de Streamlit
    test_streamlit_basic()
    
    # 4. Instrucciones finales
    print("\n🎉 VERIFICACIÓN COMPLETA FINALIZADA")
    print("=" * 60)
    print("📝 INSTRUCCIONES FINALES:")
    print("1. Si todas las verificaciones pasaron:")
    print("   streamlit run app.py")
    print("\n2. La aplicación estará disponible en:")
    print("   http://localhost:8501")
    print("\n3. Para verificar GPU en tiempo real:")
    print("   watch -n 1 nvidia-smi")
    print("\n4. Logs de la aplicación:")
    print("   tail -f ~/.streamlit/logs/streamlit.log")
    
if __name__ == "__main__":
    main()