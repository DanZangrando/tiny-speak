import streamlit as st
import inspect
from pathlib import Path

def display_code_snippet(file_path, object_name=None, start_line=None, end_line=None, language="python"):
    """
    Muestra un snippet de código en Streamlit.
    Puede mostrar un objeto específico (función/clase) o un rango de líneas.
    """
    try:
        content = Path(file_path).read_text(encoding='utf-8')
        lines = content.splitlines()
        
        code_to_show = ""
        
        if object_name:
            # Intentar encontrar el objeto parseando (simple)
            # Nota: Esto es frágil, idealmente usaríamos AST o importación dinámica,
            # pero para visualización simple basta con buscar la definición.
            # Mejor enfoque: Importar dinámicamente si es posible, o usar ast.
            # Por simplicidad y seguridad en runtime, usaremos rango de líneas si se provee,
            # o búsqueda de texto simple.
            pass 

        if start_line is not None and end_line is not None:
            # Ajustar a 0-indexed
            snippet_lines = lines[start_line-1:end_line]
            code_to_show = "\n".join(snippet_lines)
        else:
            code_to_show = content

        st.code(code_to_show, language=language)
        st.caption(f"📍 Fuente: `{file_path}`")
        
    except Exception as e:
        st.error(f"Error cargando código: {e}")

def get_function_source(func):
    """Obtiene el código fuente de una función importada."""
    try:
        return inspect.getsource(func)
    except:
        return "# No se pudo obtener el código fuente."
