# Asistente de documentos con IA

Este proyecto crea una aplicación web con Streamlit que permite cargar archivos PDF, indexarlos en un almacén vectorial y hacer preguntas sobre su contenido mediante un agente de IA usando Groq y LangChain. El usuario recibe la respuesta a su consulkta simepre y cuanto la información esté contenida dentro del pdf que subió previamente.

## Función del proyecto

La aplicación tiene dos partes principales:

1. Carga de documentos PDF
   - El usuario sube uno o varios archivos PDF desde la interfaz.
   - El sistema extrae su texto y lo divide en fragmentos pequeños.
   - Estos fragmentos se almacenan en un vector store local para poder consultarlos luego.

2. Chat de preguntas sobre los documentos
   - El usuario escribe preguntas en español.
   - La aplicación busca los fragmentos más relevantes en los documentos cargados.
   - El modelo de IA responde usando únicamente la información contenida en esos documentos.
   - Si la información no aparece en los archivos, la aplicación indica que no se encontró en los documentos.

## Tecnologías utilizadas

- Streamlit: para la interfaz web.
- LangChain: para trabajar con agentes, prompts y recuperación de información.
- LangChain Community: para cargar PDFs y trabajar con embeddings.
- FAISS: para almacenar y buscar documentos por similitud.
- Hugging Face embeddings: para convertir texto en vectores.
- Groq: como proveedor del modelo de lenguaje.

## Requisitos

- Python 3.9 o superior
- Una API key de Groq

## Pasos para ejecutar el proyecto

1. Entrar a la carpeta del proyecto
   ```bash
   cd CHALLENGE ALLURA
   ```

2. Crear un entorno virtual
   ```bash
   python -m venv .venv
   ```

3. Activar el entorno virtual
   En Windows:
   ```bash
   .venv\Scripts\activate
   ```

4. Instalar las dependencias
   ```bash
   pip install -r requirements.txt
   ```

5. Ejecutar la aplicación
   ```bash
   streamlit run app.py
   ```

6. Abrir la URL que muestra Streamlit en el navegador
   Por lo general será:
   ```bash
   http://localhost:8501
   ```

## Uso de la aplicación

1. En la barra lateral, pega el API key de Groq.
2. Sube uno o varios archivos PDF.
3. Haz clic en "Indexar documentos" o "Procesar documentos" según la interfaz.
4. Espera a que los documentos se procesen.
5. Escribe preguntas en español en el chat.
6. La IA responderá con base en los documentos cargados.

## Estructura del proyecto

- app.py: archivo principal de la aplicación Streamlit.
- requirements.txt: dependencias necesarias para ejecutar el proyecto.
- README.md: documentación del proyecto.

## Notas importantes

- La aplicación responde solo con información proveniente de los PDFs cargados.
- Si el contenido no está en los documentos, devolverá un mensaje indicando que no se encontró esa información.
- Es necesario ingresar una API key válida de Groq para poder usar el modelo de IA.
