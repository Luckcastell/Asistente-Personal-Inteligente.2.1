from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
import os, shutil
from dotenv import load_dotenv

# Cargar variables de entorno (como la API Key de Groq)
load_dotenv()

# --- Configuración y Inicialización ---

# Inicializar cliente de Groq (para el modelo de lenguaje)
CLAVE_GROQ = os.getenv("GROQ_API_KEY")
if not CLAVE_GROQ:
    # Levantar una excepción si la clave no está configurada, crucial para la ejecución
    raise ValueError("GROQ_API_KEY no encontrada. Por favor, revisá el archivo .env")

cliente_llm = Groq(api_key=CLAVE_GROQ)

# Directorios para archivos y base de datos de vectores
DIRECTORIO_SUBIDAS = "uploads"
DIRECTORIO_DB = "vector_db"

# Crear directorios si no existen
os.makedirs(DIRECTORIO_SUBIDAS, exist_ok=True)
os.makedirs(DIRECTORIO_DB, exist_ok=True)

# Inicializar FastAP
app = FastAPI(title="Backend Suriel RAG")

# Configurar CORS (necesario para que el frontend pueda comunicarse)
app.add_middleware(
    CORSMiddleware,
    # Permitir cualquier origen durante el desarrollo. En producción, se debería restringir
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar Embeddings (función para convertir texto a vectores)
# Se usa un modelo local de HuggingFace para evitar depender de APIs de embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Inicializar ChromaDB (la base de datos de vectores) y cargar la base persistente
vectorstore = Chroma(
    persist_directory=DIRECTORIO_DB,
    embedding_function=embeddings
)

# Clase de Pydantic para validar la solicitud de chat
class SolicitudChat(BaseModel):
    """Define la estructura de datos esperada para la solicitud de chat."""
    mensaje: str

# --- Funciones de Lógica RAG ---

def agregar_pdf_a_vectorstore(ruta_archivo: str):
    """
    Procesa un archivo PDF, lo divide en trozos y los indexa en la base de vectores.
    :param ruta_archivo: La ruta completa al archivo PDF a procesar.
    """
    try:
        # Cargar el documento PDF
        cargador = PyPDFLoader(ruta_archivo)
        documentos = cargador.load()

        # Dividir el texto en trozos para la búsqueda precisa
        # Un tamaño de 1000 y un solapamiento de 100 son comunes para RAG
        divisor_texto = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        trozos = divisor_texto.split_documents(documentos)

        # Agregar los trozos a la tienda de vectores
        vectorstore.add_documents(trozos)

        # Persistir los cambios en el disco para que estén disponibles después de reiniciar
        vectorstore.persist()
        return True
    except Exception as e:
        print(f"Error al procesar PDF {ruta_archivo}: {e}")
        return False

# --- Rutas de la API ---

@app.post("/upload")
# CAMBIO CLAVE: Cambiamos el nombre del parámetro de 'archivo' a 'file'
# para que coincida con lo que envía el 'formData.append("file", ...)' del script.js
async def subir_pdf(file: UploadFile = File(...)):
    """
    Ruta para subir un archivo PDF, guardarlo temporalmente y luego indexarlo.
    """
    # Usamos 'file' en lugar de 'archivo'
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF.")

    ruta_archivo = os.path.join(DIRECTORIO_SUBIDAS, file.filename)

    try:
        # Guardar el archivo temporalmente (usamos file.file)
        with open(ruta_archivo, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Procesar el archivo y agregarlo a la base de vectores
        if agregar_pdf_a_vectorstore(ruta_archivo):
            # Eliminar el archivo temporal después de procesar
            os.remove(ruta_archivo)
            return {"mensaje": f"PDF '{file.filename}' cargado y agregado al índice."}
        else:
            raise HTTPException(status_code=500, detail="Error interno al procesar el PDF.")

    except Exception as e:
        # En caso de error, aseguramos la limpieza del archivo temporal
        if os.path.exists(ruta_archivo):
            os.remove(ruta_archivo)
        print(f"Error en la ruta /upload: {e}")
        raise HTTPException(status_code=500, detail=f"Ocurrió un error: {e}")

@app.post("/chat")
async def chat(solicitud: SolicitudChat):
    """
    Ruta para enviar un mensaje de chat, buscar contexto en la base de vectores y generar una respuesta.
    """
    try:
        # Paso 1: Recuperación (Retrieval) - Buscar los 3 documentos más relevantes
        resultados = vectorstore.similarity_search(solicitud.mensaje, k=3)
        
        # Unir el contenido de los documentos recuperados en una sola cadena de contexto
        contexto = "\n\n".join([doc.page_content for doc in resultados])

        # Paso 2: Aumento (Augmentation) - Crear el prompt con el rol y el contexto
        prompt = f"""Eres Suriel, un asistente amable y servicial que responde SOLO con la información de la base de datos privada.
Si la respuesta no está claramente en la base de datos, responde simplemente: "No tengo esa información en mi base de conocimiento actual."
No inventes ni uses conocimiento externo.

Contexto relevante de la base de datos:
{contexto}

Pregunta del usuario: {solicitud.mensaje}
Respuesta:
"""

        # Paso 3: Generación (Generation) - Llamar al modelo de lenguaje
        respuesta_llm = cliente_llm.chat.completions.create(
            # ¡IMPORTANTE! El modelo ha sido actualizado para solucionar el error de "model decommissioned"
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Eres Suriel, un asistente amable y servicial."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, # Temperatura baja para respuestas más fácticas
        )

        respuesta_generada = respuesta_llm.choices[0].message.content

        # Devolvemos la respuesta generada por el LLM
        return {"respuesta": respuesta_generada}

    except Exception as e:
        # Capturamos la excepción para el debugging
        print(f"Error en la ruta /chat: {e}")
        # Retornamos un error HTTP amigable para el frontend
        raise HTTPException(status_code=500, detail=f"Error interno al generar la respuesta: {e}")

# Punto de inicio para Uvicorn si se ejecuta directamente
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
