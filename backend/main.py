from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document 
from groq import Groq
import os
import shutil
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

# --- SOLUCIÓN DE RUTAS ABSOLUTAS ---
# Obtenemos la ruta absoluta del directorio actual (donde está main.py)
RUTA_BASE_DEL_PROYECTO = os.path.dirname(os.path.abspath(__file__))

# Rutas de directorios basadas en la ruta absoluta
DIRECTORIO_SUBIDAS = os.path.join(RUTA_BASE_DEL_PROYECTO, "uploads")
DIRECTORIO_DB = os.path.join(RUTA_BASE_DEL_PROYECTO, "vector_db") 

# Constantes para identificar y prefijar los trozos de contenido
PREFIJO_CHAT = "MEMORIA_CHAT: "
PREFIJO_DOCUMENTO = "DOCUMENTO_RAG: " 

# Crear directorios si no existen (usa las nuevas rutas absolutas)
os.makedirs(DIRECTORIO_SUBIDAS, exist_ok=True)
os.makedirs(DIRECTORIO_DB, exist_ok=True)

# Inicializar FastAPI
app = FastAPI(title="Backend Suriel RAG")

# Configurar CORS para permitir peticiones desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar el modelo de embeddings
try:
    modelo_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    raise RuntimeError(f"No se pudo cargar el modelo de embeddings: {e}")

# Inicializar la base de datos de vectores Chroma (usa la ruta absoluta)
vectorstore = Chroma(
    persist_directory=DIRECTORIO_DB,
    embedding_function=modelo_embeddings
)

# --- Esquemas Pydantic para la API ---
class SolicitudChat(BaseModel):
    """Define el esquema para la petición de chat del usuario."""
    mensaje: str

# --- Funciones Auxiliares ---

def obtener_conteo_total_de_vectores():
    """
    Función de diagnóstico que retorna el número total de documentos/trozos
    indexados actualmente en la base de vectores Chroma.
    """
    # Accedemos al objeto collection interno de Chroma para obtener el conteo
    return vectorstore._collection.count() 

def agregar_pdf_a_vectorstore(ruta_archivo: str, nombre_archivo: str):
    """
    Carga un PDF, lo divide en trozos e indexa en ChromaDB.
    """
    try:
        # 1. Cargar el PDF
        cargador = PyPDFLoader(ruta_archivo)
        documentos = cargador.load()
        
        # 2. Agregar metadatos y PREFIJO a cada documento
        for doc in documentos:
            # El metadato 'source' es clave para el filtrado en la ruta /chat
            doc.metadata["source"] = nombre_archivo 
            doc.page_content = PREFIJO_DOCUMENTO + doc.page_content 

        # 3. Dividir el documento en trozos
        divisor_texto = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )
        trozos = divisor_texto.split_documents(documentos)

        # 4. Indexar los trozos en la base de vectores
        vectorstore.add_documents(trozos)
        
        # 5. Persistir los cambios para que se guarden inmediatamente en disco
        vectorstore.persist()
        
        # DEBUG: Confirmación en la consola
        print(f"DEBUG: Conteo total de vectores después de indexar: {obtener_conteo_total_de_vectores()}")
        
        return len(trozos)
    except Exception as e:
        # Imprimimos y relanzamos la excepción
        print(f"Error en la indexación: {e}")
        raise RuntimeError(f"Fallo en la indexación del PDF: {e}")

# --- Rutas de la API ---

@app.get("/count_index")
async def contar_documentos_indexados():
    """
    Ruta de diagnóstico para verificar cuántos trozos (documentos + chat)
    están cargados actualmente en la base de vectores.
    """
    try:
        conteo = obtener_conteo_total_de_vectores()
        return {"conteo_total_vectores": conteo}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener el conteo: {e}")

@app.post("/upload")
async def subir_pdf(file: UploadFile = File(...)):
    """
    Ruta para subir un archivo PDF, guardarlo e indexar su contenido en la base de vectores.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF.")

    # 1. Guardar el archivo temporalmente (usando la ruta absoluta)
    ruta_archivo = os.path.join(DIRECTORIO_SUBIDAS, file.filename)
    try:
        with open(ruta_archivo, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Indexar el contenido en ChromaDB
        num_trozos = agregar_pdf_a_vectorstore(ruta_archivo, file.filename)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 3. Limpiar: Eliminar el archivo temporal una vez indexado
        if os.path.exists(ruta_archivo):
            os.remove(ruta_archivo)
    
    return {"mensaje": f"Archivo '{file.filename}' indexado con éxito. Trozos creados: {num_trozos}"}


@app.post("/chat")
async def chat(solicitud: SolicitudChat):
    """
    Gestiona la conversación con el usuario a través del ciclo RAG,
    garantizando la recuperación de documentos y memoria de chat mediante
    búsquedas separadas y filtradas por metadatos (Total de 15 trozos: 9 RAG / 6 Chat).
    """
    try:
        # Paso 1: Recuperación (Retrieval) - Búsquedas Filtradas
        
        # Búsqueda 1: Recuperar Documentos RAG (excluye la memoria del chat)
        documentos_rag = vectorstore.similarity_search(
            solicitud.mensaje, 
            k=9, # 9 TROZOS DE DOCUMENTOS RAG
            # Filtro: la fuente NO debe ser 'Memoria del Chat'
            filter={"source": {"$ne": "Memoria del Chat"}} 
        )
        
        # Búsqueda 2: Recuperar Memoria de Chat
        documentos_memoria = vectorstore.similarity_search(
            solicitud.mensaje, 
            k=6, # 6 TROZOS DE MEMORIA DE CHAT
            # Filtro: la fuente DEBE ser 'Memoria del Chat'
            filter={"source": "Memoria del Chat"} 
        )

        # Procesar Documentos RAG y limpiar el prefijo
        contexto_documentos = [
            doc.page_content.replace(PREFIJO_DOCUMENTO, "") 
            for doc in documentos_rag
        ]
        
        # Procesar Memoria de Chat y limpiar el prefijo
        contexto_chat = [
            doc.page_content.replace(PREFIJO_CHAT, "") 
            for doc in documentos_memoria
        ]

        # --- CONSTRUCCIÓN DEL CONTEXTO COMBINADO ---
        
        # 1. Contexto de Documentos (Base de Conocimiento)
        contexto_rag_documentos = "\n\n--- CONTEXTO DE DOCUMENTOS ---\n" + "\n".join(contexto_documentos)
        
        # 2. Historial de Conversación (Memoria)
        historial_largo_plazo = ""
        if contexto_chat:
            historial_largo_plazo = "\n\n--- HISTORIAL DE CONVERSACIÓN (MEMORIA) ---\n" + "\n".join(contexto_chat)
            
        contexto_combinado = contexto_rag_documentos + historial_largo_plazo

        # Paso 2: Aumento (Augmentation) - Crear el prompt con el rol y el contexto
        # Estructura estricta para forzar el Grounding
        prompt_estricto = f"""Eres Suriel, un asistente amable y servicial que responde SOLO con la información de la base de datos privada.
Si la respuesta no está claramente en la base de datos, responde simplemente: "Disculpá, esa información no se encuentra en la base de conocimiento actual."
No inventes ni uses conocimiento externo.

Contexto relevante de la base de datos y memoria de chat:
{contexto_combinado}

Pregunta del usuario: {solicitud.mensaje}
Respuesta:
"""

        # Paso 3: Generación (Generation) - Llamar al modelo de lenguaje
        respuesta_llm = cliente_llm.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Eres Suriel, un asistente amable y servicial."},
                {"role": "user", "content": prompt_estricto}
            ],
            temperature=0.2,
        )

        respuesta_generada = respuesta_llm.choices[0].message.content

        # Paso 4: Persistencia (Persistence) - Guardar el turno actual como memoria
        turno_completo = f"USUARIO: {solicitud.mensaje}\nSURIEL: {respuesta_generada}"
        
        documento_memoria = Document(
            page_content=PREFIJO_CHAT + turno_completo, 
            metadata={
                "tipo": "chat_turn", 
                "source": "Memoria del Chat" # Clave para el filtro en el Paso 1.
            }
        )

        vectorstore.add_documents([documento_memoria])
        vectorstore.persist()

        return {"respuesta": respuesta_generada}

    except Exception as e:
        print(f"Error en la ruta /chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno al generar la respuesta: {e}")