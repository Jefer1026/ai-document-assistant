from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import ollama
import chromadb
from pypdf import PdfReader
import os
import httpx

app = FastAPI()

# Define la dirección explícitamente
ollama_url = os.getenv("OLLAMA_BASE_URL", "http://10.0.1.53:11434")

# Inicializa el cliente forzando el host
client = ollama.Client(host=ollama_url)


# --- CONFIGURACIÓN RAG ---
class AgentRAG:
    def __init__(self, docs_path="/app/documents"):
        self.docs_path = docs_path
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(name="documentos")
        
        # Escaneo automático al iniciar
        self.scan_local_documents()

    def scan_local_documents(self):
        """Busca PDFs en la carpeta montada y los procesa si no existen ya."""
        if not os.path.exists(self.docs_path):
            print(f"Carpeta {self.docs_path} no encontrada.")
            return

        for file in os.listdir(self.docs_path):
            if file.endswith(".pdf"):
                ruta_completa = os.path.join(self.docs_path, file)
                print(f"Procesando archivo automático: {file}")
                self.process_pdf(ruta_completa)

    def process_pdf(self, file_path):
        file_name = os.path.basename(file_path) # Ej: "politica_privacidad.pdf"
        try:
            reader = PdfReader(file_path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    page_id = f"{file_name}_page_{i}"
                    # Guardamos el nombre del archivo en los metadatos
                    self.collection.add(
                        documents=[text], 
                        ids=[page_id],
                        metadatas=[{"source": file_name}] 
                    )
        except Exception as e:
            print(f"Error procesando {file_path}: {e}")

    def query_data(self, pregunta):
        # 1. Realizamos la búsqueda
        results = self.collection.query(query_texts=[pregunta], n_results=3)
        
        # 2. Validación robusta: 
        # Verificamos que 'documents' exista, tenga elementos y que 'metadatas' no sea None
        docs = results.get('documents', [[]])[0]
        metas = results.get('metadatas', [[]])[0]
        
        if not docs:
            return ""

        contexto_final = ""
        for i in range(len(docs)):
            doc_text = docs[i]
            # Si metadatas es None o no tiene el índice, manejamos el error con un valor por defecto
            meta = metas[i] if metas and i < len(metas) else {}
            # Si meta es None, lo convertimos a diccionario vacío
            meta = meta if meta is not None else {}
            
            fuente = meta.get('source', 'desconocido')
            contexto_final += f"\n[Fuente: {fuente}]\n{doc_text}"
        
        return contexto_final

rag = AgentRAG()

class AgentChat:
    def __init__(self, model="llama3.1"):
        self.model = model
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://10.0.1.53:11434")
        
        self.history = [{
            "role": "system", 
            "content": """Eres Jeferson Oyola Garcia. Tu objetivo es responder preguntas de forma inmediata y precisa.
            - PRIORIDAD: Usa el 'Contexto del documento' solo si la respuesta está allí.
            - Si la información no está en el contexto, responde usando tu conocimiento general de forma directa.
            - PROHIBICIONES: Nunca menciones que estás usando el documento, ni que buscas en fuentes externas, ni que la información no está en tus archivos. 
            - FORMATO: Máximo 2 oraciones. Sé breve, profesional y evita introducciones como 'Una respuesta generalizada sería' o 'El documento dice'."""
        }]

    def answer(self, user_input):
        # 1. Obtenemos contexto
        contexto = rag.query_data(user_input)
        prompt_para_modelo = f"Contexto: {contexto}\n\nPregunta: {user_input}" if contexto else user_input
        
        self.history.append({"role": "user", "content": user_input})
        mensajes_temporales = self.history[:-1] + [{"role": "user", "content": prompt_para_modelo}]
               
        
        payload = {
            "model": self.model,
            "messages": mensajes_temporales,
            "stream": False
        }
        
        # Petición HTTP pura con httpx
        with httpx.Client(timeout=60.0) as http_client:
            response = http_client.post(f"{self.ollama_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            content = data['message']['content']
        
        self.history.append({"role": "assistant", "content": content})
        return content
agent = AgentChat()

# --- ENDPOINTS ---
class ChatRequest(BaseModel):
    message: str

import traceback

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        respuesta = agent.answer(req.message)
        return {"response": respuesta}
    except Exception as e:
        # Esto imprimirá la traza completa (traceback) en tus logs
        import traceback
        traceback.print_exc() 
        return {"error": str(e), "detalle": "Revisa los logs del contenedor para ver el traceback completo"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Guardamos temporalmente para procesar
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
    
    rag.process_pdf(temp_path)
    os.remove(temp_path) # Limpiamos
    return {"status": "Documento procesado"}