from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import ollama
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
import os

app = FastAPI()
client = ollama.Client(host='http://ollama:11434')

# --- CONFIGURACIÓN RAG ---
class AgentRAG:
    def __init__(self, docs_path="/app/documents"):
        self.docs_path = docs_path
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(name="documentos")
        
        # Escaneo automático al iniciar
        self.escanear_documentos_locales()

    def escanear_documentos_locales(self):
        """Busca PDFs en la carpeta montada y los procesa si no existen ya."""
        if not os.path.exists(self.docs_path):
            print(f"Carpeta {self.docs_path} no encontrada.")
            return

        for archivo in os.listdir(self.docs_path):
            if archivo.endswith(".pdf"):
                ruta_completa = os.path.join(self.docs_path, archivo)
                print(f"Procesando archivo automático: {archivo}")
                self.procesar_pdf(ruta_completa)

    def procesar_pdf(self, file_path):
        try:
            reader = PdfReader(file_path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    # Usamos el nombre del archivo + numero de pagina como ID único
                    page_id = f"{os.path.basename(file_path)}_page_{i}"
                    self.collection.add(documents=[text], ids=[page_id])
        except Exception as e:
            print(f"Error procesando {file_path}: {e}")

    def consultar(self, pregunta):
        resultados = self.collection.query(query_texts=[pregunta], n_results=2)
        return "\n".join(resultados['documents'][0]) if resultados['documents'][0] else ""

rag = AgentRAG()

class AgentChat:
    def __init__(self, model="llama3.1"):
        self.model = model
        self.history = [{
            "role": "system", 
            "content": """Eres un asistente eficiente y directo.
            - Si el 'Contexto del documento' es relevante, responde basándote exclusivamente en él.
            - Si el 'Contexto del documento' no es útil o es irrelevante para la pregunta, responde directamente usando tu conocimiento general.
            - REGLA DE ORO: Nunca menciones que estás usando el documento o tu conocimiento general. No des explicaciones sobre tu proceso interno.
            - Sé breve, amable y conciso. No rellenes con texto innecesario."""
        }]

    def responder(self, user_input):
        # 1. Obtenemos contexto solo si es necesario (lógica de umbral)
        contexto = ""
        if len(user_input) >= 20:
            contexto = rag.consultar(user_input)
        
        # 2. Construimos el prompt temporal (no lo guardamos en el historial aún)
        if contexto and len(contexto.strip()) > 0:
            prompt_para_modelo = f"Contexto del documento: {contexto}\n\nPregunta: {user_input}"
        else:
            prompt_para_modelo = user_input
            
        # 3. Guardamos en el historial SOLO la pregunta limpia (sin el contexto pegado)
        # Esto hace que la memoria del chat sea natural y no técnica
        self.history.append({"role": "user", "content": user_input})
        
        # 4. Creamos una copia temporal de los mensajes para la llamada al LLM
        # Esto incluye el System Prompt + Historial + Prompt con contexto actual
        mensajes_temporales = self.history[:-1] + [{"role": "user", "content": prompt_para_modelo}]
        
        # 5. LLAMADA AL LLM
        response = client.chat(model=self.model, messages=mensajes_temporales)
        content = response['message']['content']
        
        # 6. Guardamos solo la respuesta del asistente en el historial
        self.history.append({"role": "assistant", "content": content})
        
        return content
agente = AgentChat()

# --- ENDPOINTS ---
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    respuesta = agente.responder(req.message)
    return {"response": respuesta}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Guardamos temporalmente para procesar
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
    
    rag.procesar_pdf(temp_path)
    os.remove(temp_path) # Limpiamos
    return {"status": "Documento procesado"}