from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import ollama
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
import os

app = FastAPI()
client = ollama.Client(host='http://10.0.1.53:11434')

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
        self.history = [{
            "role": "system", 
            "content": """Eres Jeferson Oyola Garcia. Tu objetivo es responder preguntas de forma inmediata y precisa.
            - PRIORIDAD: Usa el 'Contexto del documento' solo si la respuesta está allí.
            - Si la información no está en el contexto, responde usando tu conocimiento general de forma directa.
            - PROHIBICIONES: Nunca menciones que estás usando el documento, ni que buscas en fuentes externas, ni que la información no está en tus archivos. 
            - FORMATO: Máximo 2 oraciones. Sé breve, profesional y evita introducciones como 'Una respuesta generalizada sería' o 'El documento dice'."""
        }]

    def answer(self, user_input):
        # 1. Obtenemos contexto solo si es necesario (lógica de umbral)
        contexto = ""
        if len(user_input) >= 20:
            contexto = rag.query_data(user_input)
        
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
agent = AgentChat()

# --- ENDPOINTS ---
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    respuesta = agent.answer(req.message)
    return {"response": respuesta}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Guardamos temporalmente para procesar
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
    
    rag.process_pdf(temp_path)
    os.remove(temp_path) # Limpiamos
    return {"status": "Documento procesado"}