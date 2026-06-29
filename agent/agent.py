from fastapi import FastAPI
from pydantic import BaseModel
import ollama # Asegúrate de que esto sea la librería oficial

app = FastAPI()

# --- AQUÍ ESTÁ EL CAMBIO ---
# Configuramos el cliente para que busque el servicio 'ollama' en la red
client = ollama.Client(host='http://ollama:11434')

class AgentChat:
    def __init__(self, model="llama3.1"):
        self.model = model
        self.history = [
            {"role": "system", "content": "Eres un asistente de IA útil y directo. Responde a las preguntas del usuario de forma breve y precisa."}
        ]

    def responder(self, user_input):
        self.history.append({"role": "user", "content": user_input})
        
        # Usamos el cliente configurado
        response = client.chat(model=self.model, messages=self.history)
        
        content = response['message']['content']
        self.history.append({"role": "assistant", "content": content})
        return content
# ---------------------------

agente = AgentChat()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    respuesta = agente.responder(req.message)
    return {"response": respuesta}