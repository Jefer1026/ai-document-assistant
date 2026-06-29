import ollama
import sys

class AgentChat:
    def __init__(self, model="llama3.1"):
        self.model = model
        self.history = [
            {"role": "system", "content": "Eres un asistente amigable y eficiente y das respuestas cortas"}
        ]

    # ESTA ES LA CORRECCIÓN: 'chatear' debe estar al mismo nivel que '__init__'
    def chatear(self):
        print(f"--- Agente iniciado con {self.model} (Escribe 'salir' para terminar) ---")

        while True:
            user_input = input("\n👤 Tú: ")

            if user_input.lower() in ["salir", "exit", "quit"]:
                print("👋 ¡Hasta luego!")
                break
            
            # Guardamos la pregunta del usuario
            self.history.append({"role": "user", "content": user_input})

            print("🤖 IA: ", end="", flush=True)

            try:
                # Usamos stream=True para que se vea profesional
                stream = ollama.chat(
                    model=self.model,
                    messages=self.history,
                    stream=True
                )

                full_response = ""
                # Corregido: 'chunk' en lugar de 'chun' y 'chunck'
                for chunk in stream:
                    content = chunk['message']['content']
                    print(content, end="", flush=True)
                    full_response += content 

                print()

                # Corregido: 'assistant' en lugar de 'assistan'
                self.history.append({"role": "assistant", "content": full_response})

            except Exception as e:
                print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    agent = AgentChat()
    agent.chatear()