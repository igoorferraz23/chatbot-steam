import ollama
from datetime import datetime
from config import Settings


class Ollama:
    def __init__(self):
        # Download local do modelo de embeddings da Ollama.
        self.download_ollama_model(Settings.OLLAMA_EMBED_MODEL, log=True)

    @staticmethod
    def download_ollama_model(modelo: str, log: bool = True):
        """Função para baixar o modelo de embeddings do Ollama"""
        try:
            modelos_locais = [m.get("model", "") for m in ollama.list().get("models", [])]
            if not any(modelo in m for m in modelos_locais):
                if log:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Baixando modelo '{modelo}'...")
                ollama.pull(modelo)
                if log:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Modelo '{modelo}' instalado com sucesso!")
            else:
                if log:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Modelo '{modelo}' já está disponível.")
        except Exception as e:
            print(f"Erro ao baixar o modelo: {e}")

    @staticmethod
    def check():
        """Verifica se o Ollama está instalado e acessível."""
        try:
            ollama.list()
            return True
        except Exception as e:
            print(f"Erro ao verificar o Ollama: {e}")
            return False
