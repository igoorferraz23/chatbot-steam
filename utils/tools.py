from datetime import datetime

import ollama
import pandas as pd


class Tools:
    # Função para baixar o modelo de embeddings do Ollama
    @staticmethod
    def download_ollama_model(modelo: str, log: bool = True):
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

    # Garante a conversão dos valores para int
    @staticmethod
    def safe_int(valor) -> int:
        try:
            v = float(valor)
            if pd.isna(v) or v == float("inf") or v == float("-inf"):
                return 0
            return int(v)
        except (ValueError, TypeError):
            return 0
