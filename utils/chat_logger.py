import json
import os
from datetime import datetime

from config import Settings


class ChatLogger:
    def __init__(self, session_id: str = None, source: str = "chats"):
        """Define o caminho do arquivo de log."""
        self.log_path = Settings.LOG_PATH

        # Garante que a pasta chats/ existe
        os.makedirs(self.log_path, exist_ok=True)

        # Define o caminho do arquivo JSON baseado na sessão
        self.session_id = session_id
        self.filepath = os.path.join(self.log_path, f"{source}_{self.session_id}.json")

    def log_interaction(self, user_input: str, bot_response: str):
        """Salva uma nova interação dentro do histórico da sessão."""

        interaction = {"timestamp": datetime.now().isoformat(), "user": user_input, "bot": bot_response}

        # Carrega o histórico existente do arquivo (se houver)
        history = []
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except json.JSONDecodeError:
                history = []

        # Adiciona a nova interação
        history.append(interaction)

        # Salva o arquivo atualizado
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=4)

    def restore_history(self):
        """Restaura o histórico de chat da sessão a partir do arquivo JSON."""

        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    history = json.load(f)
                    return history
            except json.JSONDecodeError:
                return []
        else:
            return []
