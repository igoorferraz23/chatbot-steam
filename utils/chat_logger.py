import json
import os
from datetime import datetime


class ChatLogger:
    def __init__(self, folder="chats"):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file = os.path.join(folder, f"chat_{timestamp}.json")

        self.messages = []

    def add_message(self, role, content):
        self.messages.append(
            {"role": role, "content": content, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        )

        with open(self.file, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, indent=4, ensure_ascii=False)
