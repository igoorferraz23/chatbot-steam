import os

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from config import Settings


class ChromaDB:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model=Settings.OLLAMA_EMBED_MODEL, base_url=Settings.OLLAMA_BASE_URL)

        # Garante a existência do diretório
        os.makedirs(Settings.CHROMA_DB_PATH, exist_ok=True)

        self.vector_store = Chroma(
            embedding_function=self.embeddings,
            persist_directory=Settings.CHROMA_DB_PATH,
            collection_name=Settings.CHROMA_DB_NAME,
        )

    @property
    def count(self) -> int:
        return self.vector_store._collection.count()

    def check(self) -> bool:
        """Verifica se existem registros no banco."""

        try:
            total = self.count
            return total != 0
        except Exception as e:
            print(f"Erro ao verificar banco: {e}")
            return False
