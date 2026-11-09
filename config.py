class Settings:
    """Classe para armazenar as configurações do projeto."""

    # Configuração do Ollama
    OLLAMA_EMBED_MODEL = "nomic-embed-text"

    # Configuração do ChromaDB
    CHROMA_DB_PATH = "./chroma_db"
    CHROMA_DB_NAME = "steam_games"

    # Configuração do LLM
    LLM_MODEL = "gemma3"
    LLM_TEMPERATURE = 0.3

    # Configuração do Logger
    LOG_PATH = "./logs"

    # Configuração do CSV
    CSV_PATH = "./data/games.csv"

    # Configuração do Ingest
    INGEST_BATCH_SIZE = 500
