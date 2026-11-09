import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from chatbot import Chatbot
from utils.chat_logger import ChatLogger

app = FastAPI(
    title="Steam Assistant RAG API",
    version="1.0.0",
    description="API para recomendação e busca de jogos da Steam usando RAG local",
)


# Entrada
class ChatRequest(BaseModel):
    message: str
    session_id: str = str(uuid.uuid4())


# Saída
class ChatResponse(BaseModel):
    response: str
    session_id: str


@app.get("/")
def health_check():
    """Endpoint simples para verificar se a API está online."""
    return {"status": "ok", "message": "API de recomendação de jogos da Steam está online."}


@app.get("/api/v1/health/db")
def health_check_db():
    """Endpoint para verificar a saúde do banco de dados ChromaDB."""
    from utils.chroma_db import ChromaDB

    chroma_db = ChromaDB()
    if not chroma_db.check():
        return {"status": "error", "message": "Banco de dados ChromaDB não está acessível ou vazio."}
    return {"status": "ok", "message": "Banco de dados ChromaDB acessível."}


@app.get("/api/v1/health/ollama")
def health_check_ollama():
    """Endpoint para verificar a saúde do serviço Ollama."""
    from utils.ollama import Ollama

    ollama = Ollama()
    if not ollama.check():
        return {"status": "error", "message": "Serviço Ollama não está acessível."}
    return {"status": "ok", "message": "Serviço Ollama acessível."}


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest):
    """Endpoint principal para receber mensagens, consultar e retornar a resposta."""

    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Favor preencher a mensagem.")

    # Gerar um logger para o chat
    session_id = payload.session_id if hasattr(payload, "session_id") else None
    logger = ChatLogger(session_id)
    chatbot = Chatbot(logger)
    try:
        # Executa o seu fluxo de processamento
        llm_response = chatbot.process_rag(payload.message)

        # Salva o log da interação
        logger.log_interaction(user_input=payload.message, bot_response=llm_response)

        return ChatResponse(response=llm_response, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno no processamento RAG: {str(e)}")
