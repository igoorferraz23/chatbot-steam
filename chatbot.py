from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

from utils.chat_logger import ChatLogger
from utils.chroma_db import ChromaDB
from utils.ollama import Ollama

from config import Settings

# Configuração do LLM e Logger
llm = ChatOllama(model=Settings.LLM_MODEL, temperature=Settings.LLM_TEMPERATURE)

# Configuração do Embeddings
ollama = Ollama()

# Configuração do ChromaDB
chroma_db = ChromaDB()
vector_store = chroma_db.vector_store


class Chatbot:
    chat_history = []

    def __init__(self, logger: "ChatLogger"):
        self.chat_history = logger.restore_history() if logger else []

    @staticmethod
    def process_context(input):
        """Processa o contexto dinâmico gerado com os documentos relevantes."""

        # Busca Vetorial no ChromaDB
        relevant_docs = vector_store.similarity_search(input, k=4)

        # Formata o contexto recuperado com os textos e metadados
        dinamic_contexts = ""
        for doc in relevant_docs:
            meta = doc.metadata
            dinamic_contexts += (
                f"Preço: R${meta.get('price', 'N/A')} | "
                f"Metacritic: {meta.get('metacritic', 'N/A')} | "
                f"Avaliações (+/{meta.get('positive_reviews')}, -/{meta.get('negative_reviews')})\n"
                f"{doc.page_content}\n"
                f"{'-' * 40}\n"
            )

        return dinamic_contexts

    @staticmethod
    def process_rag(input):
        """Processa a pergunta do usuário, gera o contexto e retorna a resposta do LLM."""

        # Processa o contexto dinâmico
        dynamic_context = Chatbot.process_context(input)

        # Prompt dinâmico gerado com o contexto recuperado
        system_prompt = f"\
            Você é um especialista e assistente de banco de dados de jogos apenas com base no contexto abaixo e responde em português brasileiro de forma resumida.\
            {dynamic_context}\
        "

        # --- MONTAGEM E ENVIO DAS MENSAGENS ---
        sys_msg = SystemMessage(content=system_prompt)
        usr_msg = HumanMessage(content=input)

        # Combina SystemPrompt (com contexto atualizado), histórico limitado e a pergunta atual
        all_messages = [sys_msg] + Chatbot.chat_history[-6:] + [usr_msg]

        # Filtro de segurança para objetos de mensagem válidos
        payload_messages = [msg for msg in all_messages if isinstance(msg, BaseMessage)]

        complete_answer = ""

        # Streaming da resposta do Ollama
        for chunk in llm.stream(payload_messages):
            content_answer = chunk.content
            complete_answer += str(content_answer)

        # Atualiza o histórico e logger
        Chatbot.chat_history.append(usr_msg)
        Chatbot.chat_history.append(AIMessage(content=complete_answer))

        return complete_answer
