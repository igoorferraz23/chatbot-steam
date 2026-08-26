from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

from utils.chat_logger import ChatLogger
from utils.tools import Tools

# Configuração do LLM e Logger
llm = ChatOllama(model="gemma3", temperature=0.3)

# Configuração do Embeddings
Tools.download_ollama_model("nomic-embed-text", log=False)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Configuração do ChromaDB
vector_store = Chroma(
    persist_directory="./db/chroma_db",
    embedding_function=embeddings,
    collection_name="steam_games",
)


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
            Você é um especialista e assistente de banco de dados de jogos com base no contexto abaixo.\
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
