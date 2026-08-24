from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

from utils.chat_logger import ChatLogger
from utils.tools import Tools

# 1. Conecta ao ChromaDB e ao modelo de Embeddings

# Download local do modelo de embeddings da Ollama.
Tools.download_ollama_model("nomic-embed-text", log=False)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(
    persist_directory="./db/chroma_db",
    embedding_function=embeddings,
    collection_name="steam_games",
)

# 2. Configuração do LLM e Logger
llm = ChatOllama(model="gemma3", temperature=0.3)
logger = ChatLogger()

chat_history = []

print("Assistente de Games pronto! Digite 'sair' para encerrar.\n")

# 3. Loop principal
while True:
    pergunta = input("Você: ").strip()

    if not pergunta:
        continue

    if pergunta.lower() == "sair":
        break

    # Busca Vetorial no ChromaDB
    docs_relevantes = vector_store.similarity_search(pergunta, k=4)

    # Formata o contexto recuperado com os textos e metadados
    contexto_dinamico = ""
    for doc in docs_relevantes:
        meta = doc.metadata
        contexto_dinamico += (
            f"Preço: R${meta.get('price', 'N/A')} | "
            f"Metacritic: {meta.get('metacritic', 'N/A')} | "
            f"Avaliações (+/{meta.get('positive_reviews')}, -/{meta.get('negative_reviews')})\n"
            f"{doc.page_content}\n"
            f"{'-' * 40}\n"
        )

    # Prompt dinâmico gerado com o contexto recuperado
    system_prompt = f"""Você é um especialista e assistente de banco de dados de jogos com base no contexto abaixo."
---
DADOS RECUPERADOS DO BANCO DE DADOS:
{contexto_dinamico}
---"""

    # --- MONTAGEM E ENVIO DAS MENSAGENS ---
    sys_msg = SystemMessage(content=system_prompt)
    usr_msg = HumanMessage(content=pergunta)

    # Combina SystemPrompt (com contexto atualizado), histórico limitado e a pergunta atual
    all_messages = [sys_msg] + chat_history[-6:] + [usr_msg]

    # Filtro de segurança para objetos de mensagem válidos
    payload_messages = [msg for msg in all_messages if isinstance(msg, BaseMessage)]

    logger.add_message("user", pergunta)

    print("\nBot: ", end="")
    complete_answer = ""

    # Streaming da resposta do Ollama
    for chunk in llm.stream(payload_messages):
        content_answer = chunk.content
        complete_answer += content_answer
        print(content_answer, end="", flush=True)

    print("\n")

    # Atualiza o histórico e logger
    chat_history.append(usr_msg)
    chat_history.append(AIMessage(content=complete_answer))
    logger.add_message("assistant", complete_answer)
