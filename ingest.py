import os
import time
from datetime import datetime

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from utils.tools import Tools

# Principais colunas
colunas_chave = [
    "AppID",
    "Name",
    "Genres",
    "Tags",
    "Price",
    "About the game",
    "Developers",
    "Publishers",
    "Metacritic score",
    "Positive",
    "Negative",
    "Categories",
    "Release date",
]

print(f"[{datetime.now().strftime('%H:%M:%S')}] Iniciando carregamento do CSV...")

# 1. Carrega os dados do CSV e filtra apenas as colunas necessárias
df = pd.read_csv("data/games.csv", index_col=False)[colunas_chave]

# Garante a conversão dos IDs para int
df["AppID"] = df["AppID"].fillna(0)

# Preenche os valores nulos com "Não informado"
df = df.fillna("Não informado")


# 2. Cria a lista de documentos
documents = [
    Document(
        page_content=(
            f"Nome: {row['Name']}\n"
            f"Gênero: {row['Genres']}\n"
            f"Tags: {row['Tags']}\n"
            f"Sobre o jogo: {row['About the game']}\n"
            f"Desenvolvedor: {row['Developers']}\n"
            f"Publicadora: {row['Publishers']}\n"
            f"Categorias: {row['Categories']}"
        ),
        metadata={
            "app_id": Tools.safe_int(row["AppID"]),
            "name": str(row["Name"]),
            "price": str(row["Price"]),
            "release_date": str(row["Release date"]),
            "metacritic": str(row["Metacritic score"]),
            "positive_reviews": str(row["Positive"]),
            "negative_reviews": str(row["Negative"]),
        },
    )
    for _, row in df.iterrows()
]

# 3. Download local do modelo de embeddings da Ollama.
Tools.download_ollama_model("nomic-embed-text")

# 4. Inicializa os Embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Garante a existência do diretório
os.makedirs("./db/chroma_db", exist_ok=True)

# 5. Inserção em Lotes (Batching) no ChromaDB
print(f"[{datetime.now().strftime('%H:%M:%S')}] Vetorizando {len(documents)} documentos...")

batch_size = 500  # Processa de 500 em 500 para evitar sobrecarga
total_batches = (len(documents) - 1) // batch_size + 1

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="./db/chroma_db",
    collection_name="steam_games",
)

for i in range(0, len(documents), batch_size):
    batch = documents[i : i + batch_size]
    batch_number = i // batch_size + 1

    success = False
    tentativas = 0

    # Tenta enviar o lote até 3 vezes caso o Ollama oscile
    while not success and tentativas < 3:
        try:
            vector_store.add_documents(batch)
            success = True
            print(f" -> Lote {batch_number}/{total_batches} concluído.")
        except Exception as e:
            tentativas += 1
            print(f" [!] Erro no lote {batch_number} (Tentativa {tentativas}/3): {e}")
            time.sleep(5)  # Espera 5 segundos para o Ollama se recuperar antes de tentar de novo

    if not success:
        print(f" [X] Falha crítica no lote {batch_number}. Pulando lote para continuar o processo.")

print(f"[{datetime.now().strftime('%H:%M:%S')}] Processo finalizado com sucesso em './db/chroma_db'!")
