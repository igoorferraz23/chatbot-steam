import sys
import time
from datetime import datetime

import pandas as pd
from langchain_core.documents import Document

from config import Settings
from utils.chroma_db import ChromaDB
from utils.ollama import Ollama
from utils.tools import Tools

# Verificações iniciais antes de iniciar o processo de ingestão
print(f"[{datetime.now().strftime('%H:%M:%S')}] Iniciando verificações iniciais...")

check_db = ChromaDB()
if check_db.check():
    sys.exit(
        f"[{datetime.now().strftime('%H:%M:%S')}] O banco de dados já contém registros. Abortando o processo para evitar duplicação."
    )

ollama = Ollama()
if not ollama.check():
    sys.exit(f"[{datetime.now().strftime('%H:%M:%S')}] O Ollama não está instalado ou acessível. Abortando o processo.")

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
df = pd.read_csv(Settings.CSV_PATH, index_col=False)[colunas_chave]

if df.empty:
    sys.exit(
        f"[{datetime.now().strftime('%H:%M:%S')}] O CSV está vazio ou não contém as colunas esperadas. Abortando o processo."
    )

## Garante a conversão dos IDs para int
df["AppID"] = df["AppID"].fillna(0)

## Preenche os valores nulos com "Não informado"
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

# 5. Inserção em Lotes (Batching) no ChromaDB
print(f"[{datetime.now().strftime('%H:%M:%S')}] Vetorizando {len(documents)} documentos...")

batch_size = Settings.INGEST_BATCH_SIZE  # Para evitar sobrecarga
total_batches = (len(documents) - 1) // batch_size + 1

vector_store = check_db.vector_store

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

print(
    f"[{datetime.now().strftime('%H:%M:%S')}] Processo finalizado com sucesso. Total de documentos vetorizados: {len(documents)}."
)
