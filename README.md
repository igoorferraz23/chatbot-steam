# 🎮 Steam Assistant RAG

Assistente inteligente em Python para buscar e recomendar jogos da Steam utilizando **RAG**, **ChromaDB** e **IA local**.

## 🛠️ Tecnologias

* Python 3.12+
* LangChain
* ChromaDB
* Ollama

  * `gemma3` — LLM
  * `nomic-embed-text` — Embeddings
* Pandas

## 🚀 Como Rodar

### 1. Ollama

Com o Ollama instalado e rodando:

```bash
ollama pull gemma3
ollama pull nomic-embed-text
```

### 2. Ambiente Python

```bash
python -m venv .venv
```

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

### 3. Gerar o banco vetorial

Execute:

```bash
python ingest.py
```

### 4. Iniciar o assistente

```bash
python chatbot.py
```

## 💬 Comandos

* `sair` — Encerra o assistente.

## 🧠 Arquitetura

```text
games.csv → Pandas → Embeddings → ChromaDB
                                      ↓
Usuário → Retriever → Gemma 3 → Resposta
```

O projeto utiliza RAG para recuperar informações relevantes da base de jogos antes de gerar cada resposta.
