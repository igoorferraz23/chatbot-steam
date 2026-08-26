# 🎮 Steam Assistant RAG API

Assistente inteligente em Python para buscar e recomendar jogos da Steam usando **IA local**, agora disponível através de uma **API REST com FastAPI**.

O projeto utiliza **RAG (Retrieval-Augmented Generation)** para consultar jogos armazenados no **ChromaDB** e gerar respostas contextualizadas via **Ollama**, com suporte a histórico de conversas e sem custos de API.

## 🛠️ Tecnologias

* **Python 3.12+**
* **FastAPI + Uvicorn** — API REST e servidor ASGI
* **LangChain** — orquestração do fluxo RAG
* **ChromaDB** — banco de dados vetorial
* **Ollama** — `gemma3` para geração e `nomic-embed-text` para embeddings
* **Pandas** — tratamento dos dados

## 🚀 Como executar

### 1. Instalar os modelos do Ollama

Certifique-se de que o Ollama está instalado e em execução:

```bash
ollama pull gemma3
ollama pull nomic-embed-text
```

### 2. Configurar o ambiente Python

Crie e ative um ambiente virtual:

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

Coloque o arquivo `games.csv` dentro da pasta `data/` e execute:

```bash
python ingest.py
```

### 4. Iniciar a API

```bash
uvicorn api:app --reload --port 8000
```

## 🌐 API e documentação

Após iniciar o servidor, acesse a documentação interativa do FastAPI:

**Swagger UI:** `http://127.0.0.1:8000/docs`

### Exemplo — `POST /api/v1/chat`

```json
{
  "message": "Me recomende jogos de RPG com boa história",
  "session_id": "opcional-id-da-sessao"
}
```

O `session_id` permite manter o histórico de uma conversa.
