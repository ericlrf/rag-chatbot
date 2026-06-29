# V-LAB RAG Chatbot

Projeto de portfólio para entrevista: chatbot com **Retrieval-Augmented Generation (RAG)**, API em **FastAPI**, embeddings da OpenAI, banco vetorial local com **ChromaDB** e interface web simples para demonstração.

O objetivo é demonstrar domínio prático de:

- LLMs;
- prompt engineering;
- RAG;
- chatbots e sistemas conversacionais;
- APIs com FastAPI;
- ingestão de documentos;
- busca vetorial;
- boas práticas mínimas de segurança, configuração e documentação.

---

## 1. Arquitetura

Fluxo principal:

```text
Usuário -> Frontend/API -> FastAPI -> Busca vetorial no ChromaDB -> Contexto recuperado -> LLM -> Resposta com fontes
```

Pipeline de ingestão:

```text
Documentos PDF/TXT/MD/DOCX -> extração de texto -> chunks -> embeddings -> ChromaDB persistente
```

---

## 2. Estrutura do projeto

```text
rag-chatbot-vlab/
├── app/
│   ├── main.py                  # API FastAPI
│   ├── config.py                # Configurações via .env
│   ├── schemas.py               # Schemas Pydantic
│   ├── security.py              # API key opcional
│   ├── core/
│   │   ├── chunking.py          # Divisão de textos em chunks
│   │   ├── document_loader.py   # Leitura de PDF, DOCX, TXT e MD
│   │   ├── feedback.py          # Registro de feedbacks
│   │   ├── openai_client.py     # Embeddings e geração com OpenAI
│   │   ├── rag.py               # Montagem do prompt RAG
│   │   └── vectorstore.py       # ChromaDB persistente
│   └── utils/
│       └── logging.py
├── data/documents/              # Coloque seus documentos aqui
├── frontend/index.html          # Interface web simples
├── scripts/
│   ├── ingest.py                # Script de indexação
│   └── clear_index.py           # Limpa coleção vetorial
├── storage/chroma/              # Banco vetorial local
├── tests/
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 3. Como executar localmente

### 3.1. Criar ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

No Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.2. Configurar variáveis de ambiente

```bash
cp .env.example .env
```

Edite o arquivo `.env` e preencha:

```env
OPENAI_API_KEY=sua_chave_aqui
```

Você também pode trocar o modelo de chat e o modelo de embeddings:

```env
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

---

## 4. Indexar documentos

Coloque arquivos em:

```text
data/documents/
```

Formatos suportados:

- `.pdf`
- `.txt`
- `.md`
- `.docx`

Execute:

```bash
python -m scripts.ingest --path data/documents --reset
```

Saída esperada:

```text
Indexação concluída
Arquivos indexados: 1
Chunks indexados: 3
Coleção: vlab_documents
```

---

## 5. Rodar a API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Acesse:

```text
http://localhost:8000
```

Documentação interativa da API:

```text
http://localhost:8000/docs
```

Health check:

```text
http://localhost:8000/api/health
```

---

## 6. Exemplos de uso via curl

### 6.1. Ingestão pela API

```bash
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"directory":"data/documents", "reset_collection": true}'
```

### 6.2. Enviar documento pela API

```bash
curl -X POST "http://localhost:8000/api/upload?ingest_after_upload=true" \
  -F "file=@data/documents/vaga_vlab.md"
```

### 6.3. Perguntar ao chatbot

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"Quais são os requisitos técnicos da vaga?", "top_k": 5}'
```

Resposta esperada:

```json
{
  "answer": "...",
  "sources": [
    {
      "source_id": 1,
      "file_name": "vaga_vlab.md",
      "path": "data/documents/vaga_vlab.md",
      "page": null,
      "chunk_index": 0,
      "score": 0.23,
      "excerpt": "..."
    }
  ],
  "model": "gpt-4.1-mini",
  "used_top_k": 5
}
```

---

## 7. Rodar com Docker

```bash
cp .env.example .env
# Edite OPENAI_API_KEY no .env

docker compose up --build
```

Depois acesse:

```text
http://localhost:8000
```

---

## 8. Testes

```bash
pytest -q
```

---

## 9. Segurança e responsabilidade

Este projeto inclui algumas práticas iniciais:

- Respostas devem se basear apenas nos documentos recuperados.
- O prompt orienta o modelo a dizer quando não encontrou a informação.
- As respostas incluem fontes recuperadas.
- Há autenticação opcional por `APP_API_KEY`.
- Feedbacks podem ser registrados para avaliação posterior.

Para uso em produção, recomenda-se adicionar:

- autenticação robusta;
- controle de usuários e permissões;
- proteção contra prompt injection;
- validação de arquivos enviados;
- observabilidade;
- rate limiting;
- avaliação sistemática de qualidade;
- monitoramento de custo;
- política de retenção de dados.

---

## 10. Como apresentar na entrevista

Sugestão de fala:

> Desenvolvi um protótipo de chatbot com RAG para consulta a documentos. O sistema possui uma API em FastAPI, pipeline de ingestão para PDF, DOCX, Markdown e TXT, geração de embeddings com a OpenAI, persistência vetorial com ChromaDB e uma interface web simples. A arquitetura separa extração, chunking, embeddings, recuperação e geração, o que facilita manutenção e evolução. Também inclui fontes nas respostas e uma regra explícita para evitar respostas inventadas quando o contexto não contém a informação.

Pontos técnicos para destacar:

- por que usar RAG em vez de apenas prompt longo;
- como os chunks são criados;
- como os embeddings são armazenados;
- como a pergunta vira embedding para busca semântica;
- como o contexto recuperado entra no prompt;
- como as fontes são exibidas;
- limitações do protótipo e melhorias futuras.

---

## 11. Melhorias futuras

- Upload de documentos via endpoint;
- login de usuários;
- suporte a múltiplas coleções;
- avaliação automática de respostas;
- reranking;
- filtros por tipo de documento, data ou setor;
- integração com LangChain ou LlamaIndex;
- suporte a modelos locais, como LLaMA ou Mistral;
- streaming de respostas;
- painel de métricas;
- deploy em nuvem.
