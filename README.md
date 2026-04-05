# ChatBot

Este projeto implementa um chatbot completo com backend em Python e frontend em Next.js/React/TypeScript:
- **Backend**: LangGraph para orquestração de pipeline de chat, Ollama local como LLM, FAISS para busca vetorial por categoria, SQLite e/ou URL pública de banco de dados para persistência, FastAPI como API REST, Scraping de documentos e gerenciamento de conteúdo
- **Frontend**: Interface web em Next.js com React e TypeScript, utilizando Tailwind CSS para estilização


## Visão geral do que foi feito

### Backend
1. Criou-se um backend completo em `backend/src` com módulos organizados em:
   - `database.py`: configuração de banco e sessão SQLAlchemy
   - `repositories.py`: abstração de acesso a dados
   - `data_loader.py`: integração de documentos, FAISS e persistência
   - `faiss_manager.py`: gerenciamento de índices vetoriais por categoria
   - `similarity.py`: geração de embeddings e comparação de texto
   - `scraper.py`: scraping web e armazenamento de documentos
   - `llm.py`: integração com Ollama
   - `graph/`: pipeline de chatbot baseado em LangGraph
   - `main.py`: API FastAPI com endpoints para chatbot, documentos e FAISS

2. Implementou-se LangGraph para controlar o fluxo de dados com 4 nós:
   - `retriever_node`: busca documentos relevantes com FAISS
   - `prompt_builder_node`: monta prompt com contexto e histórico
   - `llm_node`: envia prompt ao Ollama e recebe resposta
   - `response_formatter_node`: formata resposta e atualiza histórico

3. Adicionou-se FAISS com suporte a índices por categoria, persistência em disco e busca otimizada. Cada categoria mantém seu próprio índice vetorial.

4. Criou-se persistência de documentos e histórico em banco de dados usando SQLAlchemy com suporte a:
   - SQLite local (`sqlite:///./chatbot.db`)
   - qualquer URL de banco de dados pública via variável de ambiente `DATABASE_URL`

5. Implementou-se endpoints REST para:
   - verificar saúde do serviço
   - calcular similaridade entre textos
   - gerar embeddings
   - adicionar e listar documentos
   - buscar via FAISS
   - limpar índices e documentos
   - conversar com o chatbot

6. Desenvolveu-se testes locais para:
   - validar o FAISS
   - validar a persistência em banco via `test_db_persistence.py`

### Frontend
1. Criou-se um frontend em Next.js com React e TypeScript para interface do usuário.
2. Utiliza Tailwind CSS para estilização responsiva.
3. Estrutura organizada em `app/` com layout e páginas.
4. Pronto para integração com a API do backend.


## Estrutura do projeto

```
ChatBot/
├─ .env.example
├─ README.md
├─ backend/
│  ├─ chatbot.db              # banco SQLite gerado localmente
│  ├─ requirements.txt
│  ├─ init_db.py
│  ├─ test_db_persistence.py
│  ├─ src/
│  │  ├─ main.py
│  │  ├─ database.py
│  │  ├─ repositories.py
│  │  ├─ data_loader.py
│  │  ├─ faiss_manager.py
│  │  ├─ similarity.py
│  │  ├─ scraper.py
│  │  ├─ llm.py
│  │  ├─ test_faiss.py
│  │  └─ graph/
│  │     ├─ builder.py
│  │     ├─ nodes.py
│  │     ├─ state.py
│  │     └─ __init__.py
│  ├─ indices/                # índices FAISS salvos em disco
│  └─ data/
├─ frontend/
│  ├─ package.json
│  ├─ next.config.ts
│  ├─ tsconfig.json
│  ├─ eslint.config.mjs
│  ├─ postcss.config.mjs
│  ├─ tailwind.config.ts      # (se existir)
│  ├─ app/
│  │  ├─ layout.tsx
│  │  ├─ page.tsx
│  │  ├─ globals.css
│  │  └─ app.css
│  ├─ public/
│  └─ README.md
└─ tests/
```


## Dependências

### Backend
Instale as dependências do backend com:

```bash
cd backend
pip install -r requirements.txt
```

## Como executar

### Backend
1. Navegue para a pasta do backend:
   ```bash
   cd backend
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Inicialize o banco de dados (opcional, se necessário):
   ```bash
   python init_db.py
   ```

4. Execute o servidor FastAPI:
   ```bash
   python src/main.py
   ```
   O backend estará disponível em `http://localhost:8000`.

### Frontend
1. Navegue para a pasta do frontend:
   ```bash
   cd frontend
   ```

2. Instale as dependências:
   ```bash
   npm install
   ```

3. Execute o servidor de desenvolvimento:
   ```bash
   npm run dev
   ```
   O frontend estará disponível em `http://localhost:3000`.

### Executando ambos simultaneamente
Para executar o backend e frontend ao mesmo tempo, abra dois terminais:

- Terminal 1 (Backend):
  ```bash
  cd backend
  python src/main.py
  ```

- Terminal 2 (Frontend):
  ```bash
  cd frontend
  npm run dev
  ```

Certifique-se de que o Ollama esteja rodando localmente para o funcionamento completo do chatbot.

### Frontend
Instale as dependências do frontend com:

```bash
cd frontend
npm install
```

As principais bibliotecas usadas são:
- `next` (16.2.2)
- `react` (19.2.4)
- `react-dom` (19.2.4)
- `tailwindcss` (^4)
- `typescript` (^5)
- `requests`


## Configuração do banco de dados

A aplicação lê `DATABASE_URL` do ambiente.

Por padrão, se nenhuma variável estiver configurada, ela usa SQLite local:

```bash
DATABASE_URL=sqlite:///./chatbot.db
```

Para usar um banco público, crie um arquivo `.env` na raiz do projeto com a URL desejada.

Exemplo de `.env`:

```env
DATABASE_URL=postgresql://user:password@host:5432/dbname
```

Ou, para MySQL:

```env
DATABASE_URL=mysql+pymysql://user:password@host:3306/dbname
```

> O SQLite é apenas local e não deve ser usado como banco público. Para acesso a partir de múltiplos computadores, utilize PostgreSQL ou MySQL.


## Inicializando o banco de dados

Para criar as tabelas, execute:

```bash
cd backend
python init_db.py
```

Isso irá criar o banco local `chatbot.db` e as tabelas:
- `sessions`
- `messages`
- `documents`


## Como rodar o backend

```bash
cd backend
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

ou, se preferir, execute diretamente:

```bash
cd backend
python src/main.py
```

A API ficará disponível em:

- `http://localhost:8000`
- documentação automática: `http://localhost:8000/docs`


## Endpoints disponíveis

### Saúde e informação
- `GET /` — mensagem de boas-vindas
- `GET /health` — checa se o backend e embeddings foram carregados

### Similaridade e embeddings
- `POST /similarity` — calcula similaridade entre dois textos
- `POST /search` — busca textos similares em uma lista de textos
- `POST /embedding` — gera embedding para um texto

### Chatbot
- `POST /chat` — executa pipeline LangGraph e retorna resposta, documentos recuperados, histórico e modelo usado

### Documentos
- `POST /documents/add-urls` — faz scraping de URLs e adiciona documentos
- `POST /documents/add-manual` — adiciona documento manualmente
- `GET /documents` — lista todos os documentos salvos
- `GET /documents/count` — conta documentos
- `DELETE /documents` — remove todos os documentos

### FAISS
- `POST /faiss/add` — adiciona documentos a um índice FAISS por categoria
- `POST /faiss/search` — busca por similaridade vetorial em FAISS
- `GET /faiss/stats` — consulta estatísticas dos índices FAISS
- `DELETE /faiss/category/{category}` — limpa índice de categoria específica
- `DELETE /faiss/all` — limpa todos os índices FAISS


## Arquitetura e fluxo

### LangGraph
O pipeline do chatbot está em `backend/src/graph`.
Ele monta um grafo com 4 nós:

1. `retriever_node` — busca documentos relevantes com FAISS e monta contexto
2. `prompt_builder_node` — constrói prompt unindo histórico e contexto
3. `llm_node` — envia prompt ao Ollama e recebe resposta
4. `response_formatter_node` — formata resposta e atualiza histórico

Isso garante separação de responsabilidade e facilita evolução do fluxo.

### FAISS por categoria
Os índices FAISS são mantidos em `backend/indices`.
Cada categoria tem seu próprio índice e arquivo de metadados.

Categorias típicas:
- `manual`
- `scraped`
- `web`
- `chat_history`

### Persistência de documentos e histórico
A persistência usa SQLAlchemy com estas entidades:
- `Document`: documentos adicionados manualmente ou por scraping
- `SessionModel`: sessão de conversa
- `Message`: histórico de perguntas e respostas

O backend salva documentos em banco e mantém também um cache em memória para o retriever.


## Testes criados

- `backend/test_db_persistence.py`
  - valida criação de tabelas
  - valida persistência de documentos
  - valida criação de sessão e mensagens
  - valida recuperação de histórico

- `backend/src/test_faiss.py`
  - valida adição e busca FAISS


## O que foi implementado

### Funcionalidades principais
- Chatbot em Python usando LangGraph
- LLM local via Ollama
- Busca semântica via FAISS
- Persistência de documentos e histórico em banco de dados
- Suporte a banco público via `DATABASE_URL`
- Scraping de conteúdos externos via URLs
- API REST com endpoints completos para administração

### Tecnologias usadas
- FastAPI
- SQLAlchemy
- FAISS
- LangGraph
- Ollama
- Sentence Transformers
- BeautifulSoup
- Requests
- Python Dotenv


## Próximos passos sugeridos

1. criar frontend TypeScript na pasta `frontend/`
2. adicionar autenticação e gerenciamento de sessões mais avançado
3. suportar upload de documentos e parsing de PDFs
4. separar o pipeline de armazenamento (FAISS) do pipeline de chat


## Observações finais

- O arquivo `.env.example` agora está na raiz do projeto.
- Para rodar em outros computadores, basta compartilhar o código e um `.env` com a URL pública do banco.
- Se quiser usar SQLite local para testes, não é necessário definir `DATABASE_URL`.
