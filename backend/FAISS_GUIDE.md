# FAISS - Busca Vetorial com Persistência

## Visão Geral

FAISS (Facebook AI Similarity Search) foi integrado ao chatbot para fornecer:

1. **Busca rápida e escalável** - Busca de similaridade em tempo real mesmo com milhares de documentos
2. **Persistência por categoria** - Índices salvos em disco organizados por categoria
3. **Gerenciamento em memória** - Documentos mantidos em memória para rápido acesso
4. **Integração automática** - Usa o retriever do LangGraph automaticamente

## Arquitetura

### Componentes Principais

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Endpoints                         │
│  (/faiss/add, /faiss/search, /faiss/stats, etc.)            │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   data_loader.py                             │
│  (Interface de alto nível para FAISS)                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                 faiss_manager.py                             │
│  ┌──────────────────────────────────────────────────┐       │
│  │ FAISSManager (Singleton)                          │       │
│  │  - Gerencia múltiplas categorias                  │       │
│  │  - Sincroniza com disco                           │       │
│  │  - Fornece buscas rápidas                         │       │
│  │                                                   │       │
│  │  indices: Dict[categoria] → FAISSIndex            │       │
│  └──────────────────────────────────────────────────┘       │
│                                                             │
│  ┌──────────────────────────────────────────────────┐       │
│  │ FAISSIndex (Por categoria)                        │       │
│  │  - index: Índice FAISS                            │       │
│  │  - document_map: Mapeamento documento/ID          │       │
│  │  - search(): Busca rápida L2                       │       │
│  │  - add_documents(): Adiciona com embeddings       │       │
│  └──────────────────────────────────────────────────┘       │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
┌───────▼──────────┐    ┌────────────▼─────────┐
│  similarity.py   │    │  Disco: /indices/    │
│  (Embeddings)    │    │  *.index (FAISS)     │
│                  │    │  *_metadata.pkl      │
│                  │    │  *_info.json         │
└──────────────────┘    └──────────────────────┘
```

## Categorias

Cada **categoria** é um índice FAISS separado. Exemplos:

- `"python"` - Documentos sobre Python
- `"frameworks"` - Documentos sobre frameworks (FastAPI, Django, etc)
- `"ai_ml"` - Documentos sobre IA/ML
- `"scraped_urls"` - Documentos obtidos por web scraping
- `"chat_history"` - Respostas passadas do chatbot

## Como Usar

### 1. Adicionar Documentos (Endpoint)

```bash
curl -X POST "http://localhost:8000/faiss/add" \
  -H "Content-Type: application/json" \
  -d '{
    "category": "python",
    "documents": [
      {
        "title": "O que é Python?",
        "content": "Python é uma linguagem...",
        "source": "manual"
      }
    ]
  }'
```

**Resposta:**
```json
{
  "success": true,
  "message": "Adicionados 1 documentos",
  "category": "python",
  "added": 1,
  "total": 5
}
```

### 2. Buscar Documentos (Endpoint)

**Busca em uma categoria específica:**
```bash
curl -X POST "http://localhost:8000/faiss/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "programação orientada a objetos",
    "category": "python",
    "top_k": 3
  }'
```

**Busca em todas as categorias:**
```bash
curl -X POST "http://localhost:8000/faiss/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "velocidade e performance",
    "top_k": 5
  }'
```

**Resposta:**
```json
{
  "query": "programação orientada a objetos",
  "total_results": 2,
  "results": [
    {
      "document": {
        "title": "Características do Python",
        "content": "Python é conhecida pela...",
        "source": "manual"
      },
      "similarity": 0.87,
      "category": "python",
      "rank": 1
    }
  ]
}
```

### 3. Obter Estatísticas (Endpoint)

```bash
curl -X GET "http://localhost:8000/faiss/stats"
```

**Resposta:**
```json
{
  "total_categories": 3,
  "total_documents": 9,
  "categories": {
    "python": {
      "category": "python",
      "total_documents": 3,
      "index_size": 3,
      "embedding_dim": 384,
      "dirty": false
    },
    "frameworks": {
      "category": "frameworks",
      "total_documents": 3,
      "index_size": 3,
      "embedding_dim": 384,
      "dirty": false
    }
  }
}
```

### 4. Limpar Categoria (Endpoint)

```bash
curl -X DELETE "http://localhost:8000/faiss/category/python"
```

### 5. Limpar Tudo (Endpoint)

```bash
curl -X DELETE "http://localhost:8000/faiss/all"
```

## Uso em Código Python

### Adicionar Documentos

```python
from data_loader import add_documents_to_faiss

docs = [
    {
        "title": "Python Basics",
        "content": "Python é uma linguagem...",
        "source": "manual"
    }
]

result = add_documents_to_faiss("python", docs)
print(f"Adicionados: {result['added']}, Total: {result['total']}")
```

### Buscar Documentos

```python
from data_loader import search_with_faiss

# Busca em uma categoria
results = search_with_faiss(
    query="orientado a objeto",
    category="python",
    top_k=5
)

# Busca em todas as categorias
results = search_with_faiss(
    query="velocidade",
    top_k=10
)

# Processar resultados
for result in results:
    print(f"{result['document']['title']}")
    print(f"Similaridade: {result['similarity']:.2%}")
    print(f"Categoria: {result['category']}")
```

### Obter Estatísticas

```python
from data_loader import get_faiss_statistics

stats = get_faiss_statistics()
print(f"Total de categorias: {stats['total_categories']}")
print(f"Total de documentos: {stats['total_documents']}")
```

### Limpar Índices

```python
from data_loader import clear_faiss_category, clear_all_faiss

# Limpar uma categoria
clear_faiss_category("python")

# Limpar tudo
clear_all_faiss()
```

## Integração com LangGraph

O retriever do LangGraph usa automaticamente FAISS:

```python
# em graph/nodes.py
def retriever_node(self, state: ChatState) -> ChatState:
    # Busca automática com FAISS
    retrieved_docs = search_with_faiss(
        query=state.user_query,
        top_k=self.config.max_retrieved_docs
    )
    state.retrieved_documents = retrieved_docs
    return state
```

## Persistência em Disco

Os índices são automaticamente salvos em `backend/indices/`:

```
backend/
└── indices/
    ├── python.index          # Índice FAISS binário
    ├── python_metadata.pkl   # Mapeamento documento/ID
    ├── python_info.json      # Info legível
    ├── frameworks.index
    ├── frameworks_metadata.pkl
    ├── frameworks_info.json
    └── ...
```

**Carregamento automático:** Ao iniciar, o FAISSManager carrega todos os índices salvos.

## Performance

### Complexidade

- **Adição de documentos:** O(n) onde n = número de documentos
- **Busca:** O(log n) com FAISS (vs O(n) com busca linear)
- **Memória:** ~1.5KB por embedding (384 dimensões × 4 bytes float32)

### Benchmarks (Valores estimados)

| Operação | 100 docs | 10K docs | 100K docs |
|----------|----------|----------|-----------|
| Adição | <10ms | <100ms | <500ms |
| Busca | <1ms | <5ms | <50ms |
| Salvar índice | <10ms | <100ms | <500ms |

## Boas Práticas

1. **Categorias semânticas**: Use categorias que sejam semanticamente coerentes
   - ✓ Bom: "python_tutorials", "api_documentation", "blog_posts"
   - ✗ Ruim: "doc1", "doc2", "doc3"

2. **Tamanho de documentos**: Mantenha documentos com tamanho consistente
   - Muito pequenos (<10 palavras) podem ter embeddings ruidosos
   - Muito grandes (>5000 palavras) devem ser divididos em chunks

3. **Top-K**: Use `top_k` apropriado para seu caso
   - Retriever: 5-10
   - Reranking: 20-50

4. **Sincronização**: Use `save_all_faiss()` periodicamente se modificar índices dinamicamente

5. **Categorias**: Reutilize categorias para documentos similares

## Exemplos de Uso

### Exemplo 1: Adicionar URLs ao FAISS

```python
# Primeiro, fazer scraping (como antes)
result = add_urls(["https://example.com"])

# Depois, adicionar ao FAISS
docs = get_all_documents()
add_documents_to_faiss("scraped_urls", docs)
```

### Exemplo 2: Categorizar por fonte

```python
# Separar documentos por origem
manual_docs = [d for d in docs if d['source'] == 'manual']
scraped_docs = [d for d in docs if d['source'].startswith('https')]

add_documents_to_faiss("manual", manual_docs)
add_documents_to_faiss("scraped", scraped_docs)
```

### Exemplo 3: Chat com histórico em FAISS

```python
# Guardar respostas do chat como histórico
response = {"title": f"Chat #{msg_id}", "content": response_text}
add_documents_to_faiss("chat_history", [response])

# Depois, buscar contexto de conversas passadas
history_context = search_with_faiss("topic", category="chat_history")
```

## Troubleshooting

### Problema: "Índice não encontrado"
**Solução:** Certifique-se de que adicionou documentos à categoria primeiro.

### Problema: "Memória insuficiente"
**Solução:** Reduza `top_k`, use categorias separadas, ou considere usar GPU (faiss-gpu).

### Problema: Buscas lentas
**Solução:** Use FAISS em GPU com `faiss-gpu`, ou aumente a dimensão do índice com IVF.

## Próximos Passos

1. **Otimizações**: Usar IndexIVFFlat ou outras estruturas para datasets maiores
2. **GPU**: Migrar para FAISS com GPU para clusters grandes
3. **Reranking**: Adicionar reranking com modelo mais potente após busca FAISS
4. **Persistência de documentos**: Salvar documento_store com SQLite
