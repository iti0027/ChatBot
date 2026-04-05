import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Optional, List
import logging
from pydantic import BaseModel, Field

# Importar módulos locais
from .similarity import Similarity
from models import (
    SimilarityRequest, SimilarityResponse,
    SearchRequest, SearchResponse,
    EmbeddingRequest, EmbeddingResponse,
    HealthResponse, ErrorResponse
)
from .graph import build_chatbot_graph, ChatState
from .graph.state import GraphConfig, Message
from .data_loader import (
    add_urls, add_manual_document,
    get_all_documents, document_count, clear_documents,
    add_documents_to_faiss, search_with_faiss, 
    get_faiss_statistics, clear_faiss_category, clear_all_faiss
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Instância da aplicação FastAPI
app = FastAPI(
    title="ChatBot API",
    description="API do chatbot com embeddings e busca vetorial",
    version="0.1.0"
)

# Configurar CORS para aceitar requisições front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot_graph = None
graph_config = None

def get_chatbot_graph():
    global chatbot_graph, graph_config
    if chatbot_graph is None:
        try:
            graph_config = GraphConfig()
            chatbot_graph = build_chatbot_graph(graph_config)
            logger.info("Grafo LangGraph inicializado")
        except Exception as e:
            logger.error(f"Erro ao inicializar grafo: {e}")
    return chatbot_graph

# Instância global do modelo de similaridade (carregado uma vez)
similarity_model: Optional[Similarity] = None

def get_similarity_model() -> Similarity:
    global similarity_model
    if similarity_model is None:
        try:
            similarity_model = Similarity()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Erro ao carregar modelo de embeddings: {str(e)}"
            )
    return similarity_model

#rotas de verificação básicas
@app.get("/")
def read_root():
    return {
        "message": "ChatBot API está online",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    try:
        model = get_similarity_model()
        model_loaded = True
    except Exception:
        model_loaded = False

    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        version="0.1.0",
        model_loaded=model_loaded
    )

# Endpoints de similaridade
@app.post("/similarity", response_model=SimilarityResponse)
def calculate_similarity(request: SimilarityRequest):
    try:
        model = get_similarity_model()
        similarity_score = model.calculate_similarity(request.text1, request.text2)

        return SimilarityResponse(
            similarity=similarity_score,
            text1=request.text1,
            text2=request.text2
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao calcular similaridade: {str(e)}"
        )

@app.post("/search", response_model=SearchResponse)
def search_similar_texts(request: SearchRequest):
    try:
        model = get_similarity_model()
        results = model.find_most_similar(
            query=request.query,
            texts=request.texts,
            top_k=request.top_k or 5
        )

        return SearchResponse(
            query=request.query,
            results=results,
            total_texts=len(request.texts)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro na busca de textos similares: {str(e)}"
        )

@app.post("/embedding", response_model=EmbeddingResponse)
def generate_embedding(request: EmbeddingRequest):
    try:
        model = get_similarity_model()
        embedding = model.get_embedding(request.text)

        return EmbeddingResponse(
            text=request.text,
            embedding=embedding.tolist()  # Converter numpy array para lista
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao gerar embedding: {str(e)}"
        )

# Endpoints do Chatbot com LangGraph
class ChatbotRequest(BaseModel):
    query: str = Field(..., description="Pergunta do usuário", min_length=1)
    session_id: Optional[str] = Field(None, description="ID da sessão para histórico")


class MessageResponse(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None


class ChatbotResponse(BaseModel):
    query: str
    response: str
    retrieved_docs: int
    model_used: str
    history: List[MessageResponse]
    error: Optional[str] = None


@app.post("/chat", response_model=ChatbotResponse)
def chat(request: ChatbotRequest):
    try:
        logger.info(f"Chat request: {request.query}")
        
        # Obter grafo
        graph = get_chatbot_graph()
        
        if graph is None:
            raise Exception("Falha ao inicializar o grafo")
        
        # Criar estado inicial
        initial_state = ChatState(
            user_query=request.query,
            session_id=request.session_id or "default"
        )
        
        # Executar grafo (retorna dict)
        final_state_dict = graph.invoke(initial_state.model_dump())
        
        # Converter dict de volta para ChatState
        final_state = ChatState(**final_state_dict)
        
        # Construir resposta
        history_response = [
            MessageResponse(
                role=msg.role,
                content=msg.content,
                timestamp=msg.timestamp.isoformat() if msg.timestamp else None
            )
            for msg in final_state.conversation_history
        ]
        
        return ChatbotResponse(
            query=final_state.user_query,
            response=final_state.final_response,
            retrieved_docs=len(final_state.retrieved_documents),
            model_used=final_state.model_used,
            history=history_response,
            error=final_state.error
        )
        
    except Exception as e:
        logger.error(f"Erro no endpoint /chat: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar chat: {str(e)}"
        )


# ==================== ENDPOINTS DE DOCUMENTOS ====================

class AddUrlsRequest(BaseModel):
    urls: List[str] = Field(..., description="Lista de URLs para fazer scraping", min_length=1)


class AddUrlsResponse(BaseModel):
    urls_tentadas: int
    urls_sucesso: int
    documentos_adicionados: int
    total_documentos: int


class AddDocumentRequest(BaseModel):
    title: str = Field(..., description="Título do documento", min_length=1)
    content: str = Field(..., description="Conteúdo do documento", min_length=1)
    source: str = Field(default="manual", description="Fonte do documento")


class DocumentInfo(BaseModel):
    id: int
    title: str
    content: str
    source: str
    similarity_score: Optional[float] = None


class DocumentsListResponse(BaseModel):
    total: int
    documents: List[DocumentInfo]


@app.post("/documents/add-urls", response_model=AddUrlsResponse)
def add_urls_endpoint(request: AddUrlsRequest):
    try:
        logger.info(f"Adicionando {len(request.urls)} URLs")
        result = add_urls(request.urls)
        return AddUrlsResponse(**result)
    except Exception as e:
        logger.error(f"Erro ao adicionar URLs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao fazer scraping: {str(e)}"
        )


@app.post("/documents/add-manual", response_model=DocumentInfo)
def add_manual_document_endpoint(request: AddDocumentRequest):
    try:
        logger.info(f"Adicionando documento manual: {request.title}")
        doc = add_manual_document(
            title=request.title,
            content=request.content,
            source=request.source
        )
        return DocumentInfo(**doc)
    except Exception as e:
        logger.error(f"Erro ao adicionar documento: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao adicionar documento: {str(e)}"
        )


@app.get("/documents", response_model=DocumentsListResponse)
def list_documents():
    try:
        docs = get_all_documents()
        return DocumentsListResponse(
            total=len(docs),
            documents=[DocumentInfo(**doc) for doc in docs]
        )
    except Exception as e:
        logger.error(f"Erro ao listar documentos: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao listar documentos: {str(e)}"
        )


@app.get("/documents/count")
def documents_count():
    try:
        count = document_count()
        return {
            "total_documents": count,
            "message": f"Existem {count} documentos disponíveis"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao contar documentos: {str(e)}"
        )


@app.delete("/documents")
def clear_all_documents():
    try:
        logger.warning("Limpando todos os documentos")
        clear_documents()
        return {
            "message": "Todos os documentos foram removidos",
            "total_documents": 0
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao limpar documentos: {str(e)}"
        )


# ============ FAISS Endpoints ============

class FAISSAddRequest(BaseModel):
    category: str = Field(..., description="Nome da categoria", min_length=1)
    documents: List[dict] = Field(..., description="Lista de documentos com título e conteúdo", min_length=1)


class FAISSSearchRequest(BaseModel):
    query: str = Field(..., description="Texto da busca", min_length=1)
    category: Optional[str] = Field(None, description="Categoria específica (opcional)")
    top_k: Optional[int] = Field(5, description="Número de resultados", ge=1, le=100)


class FAISSSearchResult(BaseModel):
    document: dict
    similarity: float
    category: str
    rank: int


class FAISSSearchResponse(BaseModel):
    query: str
    results: List[FAISSSearchResult]
    total_results: int


class FAISSCategoryStats(BaseModel):
    category: str
    total_documents: int
    index_size: int
    embedding_dim: int
    dirty: bool


class FAISSStatistics(BaseModel):
    total_categories: int
    total_documents: int
    categories: dict


@app.post("/faiss/add", response_model=dict)
def faiss_add_documents(request: FAISSAddRequest):
    try:
        logger.info(f"FAISS: Adicionando {len(request.documents)} docs à categoria '{request.category}'")
        result = add_documents_to_faiss(request.category, request.documents)
        return {
            "success": True,
            "message": f"Adicionados {result['added']} documentos",
            "category": request.category,
            "added": result['added'],
            "total": result['total']
        }
    except Exception as e:
        logger.error(f"Erro ao adicionar docs ao FAISS: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao adicionar documentos: {str(e)}"
        )


@app.post("/faiss/search", response_model=FAISSSearchResponse)
def faiss_search(request: FAISSSearchRequest):
    try:
        logger.info(f"FAISS: Buscando '{request.query}' (categoria: {request.category})")
        results = search_with_faiss(
            query=request.query,
            category=request.category,
            top_k=request.top_k
        )
        
        faiss_results = [
            FAISSSearchResult(
                document=result['document'],
                similarity=result['similarity'],
                category=result['category'],
                rank=result['rank']
            )
            for result in results
        ]
        
        return FAISSSearchResponse(
            query=request.query,
            results=faiss_results,
            total_results=len(faiss_results)
        )
    except Exception as e:
        logger.error(f"Erro ao buscar no FAISS: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao buscar: {str(e)}"
        )


@app.get("/faiss/stats", response_model=FAISSStatistics)
def faiss_statistics():
    try:
        stats = get_faiss_statistics()
        return FAISSStatistics(**stats)
    except Exception as e:
        logger.error(f"Erro ao obter estatísticas FAISS: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao obter estatísticas: {str(e)}"
        )


@app.delete("/faiss/category/{category}")
def faiss_clear_category(category: str):
    try:
        logger.warning(f"FAISS: Limpando categoria '{category}'")
        clear_faiss_category(category)
        return {
            "success": True,
            "message": f"Categoria '{category}' foi limpa",
            "category": category
        }
    except Exception as e:
        logger.error(f"Erro ao limpar categoria FAISS: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao limpar categoria: {str(e)}"
        )


@app.delete("/faiss/all")
def faiss_clear_all():
    try:
        logger.warning("FAISS: Limpando TODOS os índices")
        clear_all_faiss()
        return {
            "success": True,
            "message": "Todos os índices FAISS foram limpos"
        }
    except Exception as e:
        logger.error(f"Erro ao limpar todos os índices FAISS: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao limpar índices: {str(e)}"
        )


# Handler global de erros
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }

if __name__ == "__main__":
    uvicorn.run(
        "backend.src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
