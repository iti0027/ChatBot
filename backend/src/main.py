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
from similarity import Similarity
from models import (
    SimilarityRequest, SimilarityResponse,
    SearchRequest, SearchResponse,
    EmbeddingRequest, EmbeddingResponse,
    HealthResponse, ErrorResponse
)
from graph import build_chatbot_graph, ChatState
from graph.state import GraphConfig, Message

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
    allow_origins=["*"],  # Em produção, especificar os domínios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instância global do grafo (construído uma vez)
chatbot_graph = None
graph_config = None

def get_chatbot_graph():
    """Obter ou criar instância do grafo"""
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
    """Obter ou criar instância do modelo de similaridade"""
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
    """Rota raiz para verificar se a API está online"""
    return {
        "message": "ChatBot API está online",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Verificar saúde da API e status do modelo"""
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
    """
    Calcular similaridade entre dois textos

    - **text1**: Primeiro texto para comparação
    - **text2**: Segundo texto para comparação
    """
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
    """
    Encontrar textos mais similares a uma query

    - **query**: Texto de busca
    - **texts**: Lista de textos para comparar
    - **top_k**: Número de resultados (padrão 5, máximo 100)
    """
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
    """
    Gerar embedding (vetor) para um texto

    - **text**: Texto para gerar embedding
    """
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
    """Requisição para o chatbot"""
    query: str = Field(..., description="Pergunta do usuário", min_length=1)
    session_id: Optional[str] = Field(None, description="ID da sessão para histórico")


class MessageResponse(BaseModel):
    """Resposta individual no histórico"""
    role: str
    content: str
    timestamp: Optional[str] = None


class ChatbotResponse(BaseModel):
    """Resposta do chatbot com histórico"""
    query: str
    response: str
    retrieved_docs: int
    model_used: str
    history: List[MessageResponse]
    error: Optional[str] = None


@app.post("/chat", response_model=ChatbotResponse)
def chat(request: ChatbotRequest):
    """
    Endpoint principal do chatbot usando LangGraph
    
    - **query**: Pergunta do usuário
    - **session_id**: ID da sessão (opcional, para histórico)
    """
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

# Handler global de erros
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
