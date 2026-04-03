from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

#calculo de similaridade e busca de textos similares
class SimilarityRequest(BaseModel):
    text1: str = Field(..., description="Primeiro texto para comparação", min_length=1)
    text2: str = Field(..., description="Segundo texto para comparação", min_length=1)

# resposta da similaridade entre dois textos
class SimilarityResponse(BaseModel):
    similarity: float = Field(..., description="Valor de similaridade entre os textos entre 0 e 1", ge=0.0, le=1.0)
    text1: str = Field(..., description="Primeiro texto comparado")
    text2: str = Field(..., description="Segundo texto comparado")

# requisição para busca de textos similares
class SearchRequest(BaseModel):
    query: str = Field(..., description="Texto para busca", min_length=1)
    texts: List[str] = Field(..., description="Lista de textos para comparação", min_items=1)
    top_k: Optional[int] = Field(5, description="Número de resultados mais similares a retornar", ge=1, le=100)

# resultado individual do texto encontrado na busca
class SearchResult(BaseModel):
    text: str = Field(..., description="Texto encontrado")
    similarity: float = Field(..., description="Nível de similaridade", ge=0.0, le=1.0)
    index: int = Field(..., description="Índice do texto na lista original", ge=0)

# resposta da busca de textos similares
class SearchResponse(BaseModel):
    query: str = Field(..., description="Texto de busca")
    results: List[SearchResult] = Field(..., description="Lista de resultados mais similares encontrados")
    total_texts: int = Field(..., description="Número total de textos comparados")

# requisição para geração de embedding
class EmbeddingRequest(BaseModel):
    text: str = Field(..., description="Texto para gerar embedding", min_length=1)

# resposta da geração de embedding
class EmbeddingResponse(BaseModel):
    text: str = Field(..., description="Texto original")
    embedding: List[float] = Field(..., description="Vetor de embedding gerado para o texto")

# resposta da verificação de saúde da API
class HealthResponse(BaseModel):
    status: str = Field(..., description="Status do serviço")
    version: str = Field(..., description="Versão da API")
    model_loaded: bool = Field(..., description="Verificar se o modelo de embeddings está carregado")

# resposta de erro para casos de falhas na API
class ErrorResponse(BaseModel):
    error: str = Field(..., description="Mensagem de erro")
    detail: Optional[str] = Field(None, description="Detalhes sobre o erro")

# modelos disponíveis para geração de embeddings
class ModelType(str, Enum):
    ALL_MINI_LM_L6_V2 = "all-MiniLM-L6-v2"

# requisição para divisão de texto em chunks
class ChunkRequest(BaseModel):
    text: str = Field(..., description="Texto para ser dividido em chunks", min_length=1)
    chunk_size: Optional[int] = Field(500, description="Tamanho máximo de cada chunk", ge=50, le=2048)
    overlap: Optional[int] = Field(100, description="Número de caracteres de sobreposição entre chunks", ge=0)

# modelo de chunk gerado a partir do texto original
class Chunk(BaseModel):
    text: str = Field(..., description="Texto do chunk")
    start_index: int = Field(..., description="Índice inicial no texto original", ge=0)
    end_index: int = Field(..., description="Índice final no texto original", ge=0)

# resposta da divisão de texto em chunks
class ChunkResponse(BaseModel):
    original_text: str = Field(..., description="Texto original que foi dividido em chunks")
    chunks: List[Chunk] = Field(..., description="Lista de chunks gerados")
    total_chunks: int = Field(..., description="Número total de chunks gerados")
