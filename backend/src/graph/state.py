from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class Message(BaseModel):
    role: str = Field(..., description="Quem enviou: 'user' ou 'assistant'")
    content: str = Field(..., description="Conteúdo da mensagem")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Dados adicionais")


class ChatState(BaseModel):
    # Input do usuário
    user_query: str = Field(..., description="Pergunta do usuário")
    
    # Contexto recuperado
    retrieved_documents: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Documentos recuperados da fonte externa"
    )
    retrieved_text: str = Field(
        default="",
        description="Texto combinado dos documentos recuperados"
    )
    
    # LLM Input/Output
    llm_input: str = Field(
        default="",
        description="Prompt compilado para enviar ao LLM"
    )
    llm_response: str = Field(
        default="",
        description="Resposta bruta do LLM"
    )
    
    # Resposta final
    final_response: str = Field(
        default="",
        description="Resposta formatada para o usuário"
    )
    
    # Histórico
    conversation_history: List[Message] = Field(
        default_factory=list,
        description="Histórico completo da conversa"
    )
    
    # Metadados
    session_id: str = Field(
        default="",
        description="ID da sessão de conversa"
    )
    model_used: str = Field(
        default="",
        description="Qual LLM foi usado"
    )
    error: Optional[str] = Field(
        None,
        description="Mensagem de erro, se houver"
    )


class GraphConfig(BaseModel):
    # LLM
    llm_model: str = Field(default="llama3.2:1b", description="Modelo LLM local (Ollama)")
    llm_base_url: str = Field(default="http://localhost:11434", description="Endpoint do Ollama")
    llm_temperature: float = Field(default=0.7, ge=0, le=1)
    
    # Retriever
    max_retrieved_docs: int = Field(default=3, description="Máximo de documentos para recuperar")
    similarity_threshold: float = Field(default=0.5, ge=0, le=1)
    
    # Fonte de dados
    data_source_url: Optional[str] = Field(None, description="URL para scraping (se externa)")
    
    # Histórico
    max_history_messages: int = Field(default=20, description="Máximo de mensagens no histórico")
