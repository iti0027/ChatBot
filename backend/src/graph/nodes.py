import logging
from typing import Dict, Any, List
from datetime import datetime

from similarity import Similarity
from .state import ChatState, Message

logger = logging.getLogger(__name__)


class ChatbotNodes:
    
    def __init__(self, config: "GraphConfig"):  # noqa
        self.config = config
        self.similarity_model = Similarity()
        self.retrieved_docs = []  # Cache de documentos
        logger.info("Nós do chatbot inicializados")
    
    def retriever_node(self, state: ChatState) -> ChatState:
        try:
            logger.info(f"Retriever: Processando query: {state.user_query}")
            
            # Se não temos documentos em cache, podemos simular ou buscar de uma fonte
            # Aqui vamos simular alguns documentos para demo
            sample_docs = [
                {"id": 1, "title": "Python Basics", "content": "Python é uma linguagem de programação versátil e poderosa."},
                {"id": 2, "title": "Web Development", "content": "FastAPI é um framework moderno para criar APIs REST em Python."},
                {"id": 3, "title": "Machine Learning", "content": "LangGraph é uma ferramenta para orquestrar fluxos de IA."},
                {"id": 4, "title": "NLP", "content": "Embeddings são representações vetoriais de texto usadas em busca semântica."},
            ]
            
            # Usar similaridade para recuperar os melhores documentos
            texts = [doc["content"] for doc in sample_docs]
            results = self.similarity_model.find_most_similar(
                query=state.user_query,
                texts=texts,
                top_k=self.config.max_retrieved_docs
            )
            
            # Mapear resultados para documentos
            retrieved_docs = []
            retrieved_text_parts = []
            
            for result in results:
                doc_idx = result["index"]
                doc = sample_docs[doc_idx]
                doc["similarity_score"] = result["similarity"]
                retrieved_docs.append(doc)
                retrieved_text_parts.append(f"[{doc['title']}]: {doc['content']}")
            
            state.retrieved_documents = retrieved_docs
            state.retrieved_text = "\n".join(retrieved_text_parts)
            
            logger.info(f"Retriever: {len(retrieved_docs)} documentos recuperados")
            return state
            
        except Exception as e:
            logger.error(f"Erro no retriever: {str(e)}")
            state.error = f"Erro ao recuperar: {str(e)}"
            return state
    
    def prompt_builder_node(self, state: ChatState) -> ChatState:
        try:
            logger.info("Prompt Builder: Construindo prompt para LLM")
            
            # Construir prompt com contexto
            context_section = ""
            if state.retrieved_text:
                context_section = f"""
Contexto relevante:
{state.retrieved_text}

"""
            # Histórico de conversas
            history_section = ""
            if state.conversation_history:
                history_lines = []
                for msg in state.conversation_history[-5:]:  # Últimas 5 mensagens
                    role = "Usuário" if msg.role == "user" else "Assistente"
                    history_lines.append(f"{role}: {msg.content}")
                history_section = "Histórico da conversa:\n" + "\n".join(history_lines) + "\n\n"
            
            # Montar prompt final
            llm_input = f"""{history_section}{context_section}Pergunta do usuário: {state.user_query}

Responda com base no contexto fornecido. Se não tiver informações suficientes, diga que não sabe."""
            
            state.llm_input = llm_input
            logger.info("Prompt Builder: Prompt construído com sucesso")
            return state
            
        except Exception as e:
            logger.error(f"Erro ao construir prompt: {str(e)}")
            state.error = f"Erro ao construir prompt: {str(e)}"
            return state
    
    def llm_node(self, state: ChatState) -> ChatState:
        try:
            logger.info(f"LLM Node: Chamando {self.config.llm_model}")
            
            # Para demo, vamos simular uma resposta do LLM
            # Em produção, chamar Ollama via requests ou langchain
            llm_response = f"""Baseado na informação fornecida:

{state.retrieved_text if state.retrieved_text else 'Sem contexto específico.'}

Respondendo à sua pergunta sobre "{state.user_query}":

A resposta depende do contexto específico. Recomendo consultar a documentação para mais detalhes."""
            
            state.llm_response = llm_response
            state.model_used = self.config.llm_model
            
            logger.info("LLM Node: Resposta gerada com sucesso")
            return state
            
        except Exception as e:
            logger.error(f"Erro ao chamar LLM: {str(e)}")
            state.error = f"Erro ao processar com LLM: {str(e)}"
            return state
    
    def response_formatter_node(self, state: ChatState) -> ChatState:
        try:
            logger.info("Response Formatter: Formatando resposta final")
            
            # Usar resposta do LLM como reposta final
            state.final_response = state.llm_response or "Desculpe, não consegui processar sua pergunta."
            
            # Adicionar user query e response ao histórico
            state.conversation_history.append(Message(
                role="user",
                content=state.user_query
            ))
            
            state.conversation_history.append(Message(
                role="assistant",
                content=state.final_response
            ))
            
            # Manter apenas as últimas N mensagens
            if len(state.conversation_history) > self.config.max_history_messages:
                state.conversation_history = state.conversation_history[-self.config.max_history_messages:]
            
            logger.info("Response Formatter: Resposta formatada e histórico atualizado")
            return state
            
        except Exception as e:
            logger.error(f"Erro ao formatar resposta: {str(e)}")
            state.error = f"Erro ao formatar: {str(e)}"
            return state
