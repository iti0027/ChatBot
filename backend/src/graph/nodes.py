import logging
from typing import Dict, Any, List
from datetime import datetime

from similarity import Similarity
from llm import OllamaClient, OllamaConfig
from data_loader import get_content_for_retrieval, get_all_documents
from .state import ChatState, Message

logger = logging.getLogger(__name__)


class ChatbotNodes:
    
    def __init__(self, config: "GraphConfig"):  # noqa
        self.config = config
        self.similarity_model = Similarity()
        
        # Inicializar cliente Ollama
        ollama_config = OllamaConfig(
            base_url=config.llm_base_url,
            model=config.llm_model,
            temperature=config.llm_temperature
        )
        self.llm_client = OllamaClient(ollama_config)
        
        # Verificar saúde do Ollama
        if self.llm_client.check_health():
            logger.info("✓ Ollama estava disponível")
        else:
            logger.warning("⚠ Ollama não está disponível - usando modo fallback")
        
        self.retrieved_docs = []  # Cache de documentos
        logger.info("Nós do chatbot inicializados")
    
    def retriever_node(self, state: ChatState) -> ChatState:
        try:
            logger.info(f"Retriever: Processando query: {state.user_query}")
            
            # Obter conteúdos do data_loader
            content_list = get_content_for_retrieval()
            
            if not content_list:
                logger.warning("Nenhum documento disponível para retrieval")
                state.retrieved_documents = []
                state.retrieved_text = ""
                return state
            
            # Usar similaridade para recuperar os melhores documentos
            results = self.similarity_model.find_most_similar(
                query=state.user_query,
                texts=content_list,
                top_k=min(self.config.max_retrieved_docs, len(content_list))
            )
            
            # Obter documentos completos do data_loader
            all_docs = get_all_documents()
            
            # Mapear resultados para documentos com metadados
            retrieved_docs = []
            retrieved_text_parts = []
            
            for result in results:
                doc_idx = result["index"]
                if doc_idx < len(all_docs):
                    doc = all_docs[doc_idx]
                    doc["similarity_score"] = result["similarity"]
                    retrieved_docs.append(doc)
                    title = doc.get("title", "Sem título")
                    content = doc.get("content", "")[:200]  # Limitar a 200 chars
                    retrieved_text_parts.append(f"[{title}]: {content}...")
            
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
            
            # Chamar Ollama para gerar resposta
            llm_response = self.llm_client.generate(
                prompt=state.llm_input,
                system_prompt="Você é um assistente inteligente e útil. Responda sucintamente com base no contexto fornecido."
            )
            
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
