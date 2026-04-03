import logging
from langgraph.graph import StateGraph

from .state import ChatState, GraphConfig
from .nodes import ChatbotNodes

logger = logging.getLogger(__name__)


def build_chatbot_graph(config: GraphConfig = None):
    if config is None:
        config = GraphConfig()
    
    # Inicializar nós
    nodes = ChatbotNodes(config)
    
    # Criar grafo
    graph = StateGraph(ChatState)
    
    # Adicionar nós
    graph.add_node("retriever", nodes.retriever_node)
    graph.add_node("prompt_builder", nodes.prompt_builder_node)
    graph.add_node("llm", nodes.llm_node)
    graph.add_node("response_formatter", nodes.response_formatter_node)
    
    # Adicionar edges (conexões entre nós)
    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "prompt_builder")
    graph.add_edge("prompt_builder", "llm")
    graph.add_edge("llm", "response_formatter")
    graph.set_finish_point("response_formatter")
    
    # Compilar grafo
    compiled_graph = graph.compile()
    
    logger.info("Grafo LangGraph construído com sucesso")
    return compiled_graph


def run_chatbot(user_query: str, config: GraphConfig = None, session_id: str = ""):
    if config is None:
        config = GraphConfig()
    
    # Construir grafo
    chatbot_graph = build_chatbot_graph(config)
    
    # Criar estado inicial
    initial_state = ChatState(
        user_query=user_query,
        session_id=session_id
    )
    
    # Executar grafo
    logger.info(f"Executando grafo para query: {user_query}")
    final_state = chatbot_graph.invoke(initial_state)
    
    logger.info(f"Grafo executado com sucesso. Resposta: {len(final_state.final_response)} caracteres")
    return final_state
