"""
Gerenciador de Documentos Global
Mantém os documentos disponíveis para o retriever
Integra com FAISS para buscas otimizadas por categoria
"""

import logging
from scraper import DocumentStore, WebScraper
from faiss_manager import get_faiss_manager
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Store global de documentos
_document_store: Optional[DocumentStore] = None
_scraper: Optional[WebScraper] = None


def get_document_store() -> DocumentStore:
    """Obter ou criar instância global do store"""
    global _document_store
    if _document_store is None:
        _document_store = DocumentStore()
        logger.info("Document Store inicializado")
    return _document_store


def get_scraper() -> WebScraper:
    """Obter ou criar instância global do scraper"""
    global _scraper
    if _scraper is None:
        _scraper = WebScraper()
        logger.info("Web Scraper inicializado")
    return _scraper


def add_urls(urls: List[str]) -> Dict:
    """
    Fazer scraping de URLs e adicionar ao store
    
    Args:
        urls: Lista de URLs para fazer scraping
        
    Returns:
        Dict com {urls_scrapeadas, documentos_adicionados, erros}
    """
    scraper = get_scraper()
    store = get_document_store()
    
    scraped_docs = scraper.scrape_urls(urls)
    added_count = store.add_scraped_documents(scraped_docs)
    
    result = {
        "urls_tentadas": len(urls),
        "urls_sucesso": len(scraped_docs),
        "documentos_adicionados": added_count,
        "total_documentos": store.count()
    }
    
    logger.info(f"Scraping resultado: {result}")
    return result


def add_manual_document(title: str, content: str, source: str = "manual") -> Dict:
    """
    Adicionar documento manualmente
    
    Args:
        title: Título do documento
        content: Conteúdo
        source: Fonte (padrão: 'manual')
        
    Returns:
        Documento adicionado
    """
    store = get_document_store()
    return store.add_document(title, content, source)


def get_all_documents() -> List[Dict]:
    """Retornar todos os documentos"""
    store = get_document_store()
    return store.get_all_documents()


def get_content_for_retrieval() -> List[str]:
    """Retornar conteúdos para o retriever usar"""
    store = get_document_store()
    contents = store.get_content_list()
    
    if not contents:
        # Se vazio, retornar documentos padrão
        logger.warning("Nenhum documento adicionado, usando exemplos padrão")
        return get_default_documents()
    
    return contents


def get_default_documents() -> List[str]:
    """Retornar documentos padrão para demo"""
    return [
        "Python é uma linguagem de programação de alto nível, interpretada, interativa e orientada a objetos.",
        "FastAPI é um framework web moderno e rápido para construir APIs com Python.",
        "LangGraph é uma biblioteca para construir aplicações complexas com múltiplos LLMs e ferramentas.",
        "Embeddings são representações vetoriais de texto que permitem calcular similaridade semântica.",
        "Ollama permite executar grandes modelos de linguagem localmente no seu computador.",
        "Web scraping é a técnica de extrair dados automaticamente de websites.",
    ]


def clear_documents():
    """Limpar todos os documentos"""
    store = get_document_store()
    store.clear()


def document_count() -> int:
    """Retornar quantidade de documentos"""
    store = get_document_store()
    return store.count()


# ============ FAISS Integration ============

def add_documents_to_faiss(category: str, documents: List[Dict]) -> Dict:
    """
    Adicionar documentos ao índice FAISS de uma categoria
    
    Args:
        category: Nome da categoria (ex: 'manual', 'scraped', 'chat_history')
        documents: Lista de dicts com {'title', 'content', 'source', ...}
        
    Returns:
        Estatísticas de adição
    """
    manager = get_faiss_manager()
    result = manager.add_documents(category, documents)
    logger.info(f"FAISS [{category}] Adicionados {result['added']} documentos. Total: {result['total']}")
    return result


def search_with_faiss(query: str, category: Optional[str] = None, top_k: int = 5) -> Dict:
    """
    Buscar documentos usando FAISS
    
    Args:
        query: Texto da busca
        category: Categoria específica (None = todas)
        top_k: Número de resultados por categoria
        
    Returns:
        Dict com resultados organizados por categoria
    """
    manager = get_faiss_manager()
    results = manager.search(query, category, top_k)
    
    # Flattena resultados para o formato esperado pelo retriever
    all_results = []
    for cat, cat_results in results.items():
        for result in cat_results:
            all_results.append({
                'document': result['document'],
                'similarity': result['similarity'],
                'category': cat,
                'rank': result['rank']
            })
    
    # Ordena por similaridade
    all_results.sort(key=lambda x: x['similarity'], reverse=True)
    return all_results[:top_k]


def get_faiss_statistics() -> Dict:
    """Obter estatísticas dos índices FAISS"""
    manager = get_faiss_manager()
    return manager.get_statistics()


def clear_faiss_category(category: str):
    """Limpar uma categoria do FAISS"""
    manager = get_faiss_manager()
    manager.clear_category(category)
    logger.info(f"FAISS [{category}] Limpado")


def clear_all_faiss():
    """Limpar todos os índices FAISS"""
    manager = get_faiss_manager()
    manager.clear_all()


def save_all_faiss():
    """Salvar todos os índices FAISS"""
    manager = get_faiss_manager()
    manager.save_all()


# Exports
__all__ = [
    "get_document_store",
    "get_scraper",
    "add_urls",
    "add_manual_document",
    "get_all_documents",
    "get_content_for_retrieval",
    "get_default_documents",
    "clear_documents",
    "document_count",
    # FAISS functions
    "add_documents_to_faiss",
    "search_with_faiss",
    "get_faiss_statistics",
    "clear_faiss_category",
    "clear_all_faiss",
    "save_all_faiss",
]
