"""
Gerenciador de Documentos Global
Mantém os documentos disponíveis para o retriever
"""

import logging
from scraper import DocumentStore, WebScraper
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
]
