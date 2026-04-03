import logging
from .scraper import DocumentStore, WebScraper
from .faiss_manager import get_faiss_manager
from .database import db_manager
from .repositories import DocumentRepository
from typing import List, Dict, Optional
import json

logger = logging.getLogger(__name__)

# Store global de documentos (para retrocompatibilidade)
_document_store: Optional[DocumentStore] = None
_scraper: Optional[WebScraper] = None


def get_document_store() -> DocumentStore:
    global _document_store
    if _document_store is None:
        _document_store = DocumentStore()
        logger.info("✅ Document Store inicializado")
    return _document_store


def get_scraper() -> WebScraper:
    global _scraper
    if _scraper is None:
        _scraper = WebScraper()
        logger.info("✅ Web Scraper inicializado")
    return _scraper


def add_urls(urls: List[str]) -> Dict:
    scraper = get_scraper()
    store = get_document_store()
    db = db_manager.get_session()
    
    try:
        scraped_docs = scraper.scrape_urls(urls)
        added_count = store.add_scraped_documents(scraped_docs)
        
        # Salvar no banco de dados
        db_count = 0
        for doc in scraped_docs:
            try:
                DocumentRepository.create(
                    db,
                    title=doc.get('title', 'Untitled'),
                    content=doc.get('content', ''),
                    source='scraped',
                    category='web'
                )
                db_count += 1
            except Exception as e:
                logger.warning(f"Erro ao salvar documento no DB: {e}")
        
        result = {
            "urls_tentadas": len(urls),
            "urls_sucesso": len(scraped_docs),
            "documentos_adicionados": added_count,
            "documentos_salvos_db": db_count,
            "total_documentos": store.count()
        }
        
        logger.info(f"✅ Scraping resultado: {result}")
        return result
        
    finally:
        db_manager.close_session(db)


def add_manual_document(title: str, content: str, source: str = "manual", category: str = "general") -> Dict:
    store = get_document_store()
    db = db_manager.get_session()
    
    try:
        # Adicionar ao store em memória
        in_memory_result = store.add_document(title, content, source)
        
        # Adicionar ao banco de dados
        db_doc = DocumentRepository.create(
            db,
            title=title,
            content=content,
            source=source,
            category=category
        )
        
        result = {
            **in_memory_result,
            "db_id": db_doc.id,
            "saved_to_db": True
        }
        
        logger.info(f"✅ Documento adicionado manualmente. ID: {db_doc.id}")
        return result
        
    finally:
        db_manager.close_session(db)


def get_all_documents() -> List[Dict]:
    db = db_manager.get_session()
    try:
        documents = DocumentRepository.get_all(db)
        return [
            {
                "id": doc.id,
                "title": doc.title,
                "content": doc.content,
                "source": doc.source,
                "category": doc.category,
                "created_at": doc.created_at.isoformat() if doc.created_at else None
            }
            for doc in documents
        ]
    finally:
        db_manager.close_session(db)


def get_content_for_retrieval() -> List[str]:
    db = db_manager.get_session()
    try:
        docs = DocumentRepository.get_all(db)
        contents = [doc.content for doc in docs]
        
        if not contents:
            # Se vazio, retornar documentos padrão
            logger.warning("⚠️ Nenhum documento no DB, usando exemplos padrão")
            return get_default_documents()
        
        logger.info(f"📚 Recuperados {len(contents)} documentos do DB")
        return contents
        
    finally:
        db_manager.close_session(db)


def get_default_documents() -> List[str]:
    return [
        "Python é uma linguagem de programação de alto nível, interpretada, interativa e orientada a objetos.",
        "FastAPI é um framework web moderno e rápido para construir APIs com Python.",
        "LangGraph é uma biblioteca para construir aplicações complexas com múltiplos LLMs e ferramentas.",
        "Embeddings são representações vetoriais de texto que permitem calcular similaridade semântica.",
        "Ollama permite executar grandes modelos de linguagem localmente no seu computador.",
        "Web scraping é a técnica de extrair dados automaticamente de websites.",
    ]


def clear_documents():
    store = get_document_store()
    store.clear()
    
    db = db_manager.get_session()
    try:
        count = DocumentRepository.delete_all(db)
        logger.info(f"✅ Deletados {count} documentos do banco de dados")
    finally:
        db_manager.close_session(db)


def document_count() -> int:
    db = db_manager.get_session()
    try:
        count = DocumentRepository.count(db)
        return count
    finally:
        db_manager.close_session(db)


def get_documents_by_category(category: str) -> List[Dict]:
    db = db_manager.get_session()
    try:
        documents = DocumentRepository.get_by_category(db, category)
        return [
            {
                "id": doc.id,
                "title": doc.title,
                "content": doc.content,
                "source": doc.source,
                "category": doc.category,
                "created_at": doc.created_at.isoformat() if doc.created_at else None
            }
            for doc in documents
        ]
    finally:
        db_manager.close_session(db)


# ============ FAISS Integration ============

def add_documents_to_faiss(category: str, documents: List[Dict]) -> Dict:
    manager = get_faiss_manager()
    result = manager.add_documents(category, documents)
    logger.info(f"🔍 FAISS [{category}] Adicionados {result['added']} documentos. Total: {result['total']}")
    return result


def search_with_faiss(query: str, category: Optional[str] = None, top_k: int = 5) -> Dict:
    manager = get_faiss_manager()
    results = manager.search(query, category, top_k)
    
    # Flatten resultados para o formato esperado pelo retriever
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
    manager = get_faiss_manager()
    return manager.get_statistics()


def clear_faiss_category(category: str):
    manager = get_faiss_manager()
    manager.clear_category(category)
    logger.info(f"🔍 FAISS [{category}] Limpado")


def clear_all_faiss():
    manager = get_faiss_manager()
    manager.clear_all()


def save_all_faiss():
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
    "get_documents_by_category",
    # FAISS functions
    "add_documents_to_faiss",
    "search_with_faiss",
    "get_faiss_statistics",
    "clear_faiss_category",
    "clear_all_faiss",
    "save_all_faiss",
    "db_manager",
]
