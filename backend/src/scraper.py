import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)


class WebScraper:   
    def __init__(self, timeout: int = 10, user_agent: Optional[str] = None):
        self.timeout = timeout
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
    
    def fetch_url(self, url: str) -> Optional[str]:
        try:
            logger.info(f"Fazendo scraping de: {url}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro ao fazer scraping de {url}: {str(e)}")
            return None
    
    def extract_text_from_html(self, html: str) -> str:
        try:
            soup = BeautifulSoup(html, "html.parser")
            
            # Remover scripts e styles
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extrair texto
            text = soup.get_text()
            
            # Limpar espaços em branco
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            logger.error(f"Erro ao extrair texto: {str(e)}")
            return ""
    
    def scrape_url(self, url: str, max_length: int = 5000) -> Optional[Dict[str, str]]:
        html = self.fetch_url(url)
        if not html:
            return None
        
        soup = BeautifulSoup(html, "html.parser")
        
        # Extrair título
        title = soup.title.string if soup.title else urlparse(url).netloc
        
        # Extrair conteúdo
        content = self.extract_text_from_html(html)
        
        # Limitar tamanho
        if len(content) > max_length:
            content = content[:max_length] + "..."
        
        return {
            "url": url,
            "title": title,
            "content": content
        }
    
    def scrape_urls(self, urls: List[str]) -> List[Dict[str, str]]:
        documents = []
        for url in urls:
            doc = self.scrape_url(url)
            if doc:
                documents.append(doc)
        
        logger.info(f"Scraped {len(documents)} documentos de {len(urls)} URLs")
        return documents


class DocumentStore:
    def __init__(self):
        self.documents: List[Dict] = []
        self.document_id_counter = 0
    
    def add_document(self, title: str, content: str, source: str = "local") -> Dict:
        doc = {
            "id": self.document_id_counter,
            "title": title,
            "content": content,
            "source": source
        }
        self.documents.append(doc)
        self.document_id_counter += 1
        logger.info(f"Documento adicionado: {title} (ID: {doc['id']})")
        return doc
    
    def add_scraped_documents(self, scraped_docs: List[Dict]) -> int:
        count = 0
        for doc in scraped_docs:
            self.add_document(
                title=doc.get("title", "Sem título"),
                content=doc.get("content", ""),
                source=doc.get("url", "desconhecida")
            )
            count += 1
        
        return count
    
    def get_all_documents(self) -> List[Dict]:
        return self.documents
    
    def get_content_list(self) -> List[str]:
        return [doc["content"] for doc in self.documents]
    
    def get_document_by_index(self, index: int) -> Optional[Dict]:
        if 0 <= index < len(self.documents):
            return self.documents[index]
        return None
    
    def clear(self):
        self.documents = []
        self.document_id_counter = 0
        logger.info("Document store limpo")
    
    def count(self) -> int:
        return len(self.documents)


# Exports
__all__ = ["WebScraper", "DocumentStore"]
