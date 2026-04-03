import os
import json
import pickle
import logging
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from .similarity import Similarity
from datetime import datetime

logger = logging.getLogger(__name__)

# Defini o diretório para armazenar índices
FAISS_INDICES_DIR = os.path.join(os.path.dirname(__file__), '..', 'indices')
Path(FAISS_INDICES_DIR).mkdir(exist_ok=True)


class FAISSIndex:
    def __init__(self, category: str, embedding_dim: int = 384):
        self.category = category
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)  # L2 (Euclidean) distance
        self.document_map: Dict[int, Dict] = {}  # Mapeia índice FAISS → documento
        self.id_counter = 0
        self.dirty = False  # Flag para saber se precisa salvar
    
    def add_documents(self, documents: List[Dict], embeddings: np.ndarray) -> int:
        if len(documents) != len(embeddings):
            raise ValueError(f"Documentos e embeddings devem ter o mesmo tamanho")
        
        # Converte para float32 (requerido pelo FAISS)
        embeddings_f32 = np.array(embeddings, dtype=np.float32)
        
        # Adiciona ao índice
        self.index.add(embeddings_f32)
        
        # Mapeia documentos
        for i, doc in enumerate(documents):
            self.document_map[self.id_counter + i] = doc
        
        self.id_counter += len(documents)
        self.dirty = True
        
        logger.info(f"[{self.category}] Adicionados {len(documents)} documentos. Total: {self.id_counter}")
        return len(documents)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        if self.id_counter == 0:
            return []
        
        query_f32 = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_f32, min(top_k, self.id_counter))
        
        results = []
        for rank, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS retorna -1 para resultados vazios
                continue
            
            doc = self.document_map.get(int(idx))
            if doc:
                # Converte L2 distance para similaridade (0-1)
                # Quanto menor a distance L2, maior a similaridade
                similarity = 1 / (1 + float(distance))
                
                results.append({
                    'document': doc,
                    'distance': float(distance),
                    'similarity': similarity,
                    'rank': rank + 1
                })
        
        return results
    
    def clear(self):
        """Limpar índice"""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.document_map.clear()
        self.id_counter = 0
        self.dirty = True
        logger.info(f"[{self.category}] Índice limpo")
    
    def get_stats(self) -> Dict:
        """Obter estatísticas do índice"""
        return {
            'category': self.category,
            'total_documents': self.id_counter,
            'index_size': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'dirty': self.dirty
        }


class FAISSManager:
    def __init__(self):
        """Inicializar manager"""
        self.similarity_model = Similarity()
        self.indices: Dict[str, FAISSIndex] = {}
        self._load_all_indices()
    
    def _get_index_path(self, category: str, file_type: str = 'index') -> str:
        safe_category = category.replace('/', '_').replace('\\', '_')
        if file_type == 'index':
            return os.path.join(FAISS_INDICES_DIR, f"{safe_category}.index")
        elif file_type == 'metadata':
            return os.path.join(FAISS_INDICES_DIR, f"{safe_category}_metadata.pkl")
        elif file_type == 'info':
            return os.path.join(FAISS_INDICES_DIR, f"{safe_category}_info.json")
    
    def _load_all_indices(self):
        """Carregar todos os índices salvos do disco"""
        if not os.path.exists(FAISS_INDICES_DIR):
            logger.info(f"Diretório {FAISS_INDICES_DIR} não existe. Criado.")
            return
        
        # Encontra todos os arquivos de índice
        for file in os.listdir(FAISS_INDICES_DIR):
            if file.endswith('.index'):
                category = file.replace('.index', '')
                try:
                    self._load_index(category)
                    logger.info(f"Índice carregado: {category}")
                except Exception as e:
                    logger.error(f"Erro ao carregar índice {category}: {e}")
    
    def _load_index(self, category: str) -> Optional[FAISSIndex]:
        index_path = self._get_index_path(category, 'index')
        metadata_path = self._get_index_path(category, 'metadata')
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            return None
        
        try:
            # Carrega índice FAISS
            faiss_index = faiss.read_index(index_path)
            
            # Carrega metadados
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # Reconstrói FAISSIndex
            idx_obj = FAISSIndex(category)
            idx_obj.index = faiss_index
            idx_obj.document_map = metadata['document_map']
            idx_obj.id_counter = metadata['id_counter']
            idx_obj.dirty = False
            
            self.indices[category] = idx_obj
            return idx_obj
        
        except Exception as e:
            logger.error(f"Erro ao carregar índice {category}: {e}")
            return None
    
    def _save_index(self, category: str):
        if category not in self.indices:
            return
        
        idx_obj = self.indices[category]
        index_path = self._get_index_path(category, 'index')
        metadata_path = self._get_index_path(category, 'metadata')
        info_path = self._get_index_path(category, 'info')
        
        try:
            # Salva índice FAISS
            faiss.write_index(idx_obj.index, index_path)
            
            # Salva metadados
            metadata = {
                'document_map': idx_obj.document_map,
                'id_counter': idx_obj.id_counter
            }
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            # Salva informações (metadados legíveis)
            info = {
                'category': category,
                'total_documents': idx_obj.id_counter,
                'saved_at': datetime.now().isoformat(),
                'embedding_dim': idx_obj.embedding_dim
            }
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)
            
            idx_obj.dirty = False
            logger.info(f"Índice salvo: {category} ({idx_obj.id_counter} documentos)")
        
        except Exception as e:
            logger.error(f"Erro ao salvar índice {category}: {e}")
    
    def add_documents(self, category: str, documents: List[Dict]) -> Dict:
        if not documents:
            return {'added': 0, 'total': 0}
        
        # Cria índice se não existir
        if category not in self.indices:
            self.indices[category] = FAISSIndex(category)
        
        # Gera embeddings para os documentos
        contents = [f"{doc.get('title', '')} {doc.get('content', '')}" for doc in documents]
        embeddings = self.similarity_model.get_embeddings_batch(contents)
        
        # Adiciona ao índice
        added = self.indices[category].add_documents(documents, embeddings)
        
        # Salva o índice
        self._save_index(category)
        
        return {
            'added': added,
            'total': self.indices[category].id_counter,
            'category': category
        }
    
    def search(self, query: str, category: Optional[str] = None, top_k: int = 5) -> Dict:
        query_embedding = self.similarity_model.get_embeddings(query)
        results = {}
        
        # Define categorias a buscar
        categories = [category] if category and category in self.indices else list(self.indices.keys())
        
        for cat in categories:
            if cat in self.indices and self.indices[cat].id_counter > 0:
                results[cat] = self.indices[cat].search(query_embedding, top_k)
        
        return results
    
    def get_documents_in_category(self, category: str) -> List[Dict]:
        if category not in self.indices:
            return []
        
        return list(self.indices[category].document_map.values())
    
    def clear_category(self, category: str):
        if category in self.indices:
            self.indices[category].clear()
            self._save_index(category)
    
    def clear_all(self):
        for category in self.indices:
            self.indices[category].clear()
            self._save_index(category)
        logger.info("Todos os índices foram limpos")
    
    def get_statistics(self) -> Dict:
        stats = {
            'total_categories': len(self.indices),
            'total_documents': sum(idx.id_counter for idx in self.indices.values()),
            'categories': {}
        }
        
        for category, idx_obj in self.indices.items():
            stats['categories'][category] = idx_obj.get_stats()
        
        return stats
    
    def save_all(self):
        for category, idx_obj in self.indices.items():
            if idx_obj.dirty:
                self._save_index(category)
        logger.info("Todos os índices foram salvos")


# Singleton global
_faiss_manager: Optional[FAISSManager] = None


def get_faiss_manager() -> FAISSManager:
    global _faiss_manager
    if _faiss_manager is None:
        _faiss_manager = FAISSManager()
        logger.info("FAISSManager inicializado")
    return _faiss_manager
