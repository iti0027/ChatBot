from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import List, Dict, Tuple

class Similarity:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        print(f"Modelo {model_name} funcionando corretamente.")

    def get_embeddings(self, text: str) -> np.ndarray:
        return self.model.encode(text)

    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts)

    def calculate_similarity(self, text1: str, text2: str) -> float:
        embedding1 = self.get_embeddings(text1)
        embedding2 = self.get_embeddings(text2)
        similarity = util.pytorch_cos_sim(embedding1, embedding2)
        return float(similarity[0][0])

    def find_most_similar(self, query: str, texts: List[str], top_k: int = 5) -> List[Dict]:
        query_embedding = self.get_embeddings(query)
        text_embeddings = self.get_embeddings_batch(texts)

        similarities = util.pytorch_cos_sim(query_embedding, text_embeddings)[0]

        top_k_indices = np.argsort(-similarities.numpy())[:min(top_k, len(texts))]
        results = []
        for idx in top_k_indices:
            results.append({
                "text": texts[idx],
                "similarity": float(similarities[idx]),
                "index": int(idx)
            })
        return results

if __name__ == "__main__":
    print(f"=" * 60)
    print(f"Testando Similarity class")
    print(f"=" * 60)

    similarity = Similarity()
    texts = [
        "Barcelona é uma cidade na Espanha.",
        "Madrid é a capital da Espanha.",
        "FC Barcelona é um clube de futebol na Espanha.",
        "FC Real Madrid é um clube de futebol na Espanha.",
        "FC Barcelona é o maior rival do FC Real Madrid."
    ]

    query = "Sport club do recife"

    print(f"Query: {query}")
    print(f"Textos mais similares:")
    print("-" * 60)

    results = similarity.find_most_similar(query, texts, top_k=3)

    for i, result in enumerate(results, 1):
        print(f"{i}. [{result["similarity"]:.4f}] {result["text"]}")

    print(f"=" * 60)
    print(f"Teste concluído")
    print(f"=" * 60)

