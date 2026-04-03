#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from faiss_manager import get_faiss_manager
from data_loader import (
    add_documents_to_faiss, search_with_faiss, 
    get_faiss_statistics, clear_all_faiss
)
import time

def print_separator(title: str = ""):
    print("\n" + "=" * 80)
    if title:
        print(f" {title}")
        print("=" * 80)

print_separator("TESTE FAISS - Gerenciador de Índices Vetoriais")

# Documentos de teste por categoria
docs_python = [
    {"title": "O que é Python?", "content": "Python é uma linguagem de programação de alto nível, interpretada, com tipagem dinâmica e interpretada. É versátil e usada em web, data science, automação e machine learning.", "source": "manual"},
    {"title": "Características do Python", "content": "Python é conhecida pela sua sintaxe simples e legível. Suporta múltiplos paradigmas de programação incluindo objeto-orientado, funcional e imperativo.", "source": "manual"},
    {"title": "Ecossistema Python", "content": "Python tem um grande ecossistema de bibliotecas como Django, Flask, NumPy, Pandas, TensorFlow e Scikit-learn para diversos casos de uso.", "source": "manual"},
]

docs_frameworks = [
    {"title": "FastAPI Basics", "content": "FastAPI é um framework web moderno para construir APIs REST em Python 3.7+. É extremamente rápido, fornece documentação automática e é baseado em type hints.", "source": "manual"},
    {"title": "Django Framework", "content": "Django é um framework completo para desenvolvimento web em Python. Inclui ORM, admin panel, autenticação e muitos outros recursos built-in.", "source": "manual"},
    {"title": "Flask Microframework", "content": "Flask é um microframework leve e flexível para criar aplicações web. Oferece apenas o essencial, deixando a escolha de ferramentas para o desenvolvedor.", "source": "manual"},
]

docs_ai = [
    {"title": "O que é LangGraph?", "content": "LangGraph é uma biblioteca para construir aplicações complexas com múltiplos LLMs e ferramentas. Permite criar fluxos de trabalho de IA escaláveis e reutilizáveis.", "source": "manual"},
    {"title": "Embedding de Texto", "content": "Embeddings são representações vetoriais contínuas de texto. Permitem calcular similaridade semântica entre textos usando operações matemáticas simples.", "source": "manual"},
    {"title": "Busca Vetorial com FAISS", "content": "FAISS é uma biblioteca para busca rápida de similaridade em espaços vetoriais altos dimensionais. Otimizado para grandes conjuntos de dados.", "source": "manual"},
]

# 1. Adicionar documentos a diferentes categorias
print_separator("1. Adicionando documentos a diferentes categorias")

print("\n[Categoria: Python]")
start = time.time()
result1 = add_documents_to_faiss("python", docs_python)
elapsed = time.time() - start
print(f"  ✓ Adicionados {result1['added']} documentos")
print(f"  Total na categoria: {result1['total']}")
print(f"  Tempo: {elapsed:.3f}s")

print("\n[Categoria: Frameworks]")
start = time.time()
result2 = add_documents_to_faiss("frameworks", docs_frameworks)
elapsed = time.time() - start
print(f"  ✓ Adicionados {result2['added']} documentos")
print(f"  Total na categoria: {result2['total']}")
print(f"  Tempo: {elapsed:.3f}s")

print("\n[Categoria: IA/ML]")
start = time.time()
result3 = add_documents_to_faiss("ai_ml", docs_ai)
elapsed = time.time() - start
print(f"  ✓ Adicionados {result3['added']} documentos")
print(f"  Total na categoria: {result3['total']}")
print(f"  Tempo: {elapsed:.3f}s")

# 2. Obter estatísticas
print_separator("2. Estatísticas dos índices FAISS")
stats = get_faiss_statistics()
print(f"  Total de categorias: {stats['total_categories']}")
print(f"  Total de documentos: {stats['total_documents']}")
print("\n  Detalhes por categoria:")
for cat, cat_stats in stats['categories'].items():
    print(f"    - {cat}:")
    print(f"        • Documentos: {cat_stats['total_documents']}")
    print(f"        • Dimensão: {cat_stats['embedding_dim']}")
    print(f"        • Dirty flag: {cat_stats['dirty']}")

# 3. Buscar em uma categoria específica
print_separator("3. Buscas em categorias específicas")

print("\n[Busca: 'programação de alto nível' na categoria 'python']")
start = time.time()
results = search_with_faiss("programação de alto nível", category="python", top_k=2)
elapsed = time.time() - start
print(f"  Resultados encontrados: {len(results)}")
print(f"  Tempo: {elapsed:.3f}s")
for i, result in enumerate(results, 1):
    print(f"\n  Resultado {i}:")
    print(f"    Titulo: {result['document'].get('title', 'N/A')}")
    print(f"    Categoria: {result['category']}")
    print(f"    Similaridade: {result['similarity']:.2%}")
    print(f"    Conteúdo: {result['document'].get('content', 'N/A')[:100]}...")

print("\n[Busca: 'velocidade e performance' (em TODAS as categorias)]")
start = time.time()
results_all = search_with_faiss("velocidade e performance", top_k=3)
elapsed = time.time() - start
print(f"  Resultados encontrados: {len(results_all)}")
print(f"  Tempo: {elapsed:.3f}s")
for i, result in enumerate(results_all, 1):
    print(f"\n  Resultado {i} [{result['category']}]:")
    print(f"    Titulo: {result['document'].get('title', 'N/A')}")
    print(f"    Similaridade: {result['similarity']:.2%}")

# 4. Buscas adicionais para demonstrar versatilidade
print_separator("4. Buscas adicionais em diferentes categorias")

queries = [
    ("APIs REST", "frameworks"),
    ("representações vetoriais", "ai_ml"),
    ("interpretada", "python"),
]

for query, category in queries:
    print(f"\n['{query}' na categoria '{category}']")
    results = search_with_faiss(query, category=category, top_k=1)
    if results:
        best = results[0]
        print(f"  ✓ Melhor resultado: {best['document'].get('title', 'N/A')}")
        print(f"    Similaridade: {best['similarity']:.2%}")
    else:
        print(f"  ✗ Nenhum resultado encontrado")

# 5. Sumário final
print_separator("5. Sumário Final")
final_stats = get_faiss_statistics()
print(f"  ✓ {final_stats['total_categories']} categorias criadas")
print(f"  ✓ {final_stats['total_documents']} documentos indexados")
print(f"  ✓ Todos os índices salvos em disco")
print(f"\n  Diretório de índices: ../indices/")

print("\n✓ TESTE CONCLUÍDO COM SUCESSO!\n")
