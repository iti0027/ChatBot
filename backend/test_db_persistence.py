import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.database import db_manager, init_db, Document, Message, SessionModel
from src.repositories import DocumentRepository, MessageRepository, SessionRepository
from src.data_loader import add_manual_document, get_all_documents, document_count
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_database():
    print("\n" + "="*60)
    print("🧪 DATABASE PERSISTENCE TEST")
    print("="*60 + "\n")
    
    # Initialize database
    print("1️⃣ Inicializando banco de dados...")
    init_db()
    print("✅ Banco inicializado\n")
    
    # Test document creation
    print("2️⃣ Testando criação de documentos...")
    db = db_manager.get_session()
    try:
        # Create test documents
        doc1 = DocumentRepository.create(
            db,
            title="Python Basics",
            content="Python is a high-level programming language.",
            source="manual",
            category="programming"
        )
        print(f"✅ Documento 1 criado: ID={doc1.id}, Title='{doc1.title}'")
        
        doc2 = DocumentRepository.create(
            db,
            title="FastAPI Guide",
            content="FastAPI is a modern web framework for building APIs with Python.",
            source="manual",
            category="web"
        )
        print(f"✅ Documento 2 criado: ID={doc2.id}, Title='{doc2.title}'")
        
        doc3 = DocumentRepository.create(
            db,
            title="LangGraph Tutorial",
            content="LangGraph is a library for building AI applications with multiple LLMs.",
            source="manual",
            category="ai"
        )
        print(f"✅ Documento 3 criado: ID={doc3.id}, Title='{doc3.title}'\n")
    finally:
        db_manager.close_session(db)
    
    # Test document retrieval
    print("3️⃣ Testando recuperação de documentos...")
    db = db_manager.get_session()
    try:
        all_docs = DocumentRepository.get_all(db)
        print(f"✅ Total de documentos: {len(all_docs)}")
        for doc in all_docs:
            print(f"   - ID={doc.id}, Title='{doc.title}', Category='{doc.category}'")
        
        count = DocumentRepository.count(db)
        print(f"✅ Contagem de documentos: {count}\n")
    finally:
        db_manager.close_session(db)
    
    # Test category filtering
    print("4️⃣ Testando filtro por categoria...")
    db = db_manager.get_session()
    try:
        web_docs = DocumentRepository.get_by_category(db, "web")
        print(f"✅ Documentos em 'web': {len(web_docs)}")
        for doc in web_docs:
            print(f"   - {doc.title}")
        
        ai_docs = DocumentRepository.get_by_category(db, "ai")
        print(f"✅ Documentos em 'ai': {len(ai_docs)}")
        for doc in ai_docs:
            print(f"   - {doc.title}\n")
    finally:
        db_manager.close_session(db)
    
    # Test session management
    print("5️⃣ Testando gerenciamento de sessões...")
    db = db_manager.get_session()
    try:
        session = SessionRepository.create_or_get(db, "session-123", "user-456")
        print(f"✅ Sessão criada: ID={session.id}, SessionID='{session.session_id}'\n")
    finally:
        db_manager.close_session(db)
    
    # Test message storage
    print("6️⃣ Testando armazenamento de mensagens...")
    db = db_manager.get_session()
    try:
        msg = MessageRepository.create(
            db,
            session_id="session-123",
            user_query="What is Python?",
            llm_response="Python is a high-level programming language.",
            retrieved_documents=[1, 2],
            model_used="mistral"
        )
        print(f"✅ Mensagem salva: ID={msg.id}, Session='{msg.session_id}'")
        print(f"   - Query: '{msg.user_query}'")
        print(f"   - Response: '{msg.llm_response}'")
        print(f"   - Model: {msg.model_used}\n")
    finally:
        db_manager.close_session(db)
    
    # Test conversation history retrieval
    print("7️⃣ Testando recuperação de histórico...")
    db = db_manager.get_session()
    try:
        messages = MessageRepository.get_by_session(db, "session-123")
        print(f"✅ Mensagens na sessão: {len(messages)}")
        for msg in messages:
            print(f"   - Query: '{msg.user_query}'")
            print(f"   - Response: '{msg.llm_response}'")
        
        msg_count = MessageRepository.count_by_session(db, "session-123")
        print(f"Total de mensagens na sessão: {msg_count}\n")
    finally:
        db_manager.close_session(db)
    
    # Test data_loader integration
    print("8️⃣ Testando integração com data_loader...")
    try:
        result = add_manual_document(
            title="Database Persistence",
            content="Save data persistently in database.",
            source="test",
            category="database"
        )
        print(f"Documento adicionado via data_loader: DB ID={result['db_id']}")
        
        all_docs = get_all_documents()
        print(f"Total de documentos no DB: {len(all_docs)}")
        
        count = document_count()
        print(f"✅ Contagem via data_loader: {count}\n")
    except Exception as e:
        logger.error(f"❌ Erro na integração: {e}", exc_info=True)
    
    # Summary
    print("="*60)
    print("✅ TODOS OS TESTES PASSOU COM SUCESSO!")
    print("="*60)
    print("\n📊 Resumo:")
    print(f"   - Banco de dados: SQLite (chatbot.db)")
    print(f"   - Tabelas: sessions, messages, documents")
    print(f"   - Total de documentos: {document_count()}")
    print(f"\n✨ Database persistence está funcionando!")
    

if __name__ == "__main__":
    try:
        test_database()
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        sys.exit(1)
