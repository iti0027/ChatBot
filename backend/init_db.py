import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.database import db_manager, init_db
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    print("🔧 Initializing ChatBot Database...\n")
    
    try:
        init_db()
        print("\n✅ Database initialization completed successfully!")
        print("📊 Database file: ./chatbot.db")
        print("\n📋 Created tables:")
        print("   - sessions: User session tracking")
        print("   - messages: Conversation history")
        print("   - documents: Stored documents")
        print("\n✨ Ready to use!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
