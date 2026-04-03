import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Load environment variables from .env if present
load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chatbot.db")
# For SQLite, add check_same_thread=False
SQLALCHEMY_KWARGS = {
    "connect_args": {"check_same_thread": False}
} if "sqlite" in DATABASE_URL else {}

engine = create_engine(
    DATABASE_URL,
    echo=os.getenv("DB_ECHO", "false").lower() == "true",
    **SQLALCHEMY_KWARGS
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Database Models
class SessionModel(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, index=True, nullable=False)
    user_id = Column(String(255), index=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    # Relationships
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), ForeignKey("sessions.session_id"), index=True, nullable=False)
    user_query = Column(Text, nullable=False)
    llm_response = Column(Text, nullable=True)
    retrieved_documents = Column(Text, nullable=True)  # JSON string of document IDs
    model_used = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    session = relationship("SessionModel", back_populates="messages")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    source = Column(String(255), nullable=False)  # 'manual', 'scraped', 'uploaded'
    category = Column(String(100), default="general", index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    embedding_vector_id = Column(Integer, nullable=True)  # Reference to FAISS index


class DatabaseManager:
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal

    def init_db(self):
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("✅ Database initialized successfully")
        except Exception as e:
            logger.error(f"❌ Error initializing database: {e}")
            raise

    def get_session(self):
        return self.SessionLocal()

    def close_session(self, session):
        if session:
            session.close()


# Global database manager instance
db_manager = DatabaseManager()


def init_db():
    db_manager.init_db()


def get_db():
    db = db_manager.get_session()
    try:
        yield db
    finally:
        db.close()
