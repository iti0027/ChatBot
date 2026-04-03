from sqlalchemy.orm import Session
from sqlalchemy import desc, and_
from datetime import datetime, timedelta
from .database import Document, Message, SessionModel
import logging
from typing import List, Optional
import json

logger = logging.getLogger(__name__)


class DocumentRepository:
    @staticmethod
    def create(db: Session, title: str, content: str, source: str, category: str = "general") -> Document:
        doc = Document(
            title=title,
            content=content,
            source=source,
            category=category
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        logger.info(f"✅ Document created: {doc.id} - {title}")
        return doc

    @staticmethod
    def get_all(db: Session, category: Optional[str] = None) -> List[Document]:
        query = db.query(Document)
        if category:
            query = query.filter(Document.category == category)
        return query.order_by(desc(Document.created_at)).all()

    @staticmethod
    def get_by_id(db: Session, doc_id: int) -> Optional[Document]:
        return db.query(Document).filter(Document.id == doc_id).first()

    @staticmethod
    def get_by_category(db: Session, category: str) -> List[Document]:
        return db.query(Document).filter(Document.category == category).all()

    @staticmethod
    def delete(db: Session, doc_id: int) -> bool:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if doc:
            db.delete(doc)
            db.commit()
            logger.info(f"✅ Document deleted: {doc_id}")
            return True
        return False

    @staticmethod
    def delete_all(db: Session) -> int:
        count = db.query(Document).delete()
        db.commit()
        logger.info(f"✅ Deleted {count} documents")
        return count

    @staticmethod
    def count(db: Session, category: Optional[str] = None) -> int:
        query = db.query(Document)
        if category:
            query = query.filter(Document.category == category)
        return query.count()

    @staticmethod
    def update_embedding_id(db: Session, doc_id: int, vector_id: int) -> bool:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if doc:
            doc.embedding_vector_id = vector_id
            db.commit()
            return True
        return False


class MessageRepository:
    @staticmethod
    def create(
        db: Session,
        session_id: str,
        user_query: str,
        llm_response: Optional[str] = None,
        retrieved_documents: Optional[List[int]] = None,
        model_used: Optional[str] = None
    ) -> Message:
        msg = Message(
            session_id=session_id,
            user_query=user_query,
            llm_response=llm_response,
            retrieved_documents=json.dumps(retrieved_documents) if retrieved_documents else None,
            model_used=model_used
        )
        db.add(msg)
        db.commit()
        db.refresh(msg)
        logger.info(f"✅ Message saved for session: {session_id}")
        return msg

    @staticmethod
    def get_by_session(db: Session, session_id: str, limit: int = 50) -> List[Message]:
        return db.query(Message).filter(
            Message.session_id == session_id
        ).order_by(Message.created_at).limit(limit).all()

    @staticmethod
    def get_recent(db: Session, session_id: str, hours: int = 24) -> List[Message]:
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return db.query(Message).filter(
            and_(
                Message.session_id == session_id,
                Message.created_at >= cutoff
            )
        ).order_by(Message.created_at).all()

    @staticmethod
    def delete_by_session(db: Session, session_id: str) -> int:
        count = db.query(Message).filter(Message.session_id == session_id).delete()
        db.commit()
        logger.info(f"✅ Deleted {count} messages for session: {session_id}")
        return count

    @staticmethod
    def count_by_session(db: Session, session_id: str) -> int:
        return db.query(Message).filter(Message.session_id == session_id).count()


class SessionRepository:
    @staticmethod
    def create_or_get(db: Session, session_id: str, user_id: Optional[str] = None) -> SessionModel:
        session = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
        if not session:
            session = SessionModel(session_id=session_id, user_id=user_id)
            db.add(session)
            db.commit()
            db.refresh(session)
            logger.info(f"✅ Session created: {session_id}")
        return session

    @staticmethod
    def get_all_active(db: Session) -> List[SessionModel]:
        return db.query(SessionModel).filter(SessionModel.is_active == True).all()

    @staticmethod
    def close_session(db: Session, session_id: str) -> bool:
        session = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
        if session:
            session.is_active = False
            db.commit()
            logger.info(f"✅ Session closed: {session_id}")
            return True
        return False

    @staticmethod
    def delete(db: Session, session_id: str) -> bool:
        session = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
        if session:
            # Delete messages first
            db.query(Message).filter(Message.session_id == session_id).delete()
            # Delete session
            db.delete(session)
            db.commit()
            logger.info(f"✅ Session deleted: {session_id}")
            return True
        return False
