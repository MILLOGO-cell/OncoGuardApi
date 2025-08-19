# app/api/v1/models/password_reset.py
from sqlalchemy import Boolean, Column, Integer, String, DateTime
from datetime import datetime, timedelta
from app.db.database import Base

class PasswordResetCode(Base):
    __tablename__ = "password_reset_codes"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, nullable=False, index=True)
    code = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    used = Column(Boolean, default=False) 

    def is_expired(self) -> bool:
        return datetime.utcnow() > self.created_at + timedelta(minutes=10)
