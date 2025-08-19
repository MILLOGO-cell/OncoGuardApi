from datetime import datetime, timedelta
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from app.db.database import Base


class TwoFactorCode(Base):
    __tablename__ = "two_factor_codes"

    id = Column(Integer, primary_key=True)
    email = Column(String, nullable=False)
    code = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    used = Column(Boolean, default=False)

    def is_expired(self):
        return datetime.utcnow() > self.created_at + timedelta(minutes=10)