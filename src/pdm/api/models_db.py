"""예측 결과 ORM 모델."""
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, Float, String, DateTime
from pdm.api.db import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    rul = Column(Float, nullable=False)
    model_version = Column(String, nullable=False)
    window_mean = Column(Float)  # 입력 윈도우 전체 평균 (드리프트 감지용 요약값)
