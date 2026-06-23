"""DB 연결·세션. URL은 환경변수 DATABASE_URL, 없으면 SQLite 기본값.

이래야 로컬(SQLite) -> Docker(PostgreSQL) 전환 시 코드 변경 없이
환경변수만 바꾸면 된다.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///predictions.db")

# SQLite는 단일스레드 기본이라 FastAPI(멀티스레드)용 옵션 필요. Postgres엔 무해하게 무시됨.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def get_db():
    """요청마다 세션 열고, 끝나면 자동 정리 (FastAPI 의존성 주입용)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
