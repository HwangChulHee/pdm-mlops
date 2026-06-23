"""PdM RUL 예측 추론 API + 예측 이력 저장."""
import warnings
warnings.filterwarnings("ignore")
from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select
from pdm.api.model import rul_model, MODEL_ALIAS
from pdm.api.schemas import PredictRequest, PredictResponse, HealthResponse
from pdm.api.db import engine, Base, get_db
from pdm.api.models_db import Prediction


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)  # 테이블 없으면 생성
    rul_model.load()
    yield


app = FastAPI(title="PdM RUL Prediction API", version="0.2.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=rul_model.ready)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, db: Session = Depends(get_db)):
    rul = rul_model.predict(req.window)
    row = Prediction(
        rul=rul,
        model_version=MODEL_ALIAS,
        window_mean=float(np.mean(req.window)),
    )
    db.add(row)
    db.commit()
    return PredictResponse(rul=rul)


@app.get("/predictions")
def list_predictions(limit: int = 10, db: Session = Depends(get_db)):
    rows = db.scalars(
        select(Prediction).order_by(Prediction.id.desc()).limit(limit)
    ).all()
    return [
        {"id": r.id, "created_at": r.created_at, "rul": r.rul,
         "window_mean": r.window_mean, "model_version": r.model_version}
        for r in rows
    ]
