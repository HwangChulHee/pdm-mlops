"""Champion 모델 로드 + 추론 래퍼.

서버 시작 시 1회 로드해서 메모리에 상주(lifespan). 매 요청마다
디스크에서 읽지 않는다.
"""
import os
import mlflow
import numpy as np
import torch
from mlflow.artifacts import download_artifacts
from pdm.models.models import CNN1D

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_ALIAS = "models:/pdm-rul-cnn@champion"
RUL_MAX = 125.0


class RULModel:
    def __init__(self):
        self._model = None

    def load(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        path = download_artifacts(MODEL_ALIAS)
        state = torch.load(os.path.join(path, "model_state.pt"), weights_only=True)
        m = CNN1D()
        m.load_state_dict(state)
        m.eval()
        self._model = m

    @property
    def ready(self) -> bool:
        return self._model is not None

    def predict(self, window) -> float:
        x = torch.tensor(np.array(window, dtype="float32")).unsqueeze(0)  # (1,30,15)
        with torch.no_grad():
            raw = self._model(x).item()
        return max(0.0, min(RUL_MAX, raw))  # 도메인 제약: RUL은 0~125


rul_model = RULModel()  # 앱 전체가 공유하는 단일 인스턴스
