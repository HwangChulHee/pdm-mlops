"""TestClient로 API + DB 저장 smoke test."""
import numpy as np
from fastapi.testclient import TestClient
from pdm.api.main import app

with TestClient(app) as client:
    print("health:", client.get("/health").json())

    x = np.load("data/processed/X_test.npy")[0].tolist()
    r1 = client.post("/predict", json={"window": x}).json()
    r2 = client.post("/predict", json={"window": x}).json()
    print("predict x2:", r1, r2)

    hist = client.get("/predictions").json()
    print(f"저장된 예측 {len(hist)}건, 최근:", hist[0] if hist else None)

    bad = [[0.0] * 15] * 10
    print("잘못된 입력:", client.post("/predict", json={"window": bad}).status_code)
