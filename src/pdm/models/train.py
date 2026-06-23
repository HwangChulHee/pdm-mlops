"""공통 학습 루프 + MLflow 추적. --model 로 모델 선택.

모델 저장은 state_dict 방식(torch.save -> mlflow.log_artifact)을 쓴다.
mlflow.pytorch.log_model은 pt2(torch.export) 추적이 기본이라 배치 차원 고정
때문에 ConstraintViolationError로 실패한다. 쓰지 말 것.
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import tempfile
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import mlflow
from pdm.models.models import MODELS

# --- 튜닝 가능한 값 ---
EPOCHS = 30
BATCH = 256
LR = 1e-3
WINDOW = 30
RUL_CLIP = 125
N_FEATURES = 15


def load(split):
    X = np.load(f'data/processed/X_{split}.npy').astype('float32')
    y = np.load(f'data/processed/y_{split}.npy').astype('float32')
    return torch.tensor(X), torch.tensor(y)


def rmse(pred, true):
    return torch.sqrt(((pred - true) ** 2).mean()).item()


def log_state_dict(model):
    """state_dict를 임시 파일로 저장 후 artifact_path='model' 아래로 올린다.
    현재 디렉토리를 오염시키지 않도록 tempfile 사용."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model_state.pt")
        torch.save(model.state_dict(), path)
        mlflow.log_artifact(path, artifact_path="model")


def main(model_name):
    Xtr, ytr = load('train')
    Xte, yte = load('test')
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=True)

    model = MODELS[model_name]()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossfn = nn.MSELoss()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("pdm-rul")
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({
            "model": model_name, "epochs": EPOCHS, "batch": BATCH,
            "lr": LR, "window": WINDOW, "rul_clip": RUL_CLIP,
            "n_features": N_FEATURES,
        })
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                lossfn(model(xb), yb).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                tr, te = rmse(model(Xtr), ytr), rmse(model(Xte), yte)
            mlflow.log_metric("train_rmse", tr, step=epoch)
            mlflow.log_metric("test_rmse", te, step=epoch)
            if epoch in (1, 10, 20, 30):
                print(f"[{model_name}] epoch {epoch:>3} | train {tr:5.2f} | test {te:5.2f}")
        log_state_dict(model)
        print(f"[{model_name}] 최종 test RMSE: {te:.2f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=MODELS.keys(), default='cnn')
    main(p.parse_args().model)
