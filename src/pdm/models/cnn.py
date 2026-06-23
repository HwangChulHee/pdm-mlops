"""베이스라인 1D-CNN + MLflow 실험 추적."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import mlflow
import mlflow.pytorch

# --- 튜닝 가능한 값 ---
EPOCHS = 30
BATCH = 256
LR = 1e-3
WINDOW = 30
RUL_CLIP = 125


class CNN1D(nn.Module):
    def __init__(self, n_features=15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.net(x).squeeze(-1)


def load(split):
    X = np.load(f'data/processed/X_{split}.npy').astype('float32')
    y = np.load(f'data/processed/y_{split}.npy').astype('float32')
    return torch.tensor(X), torch.tensor(y)


def rmse(pred, true):
    return torch.sqrt(((pred - true) ** 2).mean()).item()


def main():
    Xtr, ytr = load('train')
    Xte, yte = load('test')
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=True)

    model = CNN1D()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossfn = nn.MSELoss()

    mlflow.set_experiment("pdm-rul")
    with mlflow.start_run(run_name="cnn1d-baseline"):
        mlflow.log_params({
            "model": "1D-CNN", "epochs": EPOCHS, "batch": BATCH,
            "lr": LR, "window": WINDOW, "rul_clip": RUL_CLIP, "n_features": 15,
        })

        for epoch in range(1, EPOCHS + 1):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss = lossfn(model(xb), yb)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                tr_rmse, te_rmse = rmse(model(Xtr), ytr), rmse(model(Xte), yte)
            mlflow.log_metric("train_rmse", tr_rmse, step=epoch)
            mlflow.log_metric("test_rmse", te_rmse, step=epoch)
            if epoch == 1 or epoch % 5 == 0:
                print(f"epoch {epoch:>3} | train RMSE {tr_rmse:5.2f} | test RMSE {te_rmse:5.2f}")

        mlflow.pytorch.log_model(model, artifact_path="model")
        print(f"\n최종 test RMSE: {te_rmse:.2f} 사이클 (MLflow에 기록됨)")


if __name__ == '__main__':
    main()
