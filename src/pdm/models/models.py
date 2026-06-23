"""RUL 예측 모델 3종: MLP(대조군), 1D-CNN(베이스라인), CNN-LSTM(하이브리드)."""
import torch.nn as nn


class MLP(nn.Module):
    """윈도우를 일자로 펼침 -> 순서 정보 버림 (대조군)."""
    def __init__(self, window=30, n_features=15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window * n_features, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CNN1D(nn.Module):
    """국소 패턴 스캔 (베이스라인)."""
    def __init__(self, n_features=15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2)).squeeze(-1)


class CNNLSTM(nn.Module):
    """CNN으로 패턴 뽑고 -> LSTM으로 흐름을 순서대로 읽음 (하이브리드)."""
    def __init__(self, n_features=15):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
        )
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv(x.transpose(1, 2))   # (b, 64, time)
        x = x.transpose(1, 2)              # (b, time, 64)
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)  # 마지막 시점의 hidden


MODELS = {"mlp": MLP, "cnn": CNN1D, "cnnlstm": CNNLSTM}
