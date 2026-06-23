"""CMAPSS 전처리: txt -> 슬라이딩 윈도우 텐서 (X, y), npy로 저장."""
import numpy as np
import pandas as pd
from pathlib import Path

COLS = ['unit', 'cycle'] + [f'set{i}' for i in range(1, 4)] + [f's{i}' for i in range(1, 22)]
DEAD_SENSORS = ['s1', 's5', 's10', 's16', 's18', 's19']  # std=0, 정보 없음
RUL_CLIP = 125
WINDOW = 30


def load_raw(path):
    df = pd.read_csv(path, sep=r'\s+', header=None).iloc[:, :26]
    df.columns = COLS
    return df


def add_rul(df):
    df = df.copy()
    max_cycle = df.groupby('unit')['cycle'].transform('max')
    df['RUL'] = (max_cycle - df['cycle']).clip(upper=RUL_CLIP)
    return df


def feature_cols():
    return [f's{i}' for i in range(1, 22) if f's{i}' not in DEAD_SENSORS]


def fit_scaler(df, feats):
    return df[feats].min(), df[feats].max()


def apply_scaler(df, feats, mn, mx):
    df = df.copy()
    df[feats] = (df[feats] - mn) / (mx - mn + 1e-8)
    return df


def make_windows(df, feats, window=WINDOW):
    X, y = [], []
    for _, g in df.groupby('unit'):
        g = g.sort_values('cycle')
        arr, rul = g[feats].values, g['RUL'].values
        for i in range(len(g) - window + 1):
            X.append(arr[i:i + window])
            y.append(rul[i + window - 1])
    return np.array(X), np.array(y)


def make_test_windows(df, feats, window=WINDOW):
    """test는 엔진별 '마지막 윈도우 1개'만. 실전에선 최신 상태로 예측하니까."""
    X, last_cycle = [], []
    for _, g in df.groupby('unit'):
        g = g.sort_values('cycle')
        arr = g[feats].values
        if len(g) >= window:
            X.append(arr[-window:])          # 끝에서 30사이클
        else:                                 # 30보다 짧으면 앞을 0으로 패딩
            pad = np.zeros((window - len(g), len(feats)))
            X.append(np.vstack([pad, arr]))
        last_cycle.append(len(g))
    return np.array(X), np.array(last_cycle)


if __name__ == '__main__':
    out = Path('data/processed')
    out.mkdir(parents=True, exist_ok=True)
    feats = feature_cols()

    # --- train: 정규화 기준(min/max)을 여기서 정함 ---
    tr = add_rul(load_raw('data/raw/train_FD001.txt'))
    mn, mx = fit_scaler(tr, feats)
    tr = apply_scaler(tr, feats, mn, mx)
    Xtr, ytr = make_windows(tr, feats)

    # --- test: train의 min/max를 그대로 적용 (누수 방지) ---
    te = load_raw('data/raw/test_FD001.txt')
    te = apply_scaler(te, feats, mn, mx)
    Xte, _ = make_test_windows(te, feats)
    yte = np.loadtxt('data/raw/RUL_FD001.txt').clip(max=RUL_CLIP)  # 정답 RUL

    np.save(out / 'X_train.npy', Xtr)
    np.save(out / 'y_train.npy', ytr)
    np.save(out / 'X_test.npy', Xte)
    np.save(out / 'y_test.npy', yte)

    print(f"train X: {Xtr.shape}, y: {ytr.shape}")
    print(f"test  X: {Xte.shape}, y: {yte.shape}")
    print(f"test 엔진 수: {Xte.shape[0]} (RUL 정답 개수와 같아야: {len(yte)})")
    print(f"저장 완료 -> {out}/")
