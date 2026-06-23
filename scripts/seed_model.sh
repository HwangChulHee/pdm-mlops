#!/usr/bin/env bash
# 새로 띄운 postgres-backed MLflow 서버에 champion 모델을 재등록한다.
# 새 mlflow 백엔드(postgres)에는 기존 sqlite의 모델이 없으므로, 스택을 띄운 뒤
# 이 스크립트로 학습→등록을 다시 해야 api가 champion을 찾을 수 있다.
#
# 사용법:  bash scripts/seed_model.sh   (compose 스택이 떠 있는 상태에서, repo 루트)
set -euo pipefail
cd "$(dirname "$0")/.."

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
echo ">> MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI"

# 전처리 결과 없으면 생성 (gitignore라 새 환경엔 없음)
if [ ! -f data/processed/X_test.npy ]; then
  echo ">> data/processed 없음 -> preprocess 실행"
  uv run python src/pdm/data/preprocess.py
fi

echo ">> cnn 학습 (서버에 추적/아티팩트 기록)"
uv run python -m pdm.models.train --model cnn

echo ">> champion 등록"
uv run python -m pdm.models.register

echo ">> 완료. models:/pdm-rul-cnn@champion 등록됨"
