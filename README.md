# pdm-mlops

CMAPSS 터보팬 엔진 RUL(잔여수명) 예측 MLOps 파이프라인.
전처리 → 학습(MLflow 추적) → Model Registry(champion) → FastAPI 추론 API.

## Docker Compose로 전체 기동

세 서비스를 한 번에 띄운다:

- **postgres** — 예측 결과 DB(`predictions`) + MLflow 백엔드 DB(`mlflow`) 겸용
- **mlflow** — MLflow tracking 서버 (`--serve-artifacts`로 artifact 프록시)
- **api** — FastAPI 추론 서버 (MLflow 서버에서 champion 모델 HTTP 로드)

> 설계 핵심: MLflow를 sqlite 파일이 아니라 **서버**로 운영한다. api는
> `MLFLOW_TRACKING_URI=http://mlflow:5000`으로 champion을 받으므로, registry의
> artifact source가 호스트 절대경로가 아니라 `mlflow-artifacts:/` URI가 되어
> 컨테이너 안에서도 그대로 동작한다(절대경로 문제 해결).

```bash
cp .env.example .env          # 비밀번호 설정 (기본값 그대로도 동작)

# 1) postgres + mlflow 먼저 기동
docker compose up -d postgres mlflow

# 2) 모델 학습·등록 (새 postgres 백엔드엔 모델이 없으므로 재등록 필요)
bash scripts/seed_model.sh    # = preprocess + train cnn + register champion

# 3) api 기동 (champion 로드)
docker compose up -d api
```

확인:

```bash
curl http://localhost:8000/health        # {"status":"ok","model_loaded":true}
curl http://localhost:5000               # MLflow UI

# 예측
WINDOW=$(uv run python -c "import numpy as np,json; print(json.dumps(np.load('data/processed/X_test.npy')[0].tolist()))")
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"window\": $WINDOW}"

curl http://localhost:8000/predictions   # postgres에 저장된 예측 이력
```

초기화 후 재현:

```bash
docker compose down -v        # 볼륨까지 삭제
docker compose up -d postgres mlflow && bash scripts/seed_model.sh && docker compose up -d api
```

## 모니터링 (운영 메트릭 + 데이터 드리프트)

### 운영 메트릭 — Prometheus + Grafana
`docker compose up`에 prometheus(9090)·grafana(3000)가 포함된다.

- api가 `/metrics`로 4가지를 노출: **요청수, 응답지연, 예측 RUL 분포(ML 특화), 에러율**.
  표준 HTTP 메트릭은 `prometheus-fastapi-instrumentator`가, RUL 분포(`rul_prediction`
  Histogram)는 predict에서 직접 기록.
- **Grafana**: http://localhost:3000 (기본 `admin`/`admin`). Prometheus 데이터소스와
  대시보드 **"PdM RUL API Monitoring"**(요청수/지연 p95/RUL 분포/에러율 4패널)가
  자동 프로비저닝됨.
- **Prometheus**: http://localhost:9090 (api:8000/metrics를 5초마다 scrape).

```bash
curl http://localhost:8000/metrics | grep rul_prediction      # 커스텀 RUL 히스토그램
curl "http://localhost:9090/api/v1/query?query=rul_prediction_count"
```

### 데이터 드리프트 — Evidently
모델은 **FD001**(운영조건 1개)로 학습됨. **FD002**(운영조건 6개)는 센서 분포가
근본적으로 달라, FD001(reference) vs FD002(current) 비교 시 진짜 드리프트가 잡힌다.

```bash
uv run python scripts/drift_report.py
# -> reports/drift_fd001_vs_fd002.html  (15/15 feature 드리프트)
# -> reports/drift_fd001_split_control.html  (대조군: 같은 분포 반분할, 0/15)
```

## 로컬 실행 (Docker 없이)

`DATABASE_URL`·`MLFLOW_TRACKING_URI` 환경변수의 기본값이 로컬(sqlite/파일)이라
코드 변경 없이 그대로 돌아간다:

```bash
uv sync
uv run python src/pdm/data/preprocess.py
uv run python -m pdm.models.train --model cnn
uv run python -m pdm.models.register
uv run uvicorn pdm.api.main:app --reload
```
