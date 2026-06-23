# pdm-mlops — 예지보전 RUL 예측 MLOps 파이프라인

NASA CMAPSS 터보팬 엔진 데이터로 **잔여수명(RUL) 예측 모델을 학습부터 서빙·모니터링·에이전트까지** 운영 가능한 형태로 구현한 end-to-end MLOps 프로젝트.

> "노트북에서 도는 모델"이 아니라 "운영 환경에 올라가는 시스템"을 목표로 했다.
> `docker compose up` 한 번으로 학습 추적·추론 API·DB·모니터링·드리프트 감지가 전부 기동된다.

---

## 무엇을 푸는가

설비(터보팬 엔진)가 **언제 고장날지** 미리 예측하는 예지보전(Predictive Maintenance) 문제다. 각 엔진의 센서 시계열을 입력받아 **RUL(고장까지 남은 사이클)** 을 회귀로 예측한다.

- **입력**: 센서 15개 × 최근 30사이클 윈도우
- **출력**: RUL 숫자 (0~125 사이클)
- **데이터**: NASA CMAPSS FD001 (엔진 100대, 각 128~362사이클)

예측이 맞으면 *고장 직전*에 정비할 수 있다 — 너무 일찍(낭비) 도, 너무 늦게(사고) 도 아니게.

---

## 아키텍처

```
데이터(CMAPSS)
   │  전처리: 슬라이딩 윈도우 + RUL clip + 죽은 센서 제거
   ▼
모델 학습 ──MLflow 추적──> Model Registry (champion alias)
   │                              │
   ▼                              ▼ HTTP 로드
FastAPI 추론 API ◀──────── champion 모델
   │  POST /predict → RUL
   ▼
PostgreSQL (예측 이력, SQLAlchemy ORM)
   │
   ├──> Prometheus (요청수·지연·RUL분포·에러율) ──> Grafana 대시보드
   ├──> Evidently 드리프트 리포트 (FD001 vs FD002)
   │
   ▼
LangGraph 에이전트 (gpt-5.4-mini) ── 위 전부를 도구로 호출해 자연어 질의응답
```

전체 스택은 Docker Compose로 컨테이너화: `postgres + mlflow서버 + api + prometheus + grafana`.

---

## 기술 스택

| 영역 | 사용 |
|---|---|
| 모델 | PyTorch (1D-CNN, MLP) |
| 실험 추적 / 레지스트리 | MLflow (postgres backend, champion alias) |
| 서빙 | FastAPI (lifespan 모델 로드, Pydantic 검증) |
| DB | PostgreSQL + SQLAlchemy ORM |
| 컨테이너 | Docker, Docker Compose (멀티스테이지 빌드, uv) |
| 모니터링 | Prometheus, Grafana, Evidently |
| 에이전트 | LangGraph (StateGraph), gpt-5.4-mini |
| 패키지/환경 | uv (src-layout, uv.lock 버전 고정) |

---

## 설계 결정과 근거

이 프로젝트의 핵심은 *왜 이렇게 했는가*다. 주요 결정과 이유:

### 데이터 / 모델
- **죽은 센서 6개 제거** — 센서 21개 중 6개(`s1,s5,s10,s16,s18,s19`)는 표준편차 0(상수). FD001은 운영조건이 1개뿐이라 변할 일이 없어 정보가 없다 → 입력에서 제외(15개 사용).
- **슬라이딩 윈도우(30사이클)** — 센서값은 노이즈가 커서 단일 시점으론 추세를 못 본다. 30사이클을 묶어 흐름으로 입력. test 최소 길이(31) + RUL 논문 관행을 고려한 값.
- **RUL clip = 125** — 새 엔진은 degradation 신호가 거의 없는데 RUL 라벨만 크면 모델이 혼란스럽다. 상한을 씌워 "고장 임박" 구간 예측에 집중시킴.
- **1D-CNN을 베이스라인으로** — 작은 데이터·CPU 환경에서 빠르고 과적합에 강함. "SOTA 무조건 적용"이 아니라 문제에 맞는 모델 선택. MLP(순서 정보 버린 대조군)와 비교해 구조의 효과를 검증.

### 서빙 / 인프라
- **MLflow Model Registry + champion alias** — API는 run_id가 아니라 `models:/pdm-rul-cnn@champion`만 안다. 모델을 교체해도 API 코드는 그대로(운영과 모델 개발 분리).
- **MLflow를 파일이 아니라 "서버"로 운영** — 로컬 sqlite 파일 방식은 registry artifact source가 호스트 절대경로로 박혀 컨테이너에서 깨진다. `--serve-artifacts` 서버로 띄워 `mlflow-artifacts:/` URI로 통일 → 컨테이너 어디서든 동작.
- **환경변수로 로컬↔Docker 무코드 전환** — `DATABASE_URL`/`MLFLOW_TRACKING_URI` 기본값은 로컬(sqlite/파일), Compose에선 postgres/서버 주입. 코드 변경 0.

### 모니터링
- **메트릭 4종만** — 요청수·지연(p95)·**예측 RUL 분포**·에러율. RUL 분포가 ML 특화 메트릭: 예측이 한 값에 쏠리면 모델 이상 신호.
- **드리프트 시나리오 = FD001 vs FD002** — 모델은 FD001(운영조건 1개)로 학습. FD002(운영조건 6개)는 센서 분포가 근본적으로 다르다 → 15/15 feature 드리프트 감지. 동일분포 대조군(FD001 반분할)은 0/15 → 탐지기가 오탐하지 않음을 검증.

### 에이전트
- **LangGraph StateGraph 직접 구성** (prebuilt 미사용) — 단순 ReAct 루프에 더해, 드리프트가 심각하면 추가 진단을 거치는 **조건부 분기(`deep_check`)** 를 둠. 도메인 조건으로 그래프가 라우팅되는 걸 보임.

---

## 빠른 시작

```bash
cp .env.example .env          # POSTGRES_*, OPENAI_API_KEY 설정

# 1) postgres + mlflow 기동
docker compose up -d postgres mlflow

# 2) 모델 학습·등록 (postgres 백엔드에 champion 생성)
bash scripts/seed_model.sh    # preprocess + train + register

# 3) 나머지 전체 기동
docker compose up -d
```

확인:
```bash
curl http://localhost:8000/health         # {"status":"ok","model_loaded":true}
# Grafana   http://localhost:3000  (admin/admin) — "PdM RUL API Monitoring" 대시보드
# MLflow UI http://localhost:5000
# Prometheus http://localhost:9090
```

예측 / 에이전트:
```bash
# 예측
WINDOW=$(uv run python -c "import numpy as np,json; print(json.dumps(np.load('data/processed/X_test.npy')[0].tolist()))")
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"window\": $WINDOW}"

# 드리프트 리포트 생성
uv run python scripts/drift_report.py     # reports/*.html + drift_summary.json

# 에이전트 (multi-step + 조건부 분기)
uv run python -m pdm.agent.agent "지금 이 시스템 신뢰할 만해?"
```

초기화 후 재현:
```bash
docker compose down -v
docker compose up -d postgres mlflow && bash scripts/seed_model.sh && docker compose up -d
```

---

## 프로젝트 구조

```
src/pdm/
  data/preprocess.py      전처리: CMAPSS txt → 슬라이딩 윈도우 텐서
  models/
    models.py             모델 정의 (MLP, CNN1D, CNNLSTM)
    train.py              학습 + MLflow 추적
    register.py           champion alias 등록
  api/
    main.py               FastAPI (lifespan, /predict, /predictions, /health, /metrics)
    model.py              champion 모델 로드 + 추론(0~125 클램프)
    schemas.py            Pydantic 입출력 검증
    db.py / models_db.py  SQLAlchemy 세션 + Prediction ORM
  agent/
    tools.py              도구 4개 (LangChain @tool)
    agent.py              LangGraph StateGraph (조건부 분기 포함)
monitoring/               prometheus.yml, grafana 대시보드/프로비저닝
scripts/                  seed_model.sh, drift_report.py, init-db.sql
compose.yaml              전체 스택
docs/agent_graph.md       에이전트 그래프 (mermaid)
```

---

## 로컬 실행 (Docker 없이)

환경변수 기본값이 로컬(sqlite/파일)이라 그대로 동작:
```bash
uv sync
uv run python src/pdm/data/preprocess.py
uv run python -m pdm.models.train --model cnn
uv run python -m pdm.models.register
uv run uvicorn pdm.api.main:app --reload
```

---

## 알려진 한계 / 향후

- **모델 성능**: 베이스라인 단계. 예측이 RUL 상한(125) 근처로 쏠리는 경향이 있어, 실제 "곧 고장" 구간(RUL≤25) 예측이 약하다. → clip 조정, 비대칭 손실(늦은 예측에 더 큰 페널티), early stopping, CNN-LSTM/Transformer 비교가 다음 과제.
- **에이전트**: tool calling + multi-step 워크플로우 수준(자율 재학습 트리거는 범위 외).
- **배포**: 현재 Docker Compose. K8s(minikube) 매니페스트 전환이 다음 단계.
```