"""에이전트 도구 4종. 모두 기존 시스템(API/Prometheus/드리프트 요약)을 호출하는
얇은 래퍼다. 에이전트가 추측 대신 실제 데이터를 근거로 답하게 하는 게 목적.

API/Prometheus 주소는 환경변수로 두어 나중에 컨테이너화해도 코드 변경이 없게 한다.
"""
import json
import os
from pathlib import Path

import httpx

API_URL = os.getenv("API_URL", "http://localhost:8000")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
DRIFT_SUMMARY_PATH = os.getenv("DRIFT_SUMMARY_PATH", "reports/drift_summary.json")

_TIMEOUT = 10.0


def predict_rul(window: list[list[float]]) -> dict:
    """30x15 센서 윈도우로 RUL을 예측한다(API /predict 호출). 결과는 DB에도 저장됨."""
    r = httpx.post(f"{API_URL}/predict", json={"window": window}, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_recent_predictions(limit: int = 10) -> list:
    """최근 예측 이력을 DB에서 조회한다(API /predictions 호출)."""
    r = httpx.get(f"{API_URL}/predictions", params={"limit": limit}, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _prom_query(query: str):
    """Prometheus instant query -> 결과 벡터(list) 또는 None(실패 시)."""
    try:
        r = httpx.get(f"{PROMETHEUS_URL}/api/v1/query",
                      params={"query": query}, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "success":
            return data["data"]["result"]
    except Exception:
        return None
    return None


def get_metrics_summary() -> dict:
    """운영 메트릭 요약: 총 예측 수, p95 지연(초), 예측 RUL 분포(버킷별 누적).
    쿼리 일부가 실패해도 가능한 부분만 채워서 반환한다."""
    summary: dict = {}

    cnt = _prom_query("rul_prediction_count")
    if cnt:
        summary["total_predictions"] = float(cnt[0]["value"][1])

    p95 = _prom_query(
        'histogram_quantile(0.95, sum(rate('
        'http_request_duration_seconds_bucket{handler="/predict"}[5m])) by (le))')
    if p95 and p95[0]["value"][1] not in ("NaN", "+Inf"):
        summary["predict_latency_p95_seconds"] = round(float(p95[0]["value"][1]), 4)

    buckets = _prom_query("sum(rul_prediction_bucket) by (le)")
    if buckets:
        dist = {b["metric"]["le"]: int(float(b["value"][1])) for b in buckets}
        # le 오름차순 정렬 (+Inf 마지막)
        summary["rul_distribution_cumulative"] = dict(
            sorted(dist.items(), key=lambda kv: float(kv[0])))

    if not summary:
        return {"error": "Prometheus에서 메트릭을 가져오지 못함. 스택이 떠 있는지 확인."}
    return summary


def check_drift() -> dict:
    """미리 계산된 데이터 드리프트 요약을 읽는다(reports/drift_summary.json).
    드리프트 재계산은 무거우므로 scripts/drift_report.py가 떨군 요약만 읽는다."""
    p = Path(DRIFT_SUMMARY_PATH)
    if not p.exists():
        return {"error": f"{DRIFT_SUMMARY_PATH} 없음. 먼저 "
                         f"`uv run python scripts/drift_report.py` 실행 필요."}
    return json.loads(p.read_text(encoding="utf-8"))


# --- OpenAI function-calling 스키마 + 디스패치 테이블 ---

TOOLS = {
    "predict_rul": predict_rul,
    "get_recent_predictions": get_recent_predictions,
    "get_metrics_summary": get_metrics_summary,
    "check_drift": check_drift,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "predict_rul",
            "description": "30사이클 x 15센서 윈도우로 엔진 잔여수명(RUL, 0~125)을 "
                           "예측한다. 결과는 예측 이력 DB에도 저장된다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "window": {
                        "type": "array",
                        "description": "30개 행, 각 행은 센서값 15개인 2차원 배열",
                        "items": {"type": "array", "items": {"type": "number"}},
                    }
                },
                "required": ["window"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_predictions",
            "description": "최근 예측 이력을 DB에서 조회한다(id, 시각, rul, 모델버전, "
                           "입력 윈도우 평균).",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "가져올 건수 (기본 10)"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_metrics_summary",
            "description": "운영 메트릭 요약을 Prometheus에서 가져온다: 총 예측 수, "
                           "predict p95 지연(초), 예측 RUL 분포(버킷별 누적 카운트).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_drift",
            "description": "데이터 드리프트 요약을 반환한다: 드리프트된 feature 수/전체, "
                           "대조군 드리프트 수. 모델은 FD001로 학습, FD002 유입 시 비교.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]
