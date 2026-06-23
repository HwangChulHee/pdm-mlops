# ---- build stage: uv로 의존성 설치 ----
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# 의존성만 먼저 설치 (레이어 캐시: 코드 바뀌어도 의존성 재설치 안 함)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# 프로젝트 코드 복사 후 패키지 설치
COPY README.md ./README.md
COPY src ./src
RUN uv sync --frozen --no-dev

# ---- runtime stage: 실행에 필요한 것만 ----
FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app
# build stage의 가상환경 + 코드만 가져옴 (빌드 도구는 안 가져옴 = 가벼움)
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000
CMD ["uvicorn", "pdm.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
