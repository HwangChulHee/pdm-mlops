"""API 입출력 스키마 + 입력 형태 검증."""
from pydantic import BaseModel, Field, field_validator

WINDOW = 30
N_FEATURES = 15


class PredictRequest(BaseModel):
    window: list[list[float]] = Field(..., description=f"{WINDOW}사이클 x {N_FEATURES}센서")

    @field_validator("window")
    @classmethod
    def check_shape(cls, v):
        if len(v) != WINDOW:
            raise ValueError(f"window는 {WINDOW}행이어야 함 (받음: {len(v)})")
        for i, row in enumerate(v):
            if len(row) != N_FEATURES:
                raise ValueError(f"행 {i}: 센서 {N_FEATURES}개여야 함 (받음: {len(row)})")
        return v


class PredictResponse(BaseModel):
    rul: float = Field(..., description="예측 잔여수명 (사이클, 0~125)")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
