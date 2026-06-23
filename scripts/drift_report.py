"""데이터 드리프트 리포트 (Evidently 0.7.x).

서사: 모델은 FD001(운영조건 1개)로 학습됐다. FD002(운영조건 6개)는 센서 분포가
근본적으로 다르므로, 학습 분포(FD001 train)를 reference로, 운영 유입(FD002)을
current로 두면 진짜 데이터 드리프트가 감지된다. 인위적 시뮬레이션이 아니라
데이터셋 구조가 품은 드리프트다.

대조군(negative control): FD001 train을 무작위로 반 갈라 비교한다. 같은 분포이므로
드리프트가 거의 없어야 한다(탐지기의 오탐 baseline). FD001 train vs FD001 *test*는
test 궤적이 고장 전 잘려 있어(truncated) 같은 운영조건인데도 분포가 어긋나므로
깨끗한 대조군이 못 된다 -> 같은 분포 분할을 대조군으로 쓴다.

비교 대상은 센서 15개의 원시값 분포(전처리에서 쓰는 feature_cols 그대로).

HTML 리포트와 함께 요약 수치를 reports/drift_summary.json으로도 떨군다.
드리프트 재계산은 무거우므로 에이전트 도구(check_drift)는 이 JSON만 읽는다.
"""
import json
from pathlib import Path

from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset

from pdm.data.preprocess import load_raw, feature_cols

FEATS = feature_cols()                       # dead 센서 제외한 15개
DATA_DEF = DataDefinition(numerical_columns=FEATS)
OUT = Path("reports")


def _dataset(df):
    return Dataset.from_pandas(df[FEATS], data_definition=DATA_DEF)


def run_report(ref_df, cur_df, html_name, label):
    """reference vs current 드리프트 리포트 생성 + 드리프트 feature 수 반환."""
    ref_ds = _dataset(ref_df)
    cur_ds = _dataset(cur_df)
    result = Report([DataDriftPreset()]).run(reference_data=ref_ds, current_data=cur_ds)

    OUT.mkdir(parents=True, exist_ok=True)
    html_path = OUT / html_name
    result.save_html(str(html_path))

    # 요약 메트릭(DriftedColumnsCount)에서 드리프트된 feature 수/비율 추출
    summary = result.dict()["metrics"][0]["value"]
    count, share = int(summary["count"]), summary["share"]
    print(f"[{label}] 드리프트 감지된 feature: {count}/{len(FEATS)} "
          f"(share {share:.0%})  -> {html_path}")
    return count


if __name__ == "__main__":
    print("=== 데이터 드리프트 리포트 (센서 15개 원시값 분포) ===")
    fd001 = load_raw("data/raw/train_FD001.txt")
    fd002 = load_raw("data/raw/train_FD002.txt")

    # 메인: 학습 분포(FD001) vs 운영 유입(FD002) -> 드리프트 감지돼야 정상
    drift = run_report(fd001, fd002, "drift_fd001_vs_fd002.html", "FD001 vs FD002")

    # 대조군: FD001 train을 무작위 반분할 -> 같은 분포라 드리프트 거의 없어야 정상
    half = fd001.sample(frac=0.5, random_state=42)
    rest = fd001.drop(half.index)
    control = run_report(half, rest, "drift_fd001_split_control.html",
                         "대조: FD001 무작위 반분할")

    # 에이전트 도구가 읽을 요약 JSON (재계산 회피)
    summary = {
        "drifted_features": drift,
        "total": len(FEATS),
        "control_drifted": control,
        "reference": "FD001 train (운영조건 1개, 모델 학습 분포)",
        "current": "FD002 train (운영조건 6개, 운영 유입 가정)",
        "report_html": "reports/drift_fd001_vs_fd002.html",
    }
    (OUT / "drift_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n해석:")
    print(f"  - FD001->FD002: {drift}/{len(FEATS)} 드리프트 (운영조건 변화로 분포 어긋남)")
    print(f"  - 대조군:       {control}/{len(FEATS)} 드리프트 (같은 분포, 낮아야 정상)")
    if drift > control:
        print("  => 시나리오 정상: 운영조건이 다른 FD002에서 드리프트가 뚜렷이 더 크다.")
    print(f"  - 요약 JSON -> {OUT / 'drift_summary.json'}")
