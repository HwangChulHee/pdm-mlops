"""최근 학습한 CNN run을 MLflow Model Registry에 champion으로 등록."""
import warnings
warnings.filterwarnings("ignore")
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "pdm-rul-cnn"


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()

    exp = client.get_experiment_by_name("pdm-rul")
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="params.model = 'cnn'",
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit("cnn run을 못 찾음. train.py --model cnn 먼저 돌려.")

    run_id = runs[0].info.run_id
    rmse = runs[0].data.metrics.get("test_rmse")
    # state_dict를 log_artifact로 올렸으므로(LoggedModel 아님) runs:/{id}/model 대신
    # run의 실제 artifact URI를 source로 등록한다. mlflow 3.x에서 runs:/ URI는
    # LoggedModel을 찾으려다 실패한다.
    model_uri = f"{runs[0].info.artifact_uri}/model"

    mv = mlflow.register_model(model_uri, MODEL_NAME)        # 선반에 올리기
    client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)  # 현역 표시

    print(f"등록 완료: {MODEL_NAME} v{mv.version} (champion)")
    print(f"  run_id: {run_id}")
    print(f"  test_rmse: {rmse:.2f}")
    print(f"  로드 주소: models:/{MODEL_NAME}@champion")


if __name__ == '__main__':
    main()
