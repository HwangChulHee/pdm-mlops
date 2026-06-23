-- postgres 1개 인스턴스에 DB 2개: mlflow(백엔드) + predictions(예측 결과).
-- POSTGRES_DB=mlflow 로 mlflow DB는 엔트리포인트가 만들고, 여기서 predictions 추가.
CREATE DATABASE predictions;
