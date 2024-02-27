pull-mlflow-image:
	docker pull ghcr.io/mlflow/mlflow:v2.10.2

run-mlflow-backend:
	docker run -v ./mlruns:/mlruns -p 8000:8000 ghcr.io/mlflow/mlflow:v2.10.2 mlflow ui --port 8000 --host 0.0.0.0 &