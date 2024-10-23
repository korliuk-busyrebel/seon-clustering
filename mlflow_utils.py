import mlflow
import mlflow.sklearn


# Start an MLflow run and log parameters
def start_mlflow_run(experiment_name: str, params: dict):
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run()

    # Log the provided parameters
    for key, value in params.items():
        mlflow.log_param(key, value)

    return run


# Log metrics such as silhouette score, calinski_harabasz_score, and davies_bouldin_score
def log_mlflow_metrics(metrics: dict):
    for key, value in metrics.items():
        mlflow.log_metric(key, value)


# Log a scikit-learn model to MLflow
def log_model_to_mlflow(model, model_name: str, input_example=None):
    mlflow.sklearn.log_model(model, model_name, input_example=input_example)


# End an MLflow run
def end_mlflow_run():
    mlflow.end_run()
