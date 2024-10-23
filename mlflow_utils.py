import mlflow
import mlflow.sklearn

def start_mlflow_run(experiment_name: str, params: dict):
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run()
    for key, value in params.items():
        mlflow.log_param(key, value)
    return run

def log_mlflow_metrics(metrics: dict):
    for key, value in metrics.items():
        mlflow.log_metric(key, value)

def log_model_to_mlflow(model, model_name: str, input_example=None):
    mlflow.sklearn.log_model(model, model_name, input_example=input_example)

def log_metrics_and_model(model, model_name: str, input_example, metrics: dict):
    log_mlflow_metrics(metrics)
    log_model_to_mlflow(model, model_name, input_example)

def end_mlflow_run():
    mlflow.end_run()
