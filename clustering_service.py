from mlflow_utils import start_mlflow_run, log_mlflow_metrics, log_model_to_mlflow, end_mlflow_run
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pandas as pd


async def create_clusters(file):
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))

    # Preprocess data, find optimal eps and min_samples, etc.
    # Example of parameters to log
    params = {
        "eps": optimal_eps,
        "min_samples": optimal_min_samples
    }

    # Start the MLflow run
    run = start_mlflow_run("clustering_experiment", params)

    # Perform clustering
    clustering_model = DBSCAN(eps=optimal_eps, min_samples=optimal_min_samples)
    clusters = clustering_model.fit_predict(df_preprocessed)

    # Evaluate clustering
    silhouette_avg = silhouette_score(df_preprocessed, clusters)
    ch_score = calinski_harabasz_score(df_preprocessed, clusters)
    db_score = davies_bouldin_score(df_preprocessed, clusters)

    # Log evaluation metrics
    metrics = {
        "silhouette_score": silhouette_avg,
        "calinski_harabasz_score": ch_score,
        "davies_bouldin_score": db_score
    }
    log_mlflow_metrics(metrics)

    # Log the DBSCAN model
    input_example = df_preprocessed.head(1)
    log_model_to_mlflow(clustering_model, "dbscan_model", input_example=input_example)

    # End the MLflow run
    end_mlflow_run()

    return {
        "message": "Clusters created successfully.",
        "silhouette_score": silhouette_avg,
        "calinski_harabasz_score": ch_score,
        "davies_bouldin_score": db_score
    }
