import mlflow
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from config import MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD

# Initialize MLflow with the configuration
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("clustering_experiment")


def log_metrics_and_model(clustering_model, df_preprocessed, clusters, df, df_reduced, eps, min_samples):
    start_mlflow_run()

    with mlflow.start_run():
        mlflow.log_param("eps", eps)
        mlflow.log_param("min_samples", min_samples)

        silhouette_avg, ch_score, db_score = evaluate_clustering(df_preprocessed, clusters)
        mlflow.log_metric("silhouette_score", silhouette_avg)
        mlflow.log_metric("calinski_harabasz_score", ch_score)
        mlflow.log_metric("davies_bouldin_score", db_score)

        input_example = df_preprocessed.head(1)
        mlflow.sklearn.log_model(clustering_model, "dbscan_model", input_example=input_example)

    return {
        "message": f"Clusters created with eps={eps}, min_samples={min_samples}",
        "silhouette_score": silhouette_avg,
        "calinski_harabasz_score": ch_score,
        "davies_bouldin_score": db_score
    }


def evaluate_clustering(df_preprocessed, clusters):
    silhouette_avg = silhouette_score(df_preprocessed, clusters)
    ch_score = calinski_harabasz_score(df_preprocessed, clusters)
    db_score = davies_bouldin_score(df_preprocessed, clusters)
    return silhouette_avg, ch_score, db_score
