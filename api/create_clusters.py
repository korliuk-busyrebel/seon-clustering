import os
import pandas as pd
from fastapi import APIRouter, File, UploadFile
from services.clustering import find_optimal_dbscan, assign_noise_points
from services.dimensionality_reduction import reduce_dimensions_optimal
from services.evaluation import evaluate_clustering
from services.preprocessing import preprocess_data
from utils.column_weights import load_column_weights
from opensearchpy import OpenSearch
import mlflow

router = APIRouter()

# Set GIT_PYTHON_REFRESH to silence the warning about Git
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

# Initialize OpenSearch client
OS_HOST = os.getenv("OS_HOST", "localhost")
OS_PORT = os.getenv("OS_PORT", 9200)
OS_INDEX = os.getenv("OS_INDEX", "clustered_data")
REDUCED_INDEX = os.getenv("OS_REDUCED_INDEX", "clustered_data_visual")
OS_SCHEME = os.getenv("OS_SCHEME", "http")
OS_USERNAME = os.getenv("OS_USERNAME", "admin")
OS_PASSWORD = os.getenv("OS_PASSWORD", "admin")

client = OpenSearch(
    hosts=[{'host': OS_HOST, 'port': int(OS_PORT)}],
    http_auth=(OS_USERNAME, OS_PASSWORD),
    use_ssl=(OS_SCHEME == 'https'),
    verify_certs=False,
    scheme=OS_SCHEME,
    timeout=60
)

@router.post("/create-clusters/")
async def create_clusters(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))

        # Load column weights
        column_weights = load_column_weights('/app/utils/column_weights.json')
        df_preprocessed = preprocess_data(df, column_weights)

        # Dynamically find optimal eps and min_samples
        optimal_eps, optimal_min_samples = find_optimal_dbscan(df_preprocessed)

        # Start an MLflow run
        with mlflow.start_run() as run:
            mlflow.log_param("eps", optimal_eps)
            mlflow.log_param("min_samples", optimal_min_samples)

            # Apply DBSCAN clustering
            clustering_model = DBSCAN(eps=optimal_eps, min_samples=optimal_min_samples)
            clusters = clustering_model.fit_predict(df_preprocessed)

            # Assign noise points (-1) to nearest cluster
            clusters = assign_noise_points(df_preprocessed, clusters)

            # Save clusters to the dataframe (using pd.concat to avoid fragmentation)
            df = pd.concat([df, pd.Series(clusters, name='cluster')], axis=1)

            # Log model input example for signature
            input_example = df_preprocessed.head(1)
            mlflow.sklearn.log_model(clustering_model, "dbscan_model", input_example=input_example)

            # Optimal dimensionality reduction using PCA + UMAP
            df_reduced = reduce_dimensions_optimal(df_preprocessed)

            # Store original data with clusters in OpenSearch
            for index, row in df.iterrows():
                doc = row.to_dict()
                client.index(index=OS_INDEX, id=index, body=doc)

            # Store reduced-dimension data with cluster labels in OpenSearch
            for index, row in df_reduced.iterrows():
                doc = row.to_dict()
                doc['cluster'] = df['cluster'].iloc[index]
                client.index(index=REDUCED_INDEX, id=index, body=doc)

            # Calculate evaluation metrics
            silhouette_avg, ch_score, db_score = evaluate_clustering(df_preprocessed, clusters)

            # Log metrics in MLflow
            mlflow.log_metric("silhouette_score", silhouette_avg)
            mlflow.log_metric("calinski_harabasz_score", ch_score)
            mlflow.log_metric("davies_bouldin_score", db_score)

            # Log the DBSCAN model in MLflow
            mlflow.sklearn.log_model(clustering_model, "dbscan_model")

        return {
            "message": f"Clusters created with final eps={optimal_eps}, min_samples={optimal_min_samples}, and stored in {OS_INDEX}. Reduced dimension data stored in {REDUCED_INDEX}.",
            "silhouette_score": silhouette_avg,
            "calinski_harabasz_score": ch_score,
            "davies_bouldin_score": db_score
        }
    except Exception as e:
        return {"error": str(e)}

# Export the router for use in the main app
create_clusters = router
