from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException
import pandas as pd
from sklearn.cluster import DBSCAN
from io import StringIO
from services.clustering import assign_noise_points
from services.dimensionality_reduction import reduce_dimensions_optimal
from services.evaluation import evaluate_clustering
from services.preprocessing import preprocess_data
from utils.opensearch_client import client, router, OS_KNN_INDEX, REDUCED_INDEX
from utils.column_weights import load_column_weights
import mlflow
import logging
import time
import threading

# Set up logging for progress tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a router
router = APIRouter()

def create_knn_index_if_needed(dimension):
    """Creates an OpenSearch index with the specified dimension for KNN if it doesn't exist."""
    index_name = OS_KNN_INDEX
    index_body = {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "vector": {"type": "knn_vector", "dimension": dimension},
                "id": {"type": "keyword"},
                "cluster": {"type": "integer"}
            }
        }
    }
    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name, body=index_body)
        logger.info(f"Created KNN index '{index_name}' with dimension {dimension}.")
    else:
        logger.info(f"KNN index '{index_name}' already exists.")

def find_optimal_dbscan_params(df):
    """Finds optimal eps and min_samples for DBSCAN based on minimum noise points."""
    best_eps = 5  # Default value for eps
    best_min_samples = 2  # Default value for min_samples
    min_noise_ratio = 1.0

    # Experiment with various eps and min_samples values to minimize noise points
    for eps in [5, 10, 15]:  # Example range for eps
        for min_samples in range(2, 5):  # Example range for min_samples
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(df)
            noise_ratio = list(labels).count(-1) / len(labels)

            if noise_ratio < min_noise_ratio:
                min_noise_ratio = noise_ratio
                best_eps = eps
                best_min_samples = min_samples

    return best_eps, best_min_samples

def log_progress(batch_number, total_batches, index_name, start_time):
    """Logs indexing progress every 10 seconds."""
    while batch_number[0] < total_batches:
        elapsed_time = time.time() - start_time
        percent_complete = (batch_number[0] / total_batches) * 100
        logger.info(
            f"Indexing {index_name} - Processing batch {batch_number[0]} / {total_batches} ({percent_complete:.2f}%) - Elapsed time: {elapsed_time:.2f} seconds.")
        time.sleep(10)

def process_clusters(df: pd.DataFrame):
    logger.info("Starting clustering process...")
    total_start_time = time.time()

    # Load column weights and preprocess the data
    column_weights = load_column_weights('/app/utils/column_weights.json')
    df_preprocessed = preprocess_data(df, column_weights)
    dimension = df_preprocessed.shape[1]
    create_knn_index_if_needed(dimension)

    # Dynamically calculate optimal DBSCAN parameters
    optimal_eps, optimal_min_samples = find_optimal_dbscan_params(df_preprocessed)
    logger.info(f"Optimal DBSCAN parameters: eps={optimal_eps}, min_samples={optimal_min_samples}")

    with mlflow.start_run() as run:
        mlflow.log_param("eps", optimal_eps)
        mlflow.log_param("min_samples", optimal_min_samples)

        clustering_model = DBSCAN(eps=optimal_eps, min_samples=optimal_min_samples)
        clusters = clustering_model.fit_predict(df_preprocessed)
        clusters = assign_noise_points(df_preprocessed, clusters)
        df['cluster'] = clusters
        logger.info("Clustering completed.")

        # Dimensionality reduction
        df_reduced = reduce_dimensions_optimal(df_preprocessed)
        df_reduced = pd.DataFrame(df_reduced, columns=[f"dim_{i + 1}" for i in range(df_reduced.shape[1])])

        # Log metrics
        silhouette_avg, ch_score, db_score = evaluate_clustering(df_preprocessed, clusters)
        mlflow.log_metric("silhouette_score", silhouette_avg)
        mlflow.log_metric("calinski_harabasz_score", ch_score)
        mlflow.log_metric("davies_bouldin_score", db_score)

    logger.info(f"Clustering process completed. Total Time: {time.time() - total_start_time:.2f} seconds.")

@router.post("/create-clusters/")
async def create_clusters(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')), chunksize=5000)
    df = pd.concat(df)
    logger.info("Data successfully loaded in chunks.")

    background_tasks.add_task(process_clusters, df)
    return {"message": "Cluster creation started in background."}

# Export the router for use in the main app
create_clusters = router
