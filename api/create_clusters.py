from fastapi import APIRouter, File, UploadFile, BackgroundTasks
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors  # for custom DBSCAN implementation
from io import StringIO
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


def log_progress(batch_number, total_batches, start_time):
    """Logs estimated completion percentage every 10 seconds."""
    while batch_number[0] < total_batches:
        elapsed_time = time.time() - start_time
        percent_complete = (batch_number[0] / total_batches) * 100
        logger.info(
            f"Processing batch {batch_number[0]} / {total_batches} ({percent_complete:.2f}%) - Elapsed time: {elapsed_time:.2f} seconds.")
        time.sleep(10)


# Custom DBSCAN with logging
def dbscan_with_logging(df, eps, min_samples):
    """Custom DBSCAN implementation with progress logging."""
    start_time = time.time()
    logger.info("Starting DBSCAN clustering...")

    neighbors = NearestNeighbors(radius=eps).fit(df)
    distances, indices = neighbors.radius_neighbors(df)

    core_points = [i for i, neighbors in enumerate(indices) if len(neighbors) >= min_samples]
    clusters = -np.ones(df.shape[0], dtype=int)
    cluster_id = 0

    logger.info(f"Found {len(core_points)} core points. Expanding clusters...")
    for i, neighbors in enumerate(indices):
        if i % 100 == 0:
            logger.info(f"Processed {i} / {len(core_points)} core points.")

        if clusters[i] != -1 or i not in core_points:
            continue

        clusters[i] = cluster_id
        points_to_expand = list(neighbors)

        while points_to_expand:
            point = points_to_expand.pop()
            if clusters[point] == -1:
                clusters[point] = cluster_id
            elif clusters[point] == -1:
                clusters[point] = cluster_id
                if point in core_points:
                    points_to_expand.extend(indices[point])

        cluster_id += 1

    logger.info(f"DBSCAN clustering completed in {time.time() - start_time:.2f} seconds.")
    return clusters


# Background process for clustering
def process_clusters(df: pd.DataFrame):
    logger.info("Starting clustering process...")
    total_start_time = time.time()

    # Preprocess the data
    column_weights = load_column_weights('/app/utils/column_weights.json')
    df_preprocessed = preprocess_data(df, column_weights)
    optimal_eps, optimal_min_samples = 5, 2

    with mlflow.start_run() as run:
        mlflow.log_param("eps", optimal_eps)
        mlflow.log_param("min_samples", optimal_min_samples)

        # Apply DBSCAN clustering with custom logging
        clusters = dbscan_with_logging(df_preprocessed, eps=optimal_eps, min_samples=optimal_min_samples)
        df['cluster'] = clusters

        # Dimensionality reduction and OpenSearch storage
        batch_size = 100
        num_batches = (len(df) + batch_size - 1) // batch_size
        batch_number = [0]
        progress_thread = threading.Thread(target=log_progress, args=(batch_number, num_batches, total_start_time))
        progress_thread.start()

        logger.info("Starting dimensionality reduction...")
        df_reduced = reduce_dimensions_optimal(df_preprocessed)
        df_reduced = pd.DataFrame(df_reduced, columns=[f"dim_{i + 1}" for i in range(df_reduced.shape[1])])
        logger.info(f"Dimensionality reduction completed.")

        # Store reduced dimension data in OpenSearch
        for i in range(0, len(df_reduced), batch_size):
            batch_number[0] = i // batch_size + 1
            batch = df_reduced[i:i + batch_size]
            batch_dicts = [
                {**row.to_dict(), "cluster": df['cluster'].iloc[i + index]}
                for index, row in batch.iterrows()
            ]
            for index, doc in enumerate(batch_dicts):
                client.index(index=REDUCED_INDEX, id=i + index, body=doc)
        logger.info("Reduced dimension data storage completed.")

        # Store KNN vectors in OpenSearch
        for i in range(0, len(df), batch_size):
            batch_number[0] = i // batch_size + 1
            batch = df_preprocessed[i:i + batch_size]
            knn_docs = [
                {"id": str(df['id'].iloc[i + idx]), "vector": row.tolist(), "cluster": df['cluster'].iloc[i + idx]}
                for idx, row in batch.iterrows()
            ]
            for index, doc in enumerate(knn_docs):
                client.index(index=OS_KNN_INDEX, id=i + index, body=doc)
        logger.info("KNN vector storage completed.")

        # Evaluation metrics
        silhouette_avg, ch_score, db_score = evaluate_clustering(df_preprocessed, clusters)
        mlflow.log_metric("silhouette_score", silhouette_avg)
        mlflow.log_metric("calinski_harabasz_score", ch_score)
        mlflow.log_metric("davies_bouldin_score", db_score)
        mlflow.sklearn.log_model(clustering_model, "dbscan_model")
        logger.info("Evaluation metrics logged.")

    # Stop progress logging
    progress_thread.join()
    logger.info(f"Clustering process completed in {time.time() - total_start_time:.2f} seconds.")


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
