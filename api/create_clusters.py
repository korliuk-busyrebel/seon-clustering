from fastapi import APIRouter, File, UploadFile, BackgroundTasks
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN  # for DBSCAN clustering algorithm
from io import StringIO
from services.clustering import find_optimal_dbscan, assign_noise_points
from services.dimensionality_reduction import reduce_dimensions_optimal
from services.evaluation import evaluate_clustering
from services.preprocessing import preprocess_data
from utils.opensearch_client import client, router, OS_KNN_INDEX, REDUCED_INDEX, OS_INDEX, OS_RAW_INDEX
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
        logger.info(f"Processing batch {batch_number[0]} / {total_batches} ({percent_complete:.2f}%) - Elapsed time: {elapsed_time:.2f} seconds.")
        time.sleep(10)

# Function to handle the clustering process in the background
def process_clusters(df: pd.DataFrame):
    logger.info("Starting clustering process...")
    total_start_time = time.time()

    # Load column weights and preprocess the data
    step_start_time = time.time()
    column_weights = load_column_weights('/app/utils/column_weights.json')
    df_preprocessed = preprocess_data(df, column_weights)
    logger.info(f"Data preprocessing completed in {time.time() - step_start_time:.2f} seconds.")

    # # Dynamically find optimal eps and min_samples
    # step_start_time = time.time()
    # optimal_eps, optimal_min_samples = find_optimal_dbscan(df_preprocessed)
    # logger.info(f"Optimal parameters found: eps={optimal_eps}, min_samples={optimal_min_samples} in {time.time() - step_start_time:.2f} seconds.")
    optimal_eps = 5
    optimal_min_samples = 2
    # Start an MLflow run
    with mlflow.start_run() as run:
        mlflow.log_param("eps", optimal_eps)
        mlflow.log_param("min_samples", optimal_min_samples)

        # Apply DBSCAN clustering
        step_start_time = time.time()
        clustering_model = DBSCAN(eps=optimal_eps, min_samples=optimal_min_samples)
        clusters = clustering_model.fit_predict(df_preprocessed)
        clusters = assign_noise_points(df_preprocessed, clusters)
        df['cluster'] = clusters
        logger.info(f"Clustering completed in {time.time() - step_start_time:.2f} seconds.")

        # Save to OpenSearch in batches
        batch_size = 500
        num_batches = (len(df) + batch_size - 1) // batch_size
        batch_number = [0]  # Mutable container to track batches in thread

        # Start progress logging in a separate thread
        progress_thread = threading.Thread(target=log_progress, args=(batch_number, num_batches, total_start_time))
        progress_thread.start()

        # # Store raw data
        # logger.info("Starting to store raw data in OpenSearch...")
        # step_start_time = time.time()
        # for i in range(0, len(df), batch_size):
        #     batch_number[0] = i // batch_size + 1
        #     batch = df[i:i + batch_size].to_dict(orient="records")
        #     for index, doc in enumerate(batch):
        #         client.index(index=OS_RAW_INDEX, id=i + index, body=doc)
        # logger.info(f"Raw data storage completed in {time.time() - step_start_time:.2f} seconds.")

        # Process reduced dimensions and store them in OpenSearch
        logger.info("Starting dimensionality reduction...")
        step_start_time = time.time()
        df_reduced = reduce_dimensions_optimal(df_preprocessed)
        df_reduced = pd.DataFrame(df_reduced, columns=[f"dim_{i + 1}" for i in range(df_reduced.shape[1])])
        logger.info(f"Dimensionality reduction completed in {time.time() - step_start_time:.2f} seconds.")

        # Store reduced dimension data in OpenSearch
        step_start_time = time.time()
        logger.info("Storing reduced dimension data in OpenSearch...")
        for i in range(0, len(df_reduced), batch_size):
            batch_number[0] = i // batch_size + 1
            batch = df_reduced[i:i + batch_size]
            batch_dicts = [
                {**row.to_dict(), "cluster": df['cluster'].iloc[i + index]}
                for index, row in batch.iterrows()
            ]
            for index, doc in enumerate(batch_dicts):
                client.index(index=REDUCED_INDEX, id=i + index, body=doc)
        logger.info(f"Reduced dimension data storage completed in {time.time() - step_start_time:.2f} seconds.")

        # Prepare and store KNN vectors in OpenSearch
        step_start_time = time.time()
        logger.info("Storing KNN vectors in OpenSearch...")
        for i in range(0, len(df), batch_size):
            batch_number[0] = i // batch_size + 1
            batch = df_preprocessed[i:i + batch_size]
            knn_docs = [
                {"id": str(df['id'].iloc[i + idx]), "vector": row.tolist(), "cluster": df['cluster'].iloc[i + idx]}
                for idx, row in batch.iterrows()
            ]
            for index, doc in enumerate(knn_docs):
                client.index(index=OS_KNN_INDEX, id=i + index, body=doc)
        logger.info(f"KNN vector storage completed in {time.time() - step_start_time:.2f} seconds.")

        # Calculate and log evaluation metrics
        logger.info("Calculating clustering evaluation metrics...")
        step_start_time = time.time()
        silhouette_avg, ch_score, db_score = evaluate_clustering(df_preprocessed, clusters)
        mlflow.log_metric("silhouette_score", silhouette_avg)
        mlflow.log_metric("calinski_harabasz_score", ch_score)
        mlflow.log_metric("davies_bouldin_score", db_score)
        mlflow.sklearn.log_model(clustering_model, "dbscan_model")
        logger.info(f"Evaluation metrics logged in {time.time() - step_start_time:.2f} seconds.")

    # Stop progress logging
    progress_thread.join()
    logger.info(f"Clustering process completed in {time.time() - total_start_time:.2f} seconds.")

@router.post("/create-clusters/")
async def create_clusters(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    contents = await file.read()

    # Load data in chunks to avoid memory issues with very large files
    df = pd.read_csv(StringIO(contents.decode('utf-8')), chunksize=50000)
    df = pd.concat(df)
    logger.info("Data successfully loaded in chunks.")

    # Add the clustering task to the background
    background_tasks.add_task(process_clusters, df)

    return {"message": "Cluster creation started in background."}

# Export the router for use in the main app
create_clusters = router
