from fastapi import APIRouter, File, UploadFile, BackgroundTasks
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN  # for DBSCAN clustering algorithm
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

def log_progress(batch_number, total_batches, index_name, start_time):
    """Logs estimated completion percentage every 10 seconds for a specific index."""
    while batch_number[0] < total_batches:
        elapsed_time = time.time() - start_time
        percent_complete = (batch_number[0] / total_batches) * 100
        logger.info(f"Indexing {index_name} - Processing batch {batch_number[0]} / {total_batches} ({percent_complete:.2f}%) - Elapsed time: {elapsed_time:.2f} seconds.")
        time.sleep(10)

def index_document_with_retry(client, index, id, body, retries=3, delay=2):
    """Index a document with retry on failure, and skip if all retries fail."""
    for attempt in range(retries):
        try:
            client.index(index=index, id=id, body=body)
            return True  # Exit if successful
        except Exception as e:
            logger.error(f"Error indexing document id {id} in index {index}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                logger.error(f"Failed to index document id {id} after {retries} attempts. Skipping this document.")
    return False  # Skip document after retries are exhausted

def process_clusters(df: pd.DataFrame):
    logger.info("Starting clustering process...")
    total_start_time = time.time()

    # Track skipped documents
    skipped_docs_reduced = 0
    skipped_docs_knn = 0

    # Load column weights and preprocess the data
    step_start_time = time.time()
    column_weights = load_column_weights('/app/utils/column_weights.json')
    df_preprocessed = preprocess_data(df, column_weights)
    logger.info(f"Data preprocessing completed in {time.time() - step_start_time:.2f} seconds.")

    # Set predefined parameters for DBSCAN clustering to avoid recalculation
    optimal_eps = 5
    optimal_min_samples = 2

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

        # Prepare batch configurations and progress tracking
        batch_size = 500
        num_batches_reduced = (len(df) + batch_size - 1) // batch_size
        num_batches_knn = (len(df) + batch_size - 1) // batch_size
        batch_number_reduced = [0]  # Track batch progress for REDUCED_INDEX
        batch_number_knn = [0]      # Track batch progress for OS_KNN_INDEX

        # Start separate progress logging threads for REDUCED_INDEX and OS_KNN_INDEX
        progress_thread_reduced = threading.Thread(target=log_progress, args=(batch_number_reduced, num_batches_reduced, REDUCED_INDEX, total_start_time))
        progress_thread_knn = threading.Thread(target=log_progress, args=(batch_number_knn, num_batches_knn, OS_KNN_INDEX, total_start_time))
        progress_thread_reduced.start()
        progress_thread_knn.start()

        # Process and store reduced dimensions
        logger.info("Starting dimensionality reduction for REDUCED_INDEX...")
        step_start_time = time.time()
        df_reduced = reduce_dimensions_optimal(df_preprocessed)
        df_reduced = pd.DataFrame(df_reduced, columns=[f"dim_{i + 1}" for i in range(df_reduced.shape[1])])
        logger.info(f"Dimensionality reduction completed in {time.time() - step_start_time:.2f} seconds.")

        # Store reduced dimension data in OpenSearch (REDUCED_INDEX)
        logger.info("Storing reduced dimension data in OpenSearch...")
        for i in range(0, len(df_reduced), batch_size):
            batch_number_reduced[0] = i // batch_size + 1
            batch = df_reduced[i:i + batch_size]
            batch_dicts = [
                {**row.to_dict(), "cluster": df['cluster'].iloc[i + index]}
                for index, row in batch.iterrows()
            ]
            for index, doc in enumerate(batch_dicts):
                if not index_document_with_retry(client, REDUCED_INDEX, id=i + index, body=doc):
                    logger.warning(f"Skipping document in REDUCED_INDEX with id {i + index}")
                    skipped_docs_reduced += 1
        logger.info(f"Reduced dimension data storage in {REDUCED_INDEX} completed in {time.time() - step_start_time:.2f} seconds.")

        # Prepare and store KNN vectors in OpenSearch (OS_KNN_INDEX)
        step_start_time = time.time()
        logger.info("Storing KNN vectors in OpenSearch...")
        for i in range(0, len(df), batch_size):
            batch_number_knn[0] = i // batch_size + 1
            batch = df_preprocessed[i:i + batch_size]
            knn_docs = [
                {"id": str(df['id'].iloc[i + idx]), "vector": row.tolist(), "cluster": df['cluster'].iloc[i + idx]}
                for idx, row in batch.iterrows()
            ]
            for index, doc in enumerate(knn_docs):
                if not index_document_with_retry(client, OS_KNN_INDEX, id=i + index, body=doc):
                    logger.warning(f"Skipping document in OS_KNN_INDEX with id {i + index}")
                    skipped_docs_knn += 1
        logger.info(f"KNN vector storage in {OS_KNN_INDEX} completed in {time.time() - step_start_time:.2f} seconds.")

        # Calculate and log evaluation metrics
        logger.info("Calculating clustering evaluation metrics...")
        step_start_time = time.time()
        silhouette_avg, ch_score, db_score = evaluate_clustering(df_preprocessed, clusters)
        mlflow.log_metric("silhouette_score", silhouette_avg)
        mlflow.log_metric("calinski_harabasz_score", ch_score)
        mlflow.log_metric("davies_bouldin_score", db_score)
        mlflow.sklearn.log_model(clustering_model, "dbscan_model")
        logger.info(f"Evaluation metrics logged in {time.time() - step_start_time:.2f} seconds.")

    # Stop progress logging threads
    progress_thread_reduced.join()
    progress_thread_knn.join()

    # Log total skipped documents for each index
    logger.info(f"Total skipped documents in {REDUCED_INDEX}: {skipped_docs_reduced}")
    logger.info(f"Total skipped documents in {OS_KNN_INDEX}: {skipped_docs_knn}")
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
