from fastapi import APIRouter, File, UploadFile, BackgroundTasks
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
    # Check if the index already exists
    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name, body=index_body)
        logger.info(f"Created KNN index '{index_name}' with dimension {dimension}.")
    else:
        logger.info(f"KNN index '{index_name}' already exists.")

def log_knn_progress(batch_number, total_batches, start_time):
    """Logs KNN indexing progress every 10 seconds."""
    while batch_number[0] < total_batches:
        elapsed_time = time.time() - start_time
        percent_complete = (batch_number[0] / total_batches) * 100
        logger.info(f"Indexing {OS_KNN_INDEX} - Processing batch {batch_number[0]} / {total_batches} ({percent_complete:.2f}%) - Elapsed time: {elapsed_time:.2f} seconds.")
        time.sleep(10)

def log_visual_progress(batch_number, total_batches, start_time):
    """Logs visual indexing progress every 10 seconds."""
    while batch_number[0] < total_batches:
        elapsed_time = time.time() - start_time
        percent_complete = (batch_number[0] / total_batches) * 100
        logger.info(f"Indexing {REDUCED_INDEX} - Processing batch {batch_number[0]} / {total_batches} ({percent_complete:.2f}%) - Elapsed time: {elapsed_time:.2f} seconds.")
        time.sleep(10)

def process_clusters(df: pd.DataFrame):
    logger.info("Starting clustering process...")
    total_start_time = time.time()

    # Load column weights and preprocess the data
    step_start_time = time.time()
    column_weights = load_column_weights('/app/utils/column_weights.json')
    df_preprocessed = preprocess_data(df, column_weights)
    logger.info(f"Data preprocessing completed in {time.time() - step_start_time:.2f} seconds.")

    # Define clustering parameters
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

        # Determine dimension for KNN indexing and create index if needed
        dimension = df_preprocessed.shape[1]
        create_knn_index_if_needed(dimension)

        # Save to OpenSearch in batches
        batch_size = 100
        num_batches = (len(df) + batch_size - 1) // batch_size
        knn_batch_number = [0]
        visual_batch_number = [0]

        # Start progress logging for each index in separate threads
        knn_progress_thread = threading.Thread(target=log_knn_progress, args=(knn_batch_number, num_batches, total_start_time))
        visual_progress_thread = threading.Thread(target=log_visual_progress, args=(visual_batch_number, num_batches, total_start_time))
        knn_progress_thread.start()
        visual_progress_thread.start()

        # Process reduced dimensions and store them in OpenSearch
        logger.info("Starting dimensionality reduction...")
        step_start_time = time.time()
        df_reduced = reduce_dimensions_optimal(df_preprocessed)
        df_reduced = pd.DataFrame(df_reduced, columns=[f"dim_{i + 1}" for i in range(df_reduced.shape[1])])
        logger.info(f"Dimensionality reduction completed in {time.time() - step_start_time:.2f} seconds.")

        # Store reduced dimension data in OpenSearch
        step_start_time = time.time()
        logger.info(f"Storing reduced dimension data in {REDUCED_INDEX}...")
        for i in range(0, len(df_reduced), batch_size):
            visual_batch_number[0] = i // batch_size + 1
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
        logger.info(f"Storing KNN vectors in {OS_KNN_INDEX}...")
        for i in range(0, len(df), batch_size):
            knn_batch_number[0] = i // batch_size + 1
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
    knn_progress_thread.join()
    visual_progress_thread.join()
    logger.info(f"Clustering process completed in {time.time() - total_start_time:.2f} seconds.")

@router.post("/create-clusters/")
async def create_clusters(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    contents = await file.read()

    # Load data in chunks to avoid memory issues with very large files
    df = pd.read_csv(StringIO(contents.decode('utf-8')), chunksize=5000)
    df = pd.concat(df)
    logger.info("Data successfully loaded in chunks.")

    # Add the clustering task to the background
    background_tasks.add_task(process_clusters, df)

    return {"message": "Cluster creation started in background."}

# Export the router for use in the main app
create_clusters = router
