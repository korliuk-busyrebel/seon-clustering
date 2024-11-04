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
    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name, body=index_body)
        logger.info(f"Created KNN index '{index_name}' with dimension {dimension}.")
    else:
        logger.info(f"KNN index '{index_name}' already exists.")


def log_progress(batch_number, total_batches, index_name, start_time):
    """Logs indexing progress every 10 seconds."""
    while batch_number[0] < total_batches:
        elapsed_time = time.time() - start_time
        percent_complete = (batch_number[0] / total_batches) * 100
        logger.info(
            f"Indexing {index_name} - Processing batch {batch_number[0]} / {total_batches} ({percent_complete:.2f}%) - Elapsed time: {elapsed_time:.2f} seconds.")
        time.sleep(10)


def index_documents_in_batches(df, index_name, batch_size, batch_number, start_time):
    """Index documents in batches with retry logic."""
    skipped_docs = 0
    total_batches = (len(df) + batch_size - 1) // batch_size

    for i in range(0, len(df), batch_size):
        batch_number[0] = i // batch_size + 1
        batch = df[i:i + batch_size].to_dict(orient="records")

        for index, doc in enumerate(batch):
            success = False
            for attempt in range(3):  # Retry up to 3 times
                try:
                    client.index(index=index_name, id=i + index, body=doc)
                    success = True
                    break
                except Exception as e:
                    logger.error(f"Error indexing document {i + index} in {index_name}: {e}")
                    time.sleep(2)  # Wait before retrying
            if not success:
                skipped_docs += 1
                logger.warning(f"Document {i + index} in {index_name} skipped after 3 failed attempts.")

    elapsed_time = time.time() - start_time
    logger.info(
        f"Indexing {index_name} completed - Total skipped documents: {skipped_docs}, Time taken: {elapsed_time:.2f} seconds.")
    return skipped_docs


def process_clusters(df: pd.DataFrame):
    logger.info("Starting clustering process...")
    total_start_time = time.time()

    # Load column weights and preprocess the data
    column_weights = load_column_weights('/app/utils/column_weights.json')
    df_preprocessed = preprocess_data(df, column_weights)
    dimension = df_preprocessed.shape[1]
    create_knn_index_if_needed(dimension)

    # Clustering
    optimal_eps, optimal_min_samples = 5, 2
    with mlflow.start_run() as run:
        mlflow.log_param("eps", optimal_eps)
        mlflow.log_param("min_samples", optimal_min_samples)

        clustering_model = DBSCAN(eps=optimal_eps, min_samples=optimal_min_samples)
        clusters = clustering_model.fit_predict(df_preprocessed)
        clusters = assign_noise_points(df_preprocessed, clusters)
        df['cluster'] = clusters
        logger.info("Clustering completed.")

        # Track progress for indexing visual and KNN data
        batch_size = 500
        knn_batch_number = [0]
        visual_batch_number = [0]
        total_batches = (len(df) + batch_size - 1) // batch_size
        knn_progress_thread = threading.Thread(target=log_progress,
                                               args=(knn_batch_number, total_batches, OS_KNN_INDEX, total_start_time))
        visual_progress_thread = threading.Thread(target=log_progress, args=(
        visual_batch_number, total_batches, REDUCED_INDEX, total_start_time))
        knn_progress_thread.start()
        visual_progress_thread.start()

        # Dimensionality reduction
        df_reduced = reduce_dimensions_optimal(df_preprocessed)
        df_reduced = pd.DataFrame(df_reduced, columns=[f"dim_{i + 1}" for i in range(df_reduced.shape[1])])

        # Indexing visual and KNN data
        skipped_visual = index_documents_in_batches(df_reduced, REDUCED_INDEX, batch_size, visual_batch_number,
                                                    total_start_time)
        knn_docs = [
            {"id": str(df['id'].iloc[i]), "vector": df_preprocessed.iloc[i].tolist(), "cluster": clusters[i]}
            for i in range(len(df))
        ]
        skipped_knn = index_documents_in_batches(pd.DataFrame(knn_docs), OS_KNN_INDEX, batch_size, knn_batch_number,
                                                 total_start_time)

        # Log metrics
        silhouette_avg, ch_score, db_score = evaluate_clustering(df_preprocessed, clusters)
        mlflow.log_metric("silhouette_score", silhouette_avg)
        mlflow.log_metric("calinski_harabasz_score", ch_score)
        mlflow.log_metric("davies_bouldin_score", db_score)

    knn_progress_thread.join()
    visual_progress_thread.join()
    logger.info(
        f"Clustering and indexing process completed. Total skipped documents - Visual: {skipped_visual}, KNN: {skipped_knn}, Total Time: {time.time() - total_start_time:.2f} seconds.")


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
