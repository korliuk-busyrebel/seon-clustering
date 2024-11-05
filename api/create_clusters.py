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

def find_optimal_dbscan_params(df, sample_fraction=0.1, eps_values=[5, 10, 15], min_samples_values=range(2, 5)):
    """
    Finds optimal eps and min_samples for DBSCAN by testing on a sample of the data
    to minimize noise points, with reduced memory usage.
    """
    # Sample a fraction of the dataset to reduce memory and computation
    sample_size = int(len(df) * sample_fraction)
    df_sample = df.sample(n=sample_size, random_state=42) if sample_size > 0 else df

    best_eps = eps_values[0]
    best_min_samples = min_samples_values[0]
    min_noise_ratio = 1.0  # Initialize with the highest possible noise ratio

    # Iterate over the range of eps and min_samples values
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(df_sample)
            noise_ratio = np.sum(labels == -1) / len(labels)

            # Track the best eps and min_samples with the lowest noise ratio
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

    try:
        # Load column weights and preprocess the data
        column_weights = load_column_weights('/app/utils/column_weights.json')
        logger.info("Loaded column weights.")

        df_preprocessed = preprocess_data(df, column_weights)
        logger.info("Data preprocessing completed.")

        # Reduce dimensionality early to help with memory constraints in clustering
        df_preprocessed = reduce_dimensions_optimal(df_preprocessed)
        logger.info("Dimensionality reduction completed.")

        # Check the dimensions and sample of preprocessed data
        logger.info(f"Preprocessed data shape: {df_preprocessed.shape}")
        logger.info(f"Preprocessed data sample: {df_preprocessed.head()}")

        # Create KNN index
        dimension = df_preprocessed.shape[1]
        create_knn_index_if_needed(dimension)
        logger.info(f"Created KNN index with dimension {dimension}.")

        # Calculate optimal DBSCAN parameters on a subset
        optimal_eps, optimal_min_samples = find_optimal_dbscan_params(df_preprocessed.sample(frac=0.1))
        logger.info(f"Optimal DBSCAN parameters found: eps={optimal_eps}, min_samples={optimal_min_samples}")

        # Run DBSCAN clustering
        clustering_model = DBSCAN(eps=optimal_eps, min_samples=optimal_min_samples)
        clusters = clustering_model.fit_predict(df_preprocessed)
        logger.info("DBSCAN clustering completed.")

        # Assign clusters and check for any clusters labeled as noise (-1)
        df['cluster'] = clusters
        noise_points = (clusters == -1).sum()
        logger.info(f"Total noise points (cluster -1): {noise_points}")

        # Dimensionality reduction for visualization (if needed)
        df_reduced = reduce_dimensions_optimal(df_preprocessed)
        df_reduced = pd.DataFrame(df_reduced, columns=[f"dim_{i + 1}" for i in range(df_reduced.shape[1])])
        logger.info("Second dimensionality reduction completed for visualization.")

        # Indexing data
        logger.info("Starting indexing to OpenSearch...")
        skipped_knn = index_documents_in_batches(df, OS_KNN_INDEX, batch_size=500, batch_number=[0],
                                                 start_time=total_start_time)
        logger.info(f"Indexing completed. Skipped KNN documents: {skipped_knn}")

        # Log metrics
        silhouette_avg, ch_score, db_score = evaluate_clustering(df_preprocessed, clusters)
        mlflow.log_metric("silhouette_score", silhouette_avg)
        mlflow.log_metric("calinski_harabasz_score", ch_score)
        mlflow.log_metric("davies_bouldin_score", db_score)
        logger.info("Clustering metrics logged to MLflow.")

    except Exception as e:
        logger.error(f"Error during clustering process: {e}")
        raise HTTPException(status_code=500, detail=f"Error during clustering process: {e}")

    finally:
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
