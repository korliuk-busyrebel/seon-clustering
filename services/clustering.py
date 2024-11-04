import logging
import time
from sklearn.cluster import DBSCAN  # for DBSCAN clustering algorithm
from sklearn.neighbors import NearestNeighbors  # for NearestNeighbors algorithm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def find_optimal_dbscan(df_preprocessed, min_eps=0.1, max_eps=30.0, step_eps=0.5, min_min_samples=2, max_min_samples=10):
    """
    Finds the optimal `eps` and `min_samples` parameters for DBSCAN clustering.
    """
    start_time = time.time()
    optimal_eps = min_eps
    optimal_min_samples = min_min_samples
    best_noise_ratio = 1.0  # Start with maximum noise ratio
    last_log_time = start_time

    logger.info("Starting to find optimal DBSCAN parameters...")

    while optimal_eps <= max_eps:
        for min_samples in range(min_min_samples, max_min_samples + 1):
            clustering_model = DBSCAN(eps=optimal_eps, min_samples=min_samples)
            clusters = clustering_model.fit_predict(df_preprocessed)

            noise_points = list(clusters).count(-1)
            noise_ratio = noise_points / len(clusters)

            unique_clusters = set(clusters)
            if -1 in unique_clusters:
                unique_clusters.remove(-1)

            if len(unique_clusters) > 1 and noise_ratio < best_noise_ratio:
                best_noise_ratio = noise_ratio
                optimal_eps = optimal_eps
                optimal_min_samples = min_samples

            if noise_ratio <= 0.05:
                elapsed_time = time.time() - start_time
                logger.info(f"Found optimal parameters early with eps={optimal_eps} and min_samples={optimal_min_samples}. Time taken: {elapsed_time:.2f}s")
                return optimal_eps, optimal_min_samples

            # Log every 10 seconds
            if time.time() - last_log_time > 10:
                progress = (optimal_eps - min_eps) / (max_eps - min_eps) * 100
                logger.info(f"Progress: {progress:.2f}% complete. Currently testing eps={optimal_eps}, min_samples={min_samples}")
                last_log_time = time.time()

        optimal_eps += step_eps

    elapsed_time = time.time() - start_time
    logger.info(f"Completed search for optimal DBSCAN parameters. Best found: eps={optimal_eps}, min_samples={optimal_min_samples}. Time taken: {elapsed_time:.2f}s")
    return optimal_eps, optimal_min_samples

def assign_noise_points(df_preprocessed, clusters):
    """
    Reassigns noise points to the nearest cluster using the NearestNeighbors algorithm.
    """
    start_time = time.time()
    logger.info("Starting reassignment of noise points...")

    noise_indices = (clusters == -1)
    non_noise_indices = (clusters != -1)

    if sum(noise_indices) == 0:
        logger.info("No noise points found. Skipping reassignment.")
        return clusters

    nearest_neighbors = NearestNeighbors(n_neighbors=1)
    nearest_neighbors.fit(df_preprocessed[non_noise_indices])

    distances, indices = nearest_neighbors.kneighbors(df_preprocessed[noise_indices])

    for i, idx in enumerate(indices):
        nearest_cluster = clusters[non_noise_indices][idx][0]
        clusters[noise_indices][i] = nearest_cluster

    elapsed_time = time.time() - start_time
    logger.info(f"Completed reassignment of noise points. Time taken: {elapsed_time:.2f}s")
    return clusters
