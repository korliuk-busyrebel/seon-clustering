import logging
import time
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def evaluate_clustering(df, clusters):
    """
    Evaluates clustering performance using Silhouette, Calinski-Harabasz, and Davies-Bouldin scores.
    Logs progress and timing for each metric calculation.
    """
    # Silhouette Score
    start_time = time.time()
    logger.info("Calculating Silhouette Score...")
    silhouette_avg = silhouette_score(df, clusters)
    elapsed_silhouette = time.time() - start_time
    logger.info(f"Silhouette Score: {silhouette_avg:.4f} (Time taken: {elapsed_silhouette:.2f}s)")

    # Calinski-Harabasz Score
    start_time = time.time()
    logger.info("Calculating Calinski-Harabasz Score...")
    ch_score = calinski_harabasz_score(df, clusters)
    elapsed_ch = time.time() - start_time
    logger.info(f"Calinski-Harabasz Score: {ch_score:.4f} (Time taken: {elapsed_ch:.2f}s)")

    # Davies-Bouldin Score
    start_time = time.time()
    logger.info("Calculating Davies-Bouldin Score...")
    db_score = davies_bouldin_score(df, clusters)
    elapsed_db = time.time() - start_time
    logger.info(f"Davies-Bouldin Score: {db_score:.4f} (Time taken: {elapsed_db:.2f}s)")

    total_time = elapsed_silhouette + elapsed_ch + elapsed_db
    logger.info(f"Total time for clustering evaluation: {total_time:.2f}s")

    return silhouette_avg, ch_score, db_score
