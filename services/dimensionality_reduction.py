import logging
import time
from sklearn.decomposition import PCA
import umap

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def reduce_dimensions_optimal(df, n_components_pca=30, n_components_umap=2):
    """
    Reduces dimensions of the input DataFrame using PCA followed by UMAP.
    Logs progress and timings for each step.
    """
    # Step 1: PCA
    start_time = time.time()
    logger.info("Starting dimensionality reduction with PCA...")

    pca = PCA(n_components=n_components_pca)
    df_pca = pca.fit_transform(df)

    elapsed_time_pca = time.time() - start_time
    logger.info(f"PCA dimensionality reduction completed. Reduced to {n_components_pca} components. Time taken: {elapsed_time_pca:.2f}s")

    # Step 2: UMAP
    start_time_umap = time.time()
    logger.info("Starting dimensionality reduction with UMAP...")

    reducer = umap.UMAP(n_components=n_components_umap)
    df_umap = reducer.fit_transform(df_pca)

    elapsed_time_umap = time.time() - start_time_umap
    logger.info(f"UMAP dimensionality reduction completed. Reduced to {n_components_umap} components. Time taken: {elapsed_time_umap:.2f}s")

    total_time = elapsed_time_pca + elapsed_time_umap
    logger.info(f"Total time for dimensionality reduction: {total_time:.2f}s")

    return df_umap
