from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import umap
from mlflow_utils import log_metrics_and_model
from opensearch_utils import save_to_opensearch


def process_dbscan(df_preprocessed, df, column_weights):
    optimal_eps, optimal_min_samples = find_optimal_dbscan(df_preprocessed)

    # Apply DBSCAN
    clustering_model = DBSCAN(eps=optimal_eps, min_samples=optimal_min_samples)
    clusters = clustering_model.fit_predict(df_preprocessed)

    # Reassign noise points
    clusters = assign_noise_points(df_preprocessed, clusters)
    df['cluster'] = clusters

    # Dimensionality reduction
    df_reduced = reduce_dimensions_optimal(df_preprocessed)

    # Log in MLflow and save to OpenSearch
    results = log_metrics_and_model(clustering_model, df_preprocessed, clusters, df, df_reduced, optimal_eps,
                                    optimal_min_samples)
    save_to_opensearch(df, df_reduced)

    return results


def find_optimal_dbscan(df_preprocessed, min_eps=0.1, max_eps=30.0, step_eps=0.5, min_min_samples=2,
                        max_min_samples=10):
    optimal_eps = min_eps
    optimal_min_samples = min_min_samples
    best_noise_ratio = 1.0

    while optimal_eps <= max_eps:
        for min_samples in range(min_min_samples, max_min_samples + 1):
            clustering_model = DBSCAN(eps=optimal_eps, min_samples=min_samples)
            clusters = clustering_model.fit_predict(df_preprocessed)

            noise_ratio = list(clusters).count(-1) / len(clusters)
            unique_clusters = set(clusters)
            if -1 in unique_clusters:
                unique_clusters.remove(-1)

            if len(unique_clusters) > 1 and noise_ratio < best_noise_ratio:
                best_noise_ratio = noise_ratio
                optimal_eps, optimal_min_samples = optimal_eps, min_samples

            if noise_ratio <= 0.05:
                return optimal_eps, optimal_min_samples

        optimal_eps += step_eps

    return optimal_eps, optimal_min_samples


def assign_noise_points(df_preprocessed, clusters):
    noise_indices = (clusters == -1)
    non_noise_indices = (clusters != -1)

    if sum(noise_indices) == 0:
        return clusters

    nearest_neighbors = NearestNeighbors(n_neighbors=1)
    nearest_neighbors.fit(df_preprocessed[non_noise_indices])

    distances, indices = nearest_neighbors.kneighbors(df_preprocessed[noise_indices])

    for i, idx in enumerate(indices):
        nearest_cluster = clusters[non_noise_indices][idx][0]
        clusters[noise_indices][i] = nearest_cluster

    return clusters


def reduce_dimensions_optimal(df, n_components_pca=30, n_components_umap=2):
    pca = PCA(n_components=n_components_pca)
    df_pca = pca.fit_transform(df)

    reducer = umap.UMAP(n_components=n_components_umap)
    df_umap = reducer.fit_transform(df_pca)

    return pd.DataFrame(df_umap, columns=[f"dim_{i + 1}" for i in range(n_components_umap)])
