from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import umap


def find_optimal_dbscan(df_preprocessed, min_eps=0.1, max_eps=30.0, step_eps=0.5, min_min_samples=2,
                        max_min_samples=10):
    optimal_eps = min_eps
    optimal_min_samples = min_min_samples
    best_noise_ratio = 1.0

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
                return optimal_eps, optimal_min_samples
        optimal_eps += step_eps
    return optimal_eps, optimal_min_samples


def assign_noise_points(df_preprocessed, clusters):
    noise_indices = (clusters == -1)
    if sum(noise_indices) == 0:
        return clusters

    nearest_neighbors = NearestNeighbors(n_neighbors=1)
    nearest_neighbors.fit(df_preprocessed[clusters != -1])
    distances, indices = nearest_neighbors.kneighbors(df_preprocessed[noise_indices])

    for i, idx in enumerate(indices):
        clusters[noise_indices][i] = clusters[clusters != -1][idx][0]

    return clusters


def reduce_dimensions_optimal(df, n_components_pca=30, n_components_umap=2):
    pca = PCA(n_components=n_components_pca)
    df_pca = pca.fit_transform(df)
    reducer = umap.UMAP(n_components=n_components_umap)
    df_umap = reducer.fit_transform(df_pca)
    return pd.DataFrame(df_umap, columns=[f"dim_{i + 1}" for i in range(n_components_umap)])


def evaluate_clustering(df_preprocessed, clusters):
    silhouette_avg = silhouette_score(df_preprocessed, clusters)
    ch_score = calinski_harabasz_score(df_preprocessed, clusters)
    db_score = davies_bouldin_score(df_preprocessed, clusters)
    return silhouette_avg, ch_score, db_score
