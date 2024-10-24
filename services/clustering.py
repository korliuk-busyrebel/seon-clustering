from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

## Function to find optimal eps and min_samples
def find_optimal_dbscan(df_preprocessed, min_eps=0.1, max_eps=30.0, step_eps=0.5, min_min_samples=2, max_min_samples=10):
    optimal_eps = min_eps
    optimal_min_samples = min_min_samples
    best_noise_ratio = 1.0  # Start with maximum noise ratio

    # Iterate over eps values
    while optimal_eps <= max_eps:
        # Iterate over min_samples values
        for min_samples in range(min_min_samples, max_min_samples + 1):
            clustering_model = DBSCAN(eps=optimal_eps, min_samples=min_samples)
            clusters = clustering_model.fit_predict(df_preprocessed)

            # Calculate the noise ratio
            noise_points = list(clusters).count(-1)
            noise_ratio = noise_points / len(clusters)

            # Exclude noise (-1) when counting clusters
            unique_clusters = set(clusters)
            if -1 in unique_clusters:
                unique_clusters.remove(-1)

            # If this configuration has less noise, save it as optimal
            if len(unique_clusters) > 1 and noise_ratio < best_noise_ratio:
                best_noise_ratio = noise_ratio
                optimal_eps = optimal_eps
                optimal_min_samples = min_samples

            # If noise ratio is low enough, stop early
            if noise_ratio <= 0.05:  # Stop if less than 5% of points are noise
                return optimal_eps, optimal_min_samples

        optimal_eps += step_eps

    return optimal_eps, optimal_min_samples

# Function to reassign noise points (-1) to nearest cluster
def assign_noise_points(df_preprocessed, clusters):
    noise_indices = (clusters == -1)
    non_noise_indices = (clusters != -1)

    if sum(noise_indices) == 0:
        return clusters

    # Fit Nearest Neighbors on non-noise points
    nearest_neighbors = NearestNeighbors(n_neighbors=1)
    nearest_neighbors.fit(df_preprocessed[non_noise_indices])

    # Find the nearest non-noise point for each noise point
    distances, indices = nearest_neighbors.kneighbors(df_preprocessed[noise_indices])

    # Reassign noise points to the nearest cluster
    for i, idx in enumerate(indices):
        nearest_cluster = clusters[non_noise_indices][idx][0]
        clusters[noise_indices][i] = nearest_cluster

    return clusters
