import mlflow
import mlflow.sklearn
from fastapi import FastAPI, File, UploadFile
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from opensearchpy import OpenSearch
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from io import StringIO
import pandas as pd
import json
import os
import umap

# Initialize FastAPI app
app = FastAPI()

# Get environment variables for OpenSearch connection and index
OS_HOST = os.getenv("OS_HOST", "localhost")
OS_PORT = os.getenv("OS_PORT", 9200)
OS_INDEX = os.getenv("OS_INDEX", "clustered_data")
REDUCED_INDEX = os.getenv("OS_REDUCED_INDEX", "clustered_data_visual")
OS_SCHEME = os.getenv("OS_SCHEME", "http")
OS_USERNAME = os.getenv("OS_USERNAME", "admin")  # OpenSearch username
OS_PASSWORD = os.getenv("OS_PASSWORD", "admin")  # OpenSearch password
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-tracking")  # Set MLflow URI
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("MLFLOW_TRACKING_USERNAME", "your-username")
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_TRACKING_PASSWORD", "your-password")

# Initialize OpenSearch client with basic authentication
client = OpenSearch(
    hosts=[{'host': OS_HOST, 'port': int(OS_PORT)}],
    http_auth=(OS_USERNAME, OS_PASSWORD),  # Add the username and password here
    use_ssl=(OS_SCHEME == 'https'),  # Enable SSL if the scheme is https
    verify_certs=False,  # Disable SSL verification if using self-signed certificates (Not recommended for production)
    scheme=OS_SCHEME,
    timeout=60
)

# Initialize MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("clustering_experiment")  # Set your experiment name


# Load column weights from an external JSON file
def load_column_weights(weights_file_path):
    with open(weights_file_path, 'r') as f:
        return json.load(f)


# Apply column weights during preprocessing
def preprocess_data(df, column_weights):
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Fill missing values for numeric columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Encode categorical columns
    for col in categorical_cols:
        df[col] = df[col].fillna('missing')  # Replace NaN with 'missing'
        le = LabelEncoder()
        try:
            df[col] = le.fit_transform(df[col].astype(str))
        except Exception as e:
            print(f"Error encoding column {col}: {e}")

    # Combine the numeric and encoded categorical columns into a single DataFrame
    all_columns = list(numeric_cols) + list(categorical_cols)

    # Check if the number of columns after preprocessing matches the original set
    df_processed = df[all_columns]

    # Scale the entire dataset
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_processed)

    # Apply column weights, but only for those with weight greater than 0
    for idx, col in enumerate(all_columns):
        if col in column_weights and column_weights[col] > 0.0:
            df_scaled[:, idx] *= column_weights[col]

    # Return the DataFrame with scaled values
    return pd.DataFrame(df_scaled, columns=all_columns)


# Function to reduce dimensions using t-SNE or PCA
def reduce_dimensions(df, method="tsne", n_components=2):
    if method == "pca":
        reducer = PCA(n_components=n_components)
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, perplexity=30, learning_rate=200)
    else:
        raise ValueError("Unknown method")

    reduced_data = reducer.fit_transform(df)
    return pd.DataFrame(reduced_data, columns=[f"dim_{i+1}" for i in range(n_components)])

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


# Function to reassign noise points (-1) using K-Means if NearestNeighbors fails
def reassign_noise_points_with_kmeans(df_preprocessed, clusters, n_clusters=None):
    noise_indices = (clusters == -1)

    if sum(noise_indices) == 0:
        return clusters

    # If there are fewer clusters than points, reassign noise points using K-Means
    if n_clusters is None:
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

    # Fit K-Means on the points that are not noise
    kmeans = KMeans(n_clusters=n_clusters)
    non_noise_indices = (clusters != -1)
    kmeans.fit(df_preprocessed[non_noise_indices])

    # Predict cluster labels for noise points
    noise_clusters = kmeans.predict(df_preprocessed[noise_indices])

    # Assign K-Means clusters to the noise points
    clusters[noise_indices] = noise_clusters

    return clusters

# Function to reduce dimensions using UMAP and PCA
def reduce_dimensions_optimal(df, n_components_pca=30, n_components_umap=2):
    # Step 1: Apply PCA to reduce to n_components_pca
    pca = PCA(n_components=n_components_pca)
    df_pca = pca.fit_transform(df)

    # Step 2: Apply UMAP to reduce to 2 components for visualization
    reducer = umap.UMAP(n_components=n_components_umap)
    df_umap = reducer.fit_transform(df_pca)

    return pd.DataFrame(df_umap, columns=[f"dim_{i+1}" for i in range(n_components_umap)])


# Function to reassign noise points using LOF for outliers
def handle_outliers_with_lof(df_preprocessed, clusters):
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    is_inlier = lof.fit_predict(df_preprocessed)  # -1 is an outlier, 1 is an inlier

    # Mark points as noise that are outliers according to LOF
    for i, inlier in enumerate(is_inlier):
        if inlier == -1:
            clusters[i] = -1  # Mark as noise

    return clusters


# Function to evaluate clustering quality
def evaluate_clustering(df_preprocessed, clusters):
    # Silhouette Score
    silhouette_avg = silhouette_score(df_preprocessed, clusters)

    # Calinski-Harabasz Score
    ch_score = calinski_harabasz_score(df_preprocessed, clusters)

    # Davies-Bouldin Score
    db_score = davies_bouldin_score(df_preprocessed, clusters)

    return silhouette_avg, ch_score, db_score

# Example usage in the create_clusters function
@app.post("/create-clusters/")
async def create_clusters(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))

    column_weights = load_column_weights('/app/column_weights.json')
    df_preprocessed = preprocess_data(df, column_weights)

    # Dynamically find optimal eps and min_samples
    optimal_eps, optimal_min_samples = find_optimal_dbscan(df_preprocessed)

    # Start an MLflow run
    with mlflow.start_run() as run:
        mlflow.log_param("eps", optimal_eps)
        mlflow.log_param("min_samples", optimal_min_samples)

        # Apply DBSCAN clustering
        clustering_model = DBSCAN(eps=optimal_eps, min_samples=optimal_min_samples)
        clusters = clustering_model.fit_predict(df_preprocessed)

        # Assign noise points (-1) to nearest cluster
        clusters = assign_noise_points(df_preprocessed, clusters)

        # Save clusters to the dataframe
        df['cluster'] = clusters

        # Optimal dimensionality reduction using PCA + UMAP
        df_reduced = reduce_dimensions_optimal(df_preprocessed)

        # Store original data with clusters in OpenSearch
        for index, row in df.iterrows():
            doc = row.to_dict()
            client.index(index=OS_INDEX, id=index, body=doc)

        # Store reduced-dimension data with cluster labels in OpenSearch
        for index, row in df_reduced.iterrows():
            doc = row.to_dict()
            doc['cluster'] = df['cluster'].iloc[index]
            client.index(index=REDUCED_INDEX, id=index, body=doc)

        # Calculate cluster evaluation metrics
        silhouette_avg = silhouette_score(df_preprocessed, clusters)
        ch_score = calinski_harabasz_score(df_preprocessed, clusters)
        db_score = davies_bouldin_score(df_preprocessed, clusters)

        # Log metrics in MLflow
        mlflow.log_metric("silhouette_score", silhouette_avg)
        mlflow.log_metric("calinski_harabasz_score", ch_score)
        mlflow.log_metric("davies_bouldin_score", db_score)

        # Log the DBSCAN model in MLflow
        mlflow.sklearn.log_model(clustering_model, "dbscan_model")

    return {
        "message": f"Clusters created with final eps={optimal_eps}, min_samples={optimal_min_samples}, and stored in {OS_INDEX}. Reduced dimension data stored in {REDUCED_INDEX}.",
        "silhouette_score": silhouette_avg,
        "calinski_harabasz_score": ch_score,
        "davies_bouldin_score": db_score
    }

# Endpoint to classify a single record using OpenSearch k-NN
@app.post("/classify-record/")
async def classify_record(record: dict):
    record_data = pd.DataFrame([record])
    column_weights = load_column_weights('/app/column_weights.json')

    # Preprocess the record to get the feature vector
    record_preprocessed = preprocess_data(record_data, column_weights)

    # Convert the preprocessed record to a list (vector) for k-NN search
    vector = record_preprocessed.values[0].tolist()

    # Perform k-NN search in OpenSearch to find the nearest cluster
    k_nn_query = {
        "size": 1,  # Retrieve the closest neighbor
        "_source": False,  # Optionally, control which fields are returned
        "knn": {
            "field": "vector",  # Field where you stored your vector data
            "query_vector": vector,  # Vector for comparison
            "k": 1,  # Number of nearest neighbors to retrieve
            "num_candidates": 100  # Optional: number of top candidates to consider
        }
    }

    # Search in OpenSearch using k-NN
    response = client.search(index=OS_INDEX, body=k_nn_query)
    nearest_neighbor = response['hits']['hits'][0]

    # Retrieve the cluster information from the nearest neighbor
    predicted_cluster = nearest_neighbor['_id']  # Or any other relevant cluster field

    return {"cluster": predicted_cluster}
