from fastapi import FastAPI, File, UploadFile
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from opensearchpy import OpenSearch
from io import StringIO
import pandas as pd
import json
import os

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

# Initialize OpenSearch client with basic authentication
client = OpenSearch(
    hosts=[{'host': OS_HOST, 'port': int(OS_PORT)}],
    http_auth=(OS_USERNAME, OS_PASSWORD),  # Add the username and password here
    use_ssl=(OS_SCHEME == 'https'),  # Enable SSL if the scheme is https
    verify_certs=False,  # Disable SSL verification if using self-signed certificates (Not recommended for production)
    scheme=OS_SCHEME,
    timeout=60
)

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


# Function to reduce dimensions for each cluster
def reduce_dimensions_by_clusters(df, clusters, method="tsne", n_components=2):
    df_reduced = pd.DataFrame()

    unique_clusters = set(clusters)
    for cluster in unique_clusters:
        cluster_data = df[clusters == cluster]

        # Adjust perplexity based on the number of samples in the cluster
        perplexity = min(30, len(cluster_data) - 1)  # Ensuring perplexity is less than the number of samples

        if method == "pca":
            reducer = PCA(n_components=n_components)
        elif method == "tsne":
            reducer = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=200)
        else:
            raise ValueError("Unknown method")

        reduced_data = reducer.fit_transform(cluster_data)
        reduced_df = pd.DataFrame(reduced_data, columns=[f"dim_{i + 1}" for i in range(n_components)])
        reduced_df['cluster'] = cluster

        df_reduced = pd.concat([df_reduced, reduced_df])

    return df_reduced.reset_index(drop=True)

# Function to find optimal eps and min_samples
def find_optimal_dbscan(df_preprocessed, min_eps=0.1, max_eps=30.0, step_eps=0.5, min_min_samples=2, max_min_samples=10):
    optimal_eps = min_eps
    optimal_min_samples = min_min_samples
    max_clusters = 1  # Ensure we don't get all noise or a single cluster

    # Iterate over eps values
    while optimal_eps <= max_eps and max_clusters <= 1:
        # Iterate over min_samples values
        for min_samples in range(min_min_samples, max_min_samples + 1):
            clustering_model = DBSCAN(eps=optimal_eps, min_samples=min_samples)
            clusters = clustering_model.fit_predict(df_preprocessed)

            # Exclude noise (-1) when counting clusters
            unique_clusters = set(clusters)
            if -1 in unique_clusters:
                unique_clusters.remove(-1)

            # If more than one cluster is found, stop and return the parameters
            if len(unique_clusters) > 1:
                max_clusters = len(unique_clusters)
                optimal_min_samples = min_samples
                break

        # If more than one cluster was found, stop
        if max_clusters > 1:
            break
        # Otherwise, increase eps and try again
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

# Endpoint to create clusters from CSV
@app.post("/create-clusters/")
async def create_clusters(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))

    # Load column weights
    column_weights = load_column_weights('/app/column_weights.json')

    # Preprocess data
    df_preprocessed = preprocess_data(df, column_weights)

    # Dynamically find optimal eps and min_samples
    optimal_eps, optimal_min_samples = find_optimal_dbscan(df_preprocessed)

    # Apply DBSCAN clustering
    clustering_model = DBSCAN(eps=optimal_eps, min_samples=optimal_min_samples)
    clusters = clustering_model.fit_predict(df_preprocessed)

    # Assign noise points (-1) to nearest cluster
    clusters = assign_noise_points(df_preprocessed, clusters)

    # Save clusters to the dataframe
    df['cluster'] = clusters

    # Reduce dimensions by clusters using t-SNE
    df_reduced = reduce_dimensions_by_clusters(df_preprocessed, clusters, method="tsne", n_components=2)

    # Store original data with clusters in OpenSearch
    for index, row in df.iterrows():
        doc = row.to_dict()
        client.index(index=OS_INDEX, id=index, body=doc)

    # Store reduced-dimension data with cluster labels in OpenSearch
    for index, row in df_reduced.iterrows():
        doc = row.to_dict()
        client.index(index=REDUCED_INDEX, id=index, body=doc)

    return {
        "message": f"Clusters created with final eps={optimal_eps}, min_samples={optimal_min_samples}, and stored in {OS_INDEX}. Reduced dimension data stored in {REDUCED_INDEX}."
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
