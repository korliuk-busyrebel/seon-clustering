from fastapi import FastAPI, File, UploadFile
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from opensearchpy import OpenSearch  # Updated import
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


# Function to reduce dimensions using PCA or t-SNE
def reduce_dimensions(df, method="pca", n_components=2):
    if method == "pca":
        reducer = PCA(n_components=n_components)
    elif method == "tsne":
        reducer = TSNE(n_components=n_components)
    else:
        raise ValueError("Unknown method")

    reduced_data = reducer.fit_transform(df)
    return pd.DataFrame(reduced_data, columns=[f"dim_{i+1}" for i in range(n_components)])


# Function to dynamically adjust eps and min_samples until all points are clustered
def cluster_with_dbscan(df_preprocessed, initial_eps=0.5, min_samples=5, max_eps=20.0, step=0.5):
    eps = initial_eps
    while eps <= max_eps:
        # Apply DBSCAN clustering
        clustering_model = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = clustering_model.fit_predict(df_preprocessed)

        # If all points are assigned to clusters (no points with label -1), break the loop
        if -1 not in clusters:
            return clusters, eps

        # Increase eps and try again
        eps += step

    # If max_eps is reached and still not all points are clustered, return the best result
    return clusters, eps

# Endpoint to create initial clusters from CSV and store reduced-dimension data
@app.post("/create-clusters/")
async def create_clusters(file: UploadFile = File(...), initial_eps: float = 0.5, min_samples: int = 5):
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))

    # Load column weights for preprocessing
    column_weights = load_column_weights('/app/column_weights.json')

    # Preprocess data
    df_preprocessed = preprocess_data(df, column_weights)

    # Apply dynamic DBSCAN clustering (adjust eps and min_samples)
    clusters, final_eps = cluster_with_dbscan(df_preprocessed, initial_eps=initial_eps, min_samples=min_samples)

    # Assign clusters to the dataframe
    df['cluster'] = clusters

    # Reduce dimensions using PCA (you can also use t-SNE)
    df_reduced = reduce_dimensions(df_preprocessed, method="pca", n_components=2)

    # Store original data with clusters in the main index
    for index, row in df.iterrows():
        doc = row.to_dict()
        client.index(index=OS_INDEX, id=index, body=doc)  # Store in the main index

    # Store reduced-dimension data in another OpenSearch index
    for index, row in df_reduced.iterrows():
        doc = row.to_dict()
        doc['cluster'] = df['cluster'].iloc[index]  # Add the cluster label
        client.index(index=REDUCED_INDEX, id=index, body=doc)  # Store in the reduced-dimension index

    return {
        "message": f"Clusters created with final eps={final_eps} and stored in {OS_INDEX}. Reduced dimension data stored in {REDUCED_INDEX}."
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
