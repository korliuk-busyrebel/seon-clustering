from fastapi import FastAPI, File, UploadFile
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
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

    # Fill missing values for numeric columns with the mean
    numeric_imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

    # Fill missing values for categorical columns with the most frequent value
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    for col in categorical_cols:
        df[col] = categorical_imputer.fit_transform(df[[col]]).ravel()
        le = LabelEncoder()
        try:
            df[col] = le.fit_transform(df[col].astype(str))
        except Exception as e:
            print(f"Error encoding column {col}: {e}")

    # Combine numeric and encoded categorical columns
    all_columns = numeric_cols.union(categorical_cols)

    # Filter out columns with weight <= 0.0
    selected_columns = [col for col in all_columns if column_weights.get(col, 0.0) > 0.0]

    if not selected_columns:
        raise ValueError("No columns with weight > 0.0 found.")

    # Scale only the selected columns
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[selected_columns])

    # Apply column weights
    for idx, col in enumerate(selected_columns):
        df_scaled[:, idx] *= column_weights[col]

    return pd.DataFrame(df_scaled, columns=selected_columns)

# Endpoint to create initial clusters from CSV
@app.post("/create-clusters/")
async def create_clusters(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))

    column_weights = load_column_weights('/app/column_weights.json')

    df_preprocessed = preprocess_data(df, column_weights)

    clustering_model = DBSCAN(eps=0.5, min_samples=5)
    df['cluster'] = clustering_model.fit_predict(df_preprocessed)

    for index, row in df.iterrows():
        doc = row.to_dict()
        client.index(index=OS_INDEX, id=index, body=doc)  # Updated client

    return {"message": f"Clusters created and stored in OpenSearch index {OS_INDEX}"}


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
