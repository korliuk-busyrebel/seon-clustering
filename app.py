from fastapi import FastAPI, File, UploadFile
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from elasticsearch import Elasticsearch
from io import StringIO
import pandas as pd
import json
import os

# Initialize FastAPI app
app = FastAPI()

# Get environment variables for Elasticsearch connection and index
ES_HOST = os.getenv("ES_HOST", "localhost")
ES_PORT = os.getenv("ES_PORT", 9200)
ES_INDEX = os.getenv("ES_INDEX", "clustered_data")  # Default index name if not provided

# Connect to Elasticsearch (OpenSearch compatible)
es = Elasticsearch(hosts=[{'host': ES_HOST, 'port': ES_PORT}])


# Load column weights from an external JSON file
def load_column_weights(weights_file_path):
    with open(weights_file_path, 'r') as f:
        return json.load(f)


# Apply column weights during preprocessing
def preprocess_data(df, column_weights):
    # Normalize numerical data
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)  # Handle missing data

    # Encode categorical/string columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna('missing')  # Fill NaN with 'missing'
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Convert strings to numbers

    # Scale the entire dataset
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # Apply weights to each column
    for col in df.columns:
        if col in column_weights:
            df_scaled[:, df.columns.get_loc(col)] *= column_weights[col]

    return pd.DataFrame(df_scaled, columns=df.columns)


# Endpoint to create initial clusters from CSV
@app.post("/create-clusters/")
async def create_clusters(file: UploadFile = File(...)):
    # Read the CSV file
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))

    # Load the column weights from an external file
    column_weights = load_column_weights('/app/column_weights.json')  # Adjust path as needed

    # Preprocess the data (normalize, encode, and apply weights)
    df_preprocessed = preprocess_data(df, column_weights)

    # Apply DBSCAN clustering
    clustering_model = DBSCAN(eps=0.5, min_samples=5)
    df['cluster'] = clustering_model.fit_predict(df_preprocessed)

    # Store clusters in OpenSearch
    for index, row in df.iterrows():
        doc = row.to_dict()
        es.index(index=ES_INDEX, id=index, body=doc)

    return {"message": f"Clusters created and stored in OpenSearch index {ES_INDEX}"}


# Endpoint to classify a single record into clusters
@app.post("/classify-record/")
async def classify_record(record: dict):
    # Convert record into DataFrame
    record_data = pd.DataFrame([record])

    # Load the column weights from an external file
    column_weights = load_column_weights('/app/column_weights.json')  # Adjust path as needed

    # Preprocess the data (normalize, encode, and apply weights)
    record_preprocessed = preprocess_data(record_data, column_weights)

    # Fetch the latest cluster data from OpenSearch
    res = es.search(index=ES_INDEX, size=10000)  # Limit to 10000 for now
    clustered_data = [hit['_source'] for hit in res['hits']['hits']]
    clustered_df = pd.DataFrame(clustered_data)

    # Preprocess fetched clustered data
    clustered_df_preprocessed = preprocess_data(clustered_df, column_weights)

    # Use the previously fitted DBSCAN model to predict the cluster
    clustering_model = DBSCAN(eps=0.5, min_samples=5)
    clustering_model.fit(clustered_df_preprocessed)

    # Classify the new record
    predicted_cluster = clustering_model.fit_predict(
        pd.concat([clustered_df_preprocessed, record_preprocessed])
    )[-1]

    return {"cluster": int(predicted_cluster)}
