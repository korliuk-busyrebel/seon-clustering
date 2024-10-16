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
ES_INDEX = os.getenv("ES_INDEX", "clustered_data")
ES_SCHEME = os.getenv("ES_SCHEME", "http")

# Convert ES_PORT to an integer
es = Elasticsearch(hosts=[{
    'host': ES_HOST,
    'port': int(ES_PORT),  # Ensure the port is passed as an integer
    'scheme': ES_SCHEME
}])


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

    # Now combine the numeric and encoded categorical columns
    all_columns = numeric_cols.union(categorical_cols)

    # Scale the entire dataset
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[all_columns])

    # Apply column weights
    for idx, col in enumerate(all_columns):
        if col in column_weights:
            df_scaled[:, idx] *= column_weights[col]

    return pd.DataFrame(df_scaled, columns=all_columns)


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
        es.index(index=ES_INDEX, id=index, body=doc)

    return {"message": f"Clusters created and stored in OpenSearch index {ES_INDEX}"}


# Endpoint to classify a single record into clusters
@app.post("/classify-record/")
async def classify_record(record: dict):
    record_data = pd.DataFrame([record])
    column_weights = load_column_weights('/app/column_weights.json')

    record_preprocessed = preprocess_data(record_data, column_weights)

    res = es.search(index=ES_INDEX, size=10000)
    clustered_data = [hit['_source'] for hit in res['hits']['hits']]
    clustered_df = pd.DataFrame(clustered_data)

    clustered_df_preprocessed = preprocess_data(clustered_df, column_weights)

    clustering_model = DBSCAN(eps=0.5, min_samples=5)
    clustering_model.fit(clustered_df_preprocessed)

    predicted_cluster = clustering_model.fit_predict(
        pd.concat([clustered_df_preprocessed, record_preprocessed])
    )[-1]

    return {"cluster": int(predicted_cluster)}
