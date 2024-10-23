from fastapi import FastAPI, File, UploadFile
from io import StringIO
import pandas as pd
from dbscan_cluster import process_dbscan
from preprocessing import preprocess_data, load_column_weights
from opensearch_utils import classify_record_knn

app = FastAPI()


@app.post("/create-clusters/")
async def create_clusters(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))

    # Load column weights and preprocess the data
    column_weights = load_column_weights('/app/column_weights.json')
    df_preprocessed = preprocess_data(df, column_weights)

    # Process DBSCAN clustering
    results = process_dbscan(df_preprocessed, df, column_weights)
    return results


@app.post("/classify-record/")
async def classify_record(record: dict):
    return classify_record_knn(record)
