from fastapi import FastAPI, File, UploadFile
from io import StringIO
import pandas as pd


from dbscan_cluster import process_dbscan
from preprocess import preprocess_data, load_column_weights
from opensearch_utils import save_to_opensearch, classify_record_knn
from mlflow_utils import start_mlflow_run

app = FastAPI()

@app.post("/create-clusters/")
async def create_clusters(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))

    column_weights = load_column_weights('/app/column_weights.json')
    df_preprocessed = preprocess_data(df, column_weights)

    # Process DBSCAN clustering and log in MLflow
    results = process_dbscan(df_preprocessed, df, column_weights)
    return results

@app.post("/classify-record/")
async def classify_record(record: dict):
    return classify_record_knn(record)
