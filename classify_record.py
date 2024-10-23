from fastapi import FastAPI
import pandas as pd
from config import client, OS_INDEX
from preprocessing import load_column_weights, preprocess_data

@app.post("/classify-record/")
async def classify_record(record: dict):
    record_data = pd.DataFrame([record])
    column_weights = load_column_weights('/app/column_weights.json')
    record_preprocessed = preprocess_data(record_data, column_weights)
    vector = record_preprocessed.values[0].tolist()

    k_nn_query = {
        "size": 1,
        "_source": False,
        "knn": {
            "field": "vector",
            "query_vector": vector,
            "k": 1,
            "num_candidates": 100
        }
    }

    response = client.search(index=OS_INDEX, body=k_nn_query)
    nearest_neighbor = response['hits']['hits'][0]
    predicted_cluster = nearest_neighbor['_id']

    return {"cluster": predicted_cluster}
