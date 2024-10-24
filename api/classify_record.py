from fastapi import APIRouter
import pandas as pd
from services.preprocessing import preprocess_data
from services.opensearch_client import classify_record_opensearch, knn_search_opensearch
from utils.column_weights import load_column_weights

router = APIRouter()

@app.post("/classify-record/")
async def classify_record(record: dict):
    record_data = pd.DataFrame([record])
    column_weights = load_column_weights('/app/utils/column_weights.json')

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
            "k": 2,  # Number of nearest neighbors to retrieve
            "num_candidates": 10  # Optional: number of top candidates to consider
        }
    }

    # Search in OpenSearch using k-NN
    response = client.search(index=OS_INDEX, body=k_nn_query)
    nearest_neighbor = response['hits']['hits'][0]

    # Retrieve the cluster information from the nearest neighbor
    predicted_cluster = nearest_neighbor['_id']  # Or any other relevant cluster field

    return {"cluster": predicted_cluster}

# Export the router for use in the main app
classify_record = router
