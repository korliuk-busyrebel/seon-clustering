from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
from services.preprocessing import preprocess_data
from utils.column_weights import load_column_weights
from opensearchpy import OpenSearch
import urllib3
import os

# Suppress the InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

router = APIRouter()

# Initialize OpenSearch client
OS_HOST = os.getenv("OS_HOST", "localhost")
OS_PORT = os.getenv("OS_PORT", 9200)
OS_KNN_INDEX = os.getenv("OS_KNN_INDEX", "clustered_knn_data")
OS_SCHEME = os.getenv("OS_SCHEME", "http")
OS_USERNAME = os.getenv("OS_USERNAME", "admin")
OS_PASSWORD = os.getenv("OS_PASSWORD", "admin")

client = OpenSearch(
    hosts=[{'host': OS_HOST, 'port': int(OS_PORT)}],
    http_auth=(OS_USERNAME, OS_PASSWORD),
    use_ssl=(OS_SCHEME == 'https'),
    verify_certs=False,
    scheme=OS_SCHEME,
    timeout=60
)

# Define request model
class ClassifyRequest(BaseModel):
    record: dict
    k: int = 10  # Default to 10 nearest neighbors

@router.post("/classify-record/")
async def classify_record(request: ClassifyRequest):
    # Define required fields based on the keys in column weights
    required_fields = list(load_column_weights('/app/utils/column_weights.json').keys())

    # Merge `request.record` with default values for missing fields
    record_data = pd.DataFrame([{**{field: 0 for field in required_fields}, **request.record}])

    # Preprocess the record to get the feature vector
    record_preprocessed = preprocess_data(record_data, load_column_weights('/app/utils/column_weights.json'))

    # Convert the preprocessed record to a list (vector) for k-NN search
    vector = record_preprocessed.iloc[0].tolist()

    # Perform k-NN search in OpenSearch to find the nearest clusters
    knn_query = {
        "size": request.k,
        "query": {
            "knn": {
                "vector": {
                    "vector": vector,
                    "k": request.k
                }
            }
        }
    }

    # Perform search and handle the response
    try:
        response = client.search(index=OS_KNN_INDEX, body=knn_query)
        knn_results = [
            {
                "id": hit["_id"],
                "score": hit["_score"],
                "vector": hit["_source"].get("vector", []),
                "cluster": hit["_source"].get("cluster")
            }
            for hit in response['hits']['hits']
        ]
        return {"nearest_neighbors": knn_results}
    except Exception as e:
        print(f"Error during KNN search: {e}")
        return {"error": str(e)}

# Export the router for use in the main app
classify_record = router
