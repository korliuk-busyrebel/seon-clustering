from fastapi import APIRouter
import pandas as pd
from services.preprocessing import preprocess_data
from services.opensearch_client import classify_record_opensearch
from opensearchpy import OpenSearch
from utils.column_weights import load_column_weights
import urllib3
import os
# Suppress the InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

router = APIRouter()


# Initialize OpenSearch client
OS_HOST = os.getenv("OS_HOST", "localhost")
OS_PORT = os.getenv("OS_PORT", 9200)
OS_INDEX = os.getenv("OS_INDEX", "clustered_data")
REDUCED_INDEX = os.getenv("OS_REDUCED_INDEX", "clustered_data_visual")
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

@router.post("/classify-record/")
async def classify_record(record: dict):
    record_data = pd.DataFrame([record])
    column_weights = load_column_weights('/app/utils/column_weights.json')

    # Preprocess the record to get the feature vector
    record_preprocessed = preprocess_data(record_data, column_weights)

    # Convert the preprocessed record to a list (vector) for k-NN search
    #vector = record_preprocessed.values[0].tolist()
    vector = record_preprocessed.iloc[0].tolist()
    # Perform k-NN search in OpenSearch to find the nearest cluster
    # Construct the correct KNN query
    knn_query = {
        "size": 10,
        "query": {
            "knn": {
                "id": {
                    "vector": vector,
                    "k": 10
                }
            }
        }
    }

    # Perform search
    try:
        response = client.search(index=OS_INDEX, body=knn_query)
        # Extract the relevant result from the response
        knn_result = response['hits']['hits'][0]['_source']
        return knn_result
    except Exception as e:
        print(f"Error during KNN search: {e}")
        return None

# Export the router for use in the main app
classify_record = router
