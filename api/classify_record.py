from pydantic import BaseModel
import pandas as pd
from services.preprocessing import preprocess_data
from utils.opensearch_client import client, router
from utils.column_weights import load_column_weights



# Define request model
class ClassifyRequest(BaseModel):
    record: dict
    k: int = 10  # Default to 10 nearest neighbors

@router.post("/classify-record/")
async def classify_record(request: ClassifyRequest):
    # Load column weights to define required fields (all 898 fields expected)
    column_weights = load_column_weights('/app/utils/column_weights.json')
    required_fields = list(column_weights.keys())

    # Fill missing fields with default value (0) to ensure exactly 898 fields in input
    input_record = {**{field: 0 for field in required_fields}, **request.record}
    record_data = pd.DataFrame([input_record])

    # Preprocess the record to get the feature vector
    record_preprocessed = preprocess_data(record_data, column_weights)

    # Convert the preprocessed record to a list (vector) for k-NN search
    vector = record_preprocessed.iloc[0].tolist()

    # Check if the vector matches the expected 898 dimensions
    if len(vector) < 898:
        # Pad vector with zeros to reach 898 dimensions
        vector.extend([0] * (898 - len(vector)))
    elif len(vector) > 898:
        # Trim vector to 898 if it has excess dimensions (unlikely but safe)
        vector = vector[:898]

    # Perform k-NN search with the specified number of nearest neighbors
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

    # Execute search and handle the response
    try:
        response = client.search(index=OS_KNN_INDEX, body=knn_query)
        knn_results = [
            {
                "id": hit["_id"],
                "score": hit["_score"],
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
