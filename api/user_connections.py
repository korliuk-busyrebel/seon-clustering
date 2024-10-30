from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from utils.opensearch_client import client, router
from utils.column_weights import load_column_weights
import numpy as np

# Load column weights for closeness calculations
column_weights = load_column_weights('/app/utils/column_weights.json')

# Define request models
class ConnectionRequest(BaseModel):
    user_id: str
    min_closeness: float = 0.5  # Minimum closeness as a percentage
    k: int = 10  # Default nearest neighbors
    index: str = "clustered_knn_data"  # Default index name, can be overridden

@router.post("/user-connections/")
async def user_connections(request: ConnectionRequest):
    user_index = request.index

    # Retrieve user data by `id` field and get vector
    try:
        user_data_query = {"query": {"term": {"id": request.user_id}}}
        user_search = client.search(index=user_index, body=user_data_query)

        if not user_search['hits']['hits']:
            raise HTTPException(status_code=404, detail=f"User {request.user_id} not found in index {user_index}.")

        user_data = user_search['hits']['hits'][0]["_source"]
        user_vector = user_data.get("vector", [])

        # Ensure the vector matches the expected 898 dimensions
        if not user_vector or len(user_vector) != 898:
            raise HTTPException(status_code=400, detail="User vector has invalid dimensions.")

    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Error retrieving data for user with id {request.user_id}: {e}")

    # Perform KNN search with this vector
    knn_query = {
        "size": request.k,
        "query": {
            "knn": {
                "vector": {
                    "vector": user_vector,
                    "k": request.k
                }
            }
        }
    }

    try:
        response = client.search(index=user_index, body=knn_query)
        connections = []

        for hit in response['hits']['hits']:
            connected_user_data = hit["_source"]
            connected_user_id = connected_user_data.get("id")
            connected_user_vector = connected_user_data.get("vector", [])

            # Skip if vector is missing or incorrect dimension
            if not connected_user_vector or len(connected_user_vector) != 898:
                continue

            # Calculate cosine similarity between vectors
            closeness_score = np.dot(user_vector, connected_user_vector) / (
                np.linalg.norm(user_vector) * np.linalg.norm(connected_user_vector)
            )

            # Adjust closeness based only on vector similarity
            if closeness_score >= request.min_closeness:
                connections.append({
                    "user_id": connected_user_id,
                    "closeness": round(closeness_score * 100, 2),  # Present closeness as percentage
                    "user_name": connected_user_data.get("user_name", "N/A"),
                    "earliest_shared_date": connected_user_data.get("share_date")
                })

        # Sort results by highest closeness score
        connections = sorted(connections, key=lambda x: -x["closeness"])

        return {"connected_users": connections}

    except Exception as e:
        return {"error": str(e)}



# Helper function to calculate closeness
def calculate_closeness(shared_values, weights):
    return sum(weights.get(feature, 1) for feature in shared_values)


def extract_shared_values(user_data, connected_user_data):
    shared_values = {key: value for key, value in user_data.items() if
                     key in connected_user_data and user_data[key] == connected_user_data[key]}
    return shared_values, len(shared_values)

# Export the router for use in the main app
user_connections = router
