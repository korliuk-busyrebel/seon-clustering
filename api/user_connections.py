from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
from services.preprocessing import preprocess_data
from utils.opensearch_client import client, router, OS_KNN_INDEX
from utils.column_weights import load_column_weights

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
        user_vector = preprocess_data(pd.DataFrame([user_data]), column_weights).iloc[0].tolist()

        # Ensure the vector matches the expected 898 dimensions
        if len(user_vector) < 898:
            user_vector.extend([0] * (898 - len(user_vector)))
        elif len(user_vector) > 898:
            user_vector = user_vector[:898]

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

            # Calculate shared values and closeness
            shared_values, num_shared_values = extract_shared_values(user_data, connected_user_data)
            closeness = calculate_closeness(shared_values, column_weights)

            # Apply minimum closeness filter
            if closeness >= request.min_closeness:
                connections.append({
                    "user_id": connected_user_id,
                    "closeness": round(closeness * 100, 2),
                    "user_name": connected_user_data.get("user_name", "N/A"),
                    "shared_values": shared_values,
                    "num_shared_values": num_shared_values,
                    "earliest_shared_date": connected_user_data.get("share_date")
                })

        connections = sorted(connections, key=lambda x: -x["closeness"])
        return {"connected_users": connections}

    except Exception as e:
        print(f"Error retrieving user connections: {e}")
        return {"error": str(e)}


# Helper function to calculate closeness
def calculate_closeness(shared_values, weights):
    closeness = 0
    for feature, value in shared_values.items():
        feature_weight = weights.get(feature, 1)
        closeness += feature_weight
    return closeness


def extract_shared_values(user_data, connected_user_data):
    shared_values = {key: value for key, value in user_data.items() if
                     key in connected_user_data and user_data[key] == connected_user_data[key]}
    return shared_values, len(shared_values)

# Export the router for use in the main app
user_connections = router
