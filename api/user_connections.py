from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
from services.preprocessing import preprocess_data
from utils.opensearch_client import client, router, OS_KNN_INDEX
from utils.column_weights import load_column_weights


# Define request models
class ConnectionRequest(BaseModel):
    user_id: str
    min_closeness: float = 0.5  # Minimum closeness as a percentage
    k: int = 10  # Default nearest neighbors


@router.post("/user-connections/")
async def get_user_connections(request: ConnectionRequest):
    # Load the investigated user data from OpenSearch
    user_data = client.get(index=OS_KNN_INDEX, id=request.user_id)["_source"]
    column_weights = load_column_weights('/app/utils/column_weights.json')

    if not user_data:
        raise HTTPException(status_code=404, detail=f"User {request.user_id} not found.")

    # Preprocess the investigated user's data
    user_vector = preprocess_data(pd.DataFrame([user_data]), column_weights).iloc[0].tolist()

    # KNN search to find connected users
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
        # Perform the search
        response = client.search(index=OS_KNN_INDEX, body=knn_query)
        connections = []

        for hit in response['hits']['hits']:
            connected_user_data = hit["_source"]
            shared_values, num_shared_values = extract_shared_values(user_data, connected_user_data)
            closeness = calculate_closeness(shared_values, column_weights)

            # Apply minimum closeness filter
            if closeness >= request.min_closeness:
                connections.append({
                    "user_id": hit["_id"],
                    "num_users_in_cluster": connected_user_data.get("num_users_in_cluster", 1),
                    "closeness": round(closeness * 100, 2),
                    "user_name": connected_user_data.get("user_name", "N/A"),
                    "shared_values": shared_values,
                    "num_shared_values": num_shared_values,
                    "earliest_shared_date": connected_user_data.get("earliest_shared_date")
                })

        # Rank connections by closeness, with higher weights prioritizing rare shared connections
        connections = sorted(connections, key=lambda x: -x["closeness"])

        return {"connected_users": connections}
    except Exception as e:
        print(f"Error retrieving user connections: {e}")
        return {"error": str(e)}


# Helper function to calculate closeness
def calculate_closeness(shared_values, weights):
    closeness = 0
    for feature, value in shared_values.items():
        # Prioritize rare features by weighting
        feature_weight = weights.get(feature, 1)  # Default weight is 1 if not found in weights
        closeness += feature_weight
    return closeness


# Extract shared values based on the intersection of non-zero features
def extract_shared_values(user_data, connected_user_data):
    shared_values = {key: value for key, value in user_data.items() if
                     key in connected_user_data and user_data[key] == connected_user_data[key]}
    return shared_values, len(shared_values)


# Export the router for use in the main app
user_connections = router
