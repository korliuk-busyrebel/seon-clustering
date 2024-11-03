from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from utils.opensearch_client import client, router
from utils.column_weights import load_column_weights
from datetime import datetime

# Load column weights for closeness calculations
column_weights = load_column_weights('/app/utils/column_weights.json')
column_names = list(column_weights.keys())  # Get the list of feature names in the correct order


# Define request models
class ConnectionRequest(BaseModel):
    user_id: str
    min_closeness: float = 0.5  # Minimum closeness as a percentage
    k: int = 10  # Default nearest neighbors
    index: str = "clustered_knn_data"  # Default index name, can be overridden


def get_shared_values(user_vector, connected_user_vector, column_names, column_weights):
    """
    Extracts shared values between two vectors, ignoring zero-weight fields.
    Returns a dictionary of shared field names and values.
    """
    shared_values = {
        column_names[i]: user_vector[i]
        for i in range(len(user_vector))
        if user_vector[i] == connected_user_vector[i] and column_weights.get(column_names[i], 0.0) != 0.0
    }
    return shared_values


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

        # Validate user_vector to ensure it's a list
        if not isinstance(user_vector, list) or len(user_vector) != 898:
            raise HTTPException(status_code=400, detail="User vector has invalid dimensions.")

    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Error retrieving data for user with id {request.user_id}: {e}")

    # Perform KNN search to get top `k` most similar users
    knn_query = {
        "size": request.k,
        "query": {
            "knn": {
                "field": "vector",
                "query_vector": user_vector,  # Make sure user_vector is an array
                "k": request.k,
                "num_candidates": request.k * 2  # Adjust based on accuracy/performance needs
            }
        }
    }

    try:
        knn_response = client.search(index=user_index, body=knn_query)
        connections = []

        for hit in knn_response['hits']['hits']:
            connected_user_data = hit["_source"]
            connected_user_id = connected_user_data.get("id")

            # Exclude the requested user_id from results
            if connected_user_id == request.user_id:
                continue

            # Get similarity score from OpenSearch KNN
            similarity_score = hit["_score"]

            connected_user_vector = connected_user_data.get("vector", [])
            if not isinstance(connected_user_vector, list) or len(connected_user_vector) != 898:
                continue

            # Calculate shared values using helper function
            shared_values = get_shared_values(user_vector, connected_user_vector, column_names, column_weights)
            num_shared_values = len(shared_values)

            # Calculate final closeness score combining OpenSearch similarity and shared values
            shared_value_score = sum(column_weights.get(key, 1) for key in shared_values) / sum(column_weights.values())
            final_closeness = 0.7 * similarity_score + 0.3 * shared_value_score
            final_closeness = min(final_closeness * 100, 100)  # Ensure it's between 0-100

            # Apply minimum closeness filter
            if final_closeness >= request.min_closeness:
                connections.append({
                    "user_id": connected_user_id,
                    "closeness": round(final_closeness, 2),
                    "user_name": connected_user_data.get("user_name", "N/A"),
                    "shared_values": shared_values,
                    "num_shared_values": num_shared_values,
                    "earliest_shared_date": connected_user_data.get("share_date", datetime.now().isoformat())
                })

        # Sort connections by closeness score
        connections = sorted(connections, key=lambda x: -x["closeness"])

        return {"connected_users": connections}

    except Exception as e:
        return {"error": str(e)}


# Export the router for use in the main app
user_connections = router
