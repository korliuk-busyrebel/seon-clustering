from fastapi import APIRouter
from pydantic import BaseModel
from utils.opensearch_client import client, router
from utils.column_weights import load_column_weights
import numpy as np
from datetime import datetime

# Load column weights for closeness calculations
column_weights = load_column_weights('/app/utils/column_weights.json')
column_names = list(column_weights.keys())  # Get the list of feature names in the correct order

# Define request model with an optional index parameter
class MultiUserRequest(BaseModel):
    user_ids: list
    min_closeness: float = 0.5  # Minimum closeness threshold
    index: str = "clustered_knn_data"  # Default index name, can be overridden in the request
    k: int = 100  # Default number of users to retrieve from the cluster, can be overridden


@router.post("/multi-user-analysis/")
async def multi_user_analysis(request: MultiUserRequest):
    results = []

    for user_id in request.user_ids:
        user_index = request.index

        # Retrieve user data by `id` field (not `_id`)
        try:
            user_data_query = {"query": {"term": {"id": user_id}}}
            user_search = client.search(index=user_index, body=user_data_query)

            if not user_search['hits']['hits']:
                print(f"User with id {user_id} not found in index {user_index}.")
                continue

            user_data = user_search['hits']['hits'][0]["_source"]
            user_vector = user_data.get("vector", [])
            if not user_vector or len(user_vector) != 898:
                print(f"User {user_id} vector is missing or incorrect dimension.")
                continue

        except Exception as e:
            print(f"Error retrieving data for user with id {user_id}: {e}")
            continue

        user_cluster_id = user_data.get("cluster")
        if user_cluster_id is None:
            print(f"User with id {user_id} does not have a cluster ID.")
            continue

        # Retrieve all users in the same cluster with a customizable size (k)
        cluster_query = {
            "size": request.k,
            "query": {
                "term": {"cluster": user_cluster_id}
            }
        }
        cluster_response = client.search(index=user_index, body=cluster_query)
        cluster_users = cluster_response['hits']['hits']
        num_users_in_cluster = len(cluster_users)

        # Closeness calculation for each user in the cluster
        closest_users = []
        for connected_user in cluster_users:
            connected_user_data = connected_user["_source"]
            connected_user_id = connected_user_data.get("id")

            if connected_user_id == user_id:
                continue

            connected_user_vector = connected_user_data.get("vector", [])
            if not connected_user_vector or len(connected_user_vector) != 898:
                print(f"Skipping connected user {connected_user_id} due to missing or incorrect vector.")
                continue

            # Calculate closeness as cosine similarity
            closeness_score = np.dot(user_vector, connected_user_vector) / (
                np.linalg.norm(user_vector) * np.linalg.norm(connected_user_vector))

            # Extract shared values: match vector indices with feature names, ignore zeros, and exclude features with zero weights
            shared_values = {
                column_names[i]: user_vector[i]
                for i in range(len(user_vector))
                if user_vector[i] == connected_user_vector[i] != 0.0 and column_weights.get(column_names[i], 0.0) != 0.0
            }
            num_shared_values = len(shared_values)

            # Calculate final closeness score, combining vector similarity and shared values
            shared_value_score = sum(column_weights.get(key, 1) for key in shared_values) / sum(column_weights.values())
            final_closeness = 0.7 * closeness_score + 0.3 * shared_value_score

            if final_closeness >= request.min_closeness:
                closest_users.append({
                    "user_id": connected_user_id,
                    "num_users_in_cluster": num_users_in_cluster,
                    "closeness": round(final_closeness * 100, 2),
                    "user_name": connected_user_data.get("user_name", "N/A"),
                    "shared_values": shared_values,
                    "num_shared_values": num_shared_values,
                    "earliest_date_of_sharing": connected_user_data.get("share_date", datetime.now().isoformat())
                })

        # Store closest users for this user_id
        results.append({user_id: closest_users})

    # Return closest users grouped by each requested user ID
    return {"connected_users_per_user": results}

# Export the router for integration into the main FastAPI app
multi_user_analysis = router
