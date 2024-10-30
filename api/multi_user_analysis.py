from fastapi import APIRouter
from pydantic import BaseModel
from utils.opensearch_client import client, router
from utils.column_weights import load_column_weights
import numpy as np
from datetime import datetime

# Load column weights for closeness calculations
column_weights = load_column_weights('/app/utils/column_weights.json')

# Define request model with an optional index parameter
class MultiUserRequest(BaseModel):
    user_ids: list
    min_closeness: float = 0.5  # Minimum closeness threshold
    index: str = "clustered_knn_data"  # Default index name, can be overridden in the request


@router.post("/multi-user-analysis/")
async def multi_user_analysis(request: MultiUserRequest):
    results = {}

    for user_id in request.user_ids:
        user_index = request.index

        # Retrieve user data by `id` field
        try:
            user_data_query = {"query": {"term": {"id": user_id}}}
            user_search = client.search(index=user_index, body=user_data_query)

            if not user_search['hits']['hits']:
                print(f"User with id {user_id} not found in index {user_index}.")
                results[user_id] = {"error": f"User {user_id} not found"}
                continue

            user_data = user_search['hits']['hits'][0]["_source"]
            user_vector = user_data.get("vector", [])
            if not user_vector or len(user_vector) != 898:
                print(f"User {user_id} vector is missing or incorrect dimension.")
                results[user_id] = {"error": "Invalid vector dimensions"}
                continue

        except Exception as e:
            print(f"Error retrieving data for user with id {user_id}: {e}")
            results[user_id] = {"error": str(e)}
            continue

        user_cluster_id = user_data.get("cluster")
        if user_cluster_id is None:
            results[user_id] = {"error": "Cluster ID is missing"}
            continue

        # Perform KNN search for each user to find nearest neighbors
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

                if connected_user_id == user_id:
                    continue  # Exclude self from results

                connected_user_vector = connected_user_data.get("vector", [])
                if not connected_user_vector or len(connected_user_vector) != 898:
                    print(f"Skipping connected user {connected_user_id} due to invalid vector.")
                    continue

                # Calculate cosine similarity between vectors
                closeness_score = np.dot(user_vector, connected_user_vector) / (
                    np.linalg.norm(user_vector) * np.linalg.norm(connected_user_vector)
                )

                shared_values = {
                    key: user_data[key] for key in user_data.keys()
                    if key in connected_user_data and user_data[key] == connected_user_data[key] and key in column_weights
                }
                num_shared_values = len(shared_values)
                shared_value_score = sum(column_weights.get(key, 1) for key in shared_values) / sum(column_weights.values())

                # Final closeness score combining vector similarity and shared values
                final_closeness = 0.7 * closeness_score + 0.3 * shared_value_score

                # Apply minimum closeness filter
                if final_closeness >= request.min_closeness:
                    connections.append({
                        "user_id": connected_user_id,
                        "closeness": round(final_closeness * 100, 2),
                        "user_name": connected_user_data.get("user_name", "N/A"),
                        "shared_values": shared_values,
                        "num_shared_values": num_shared_values,
                        "earliest_date_of_sharing": connected_user_data.get("share_date", datetime.now().isoformat())
                    })
                    print(f"Connected User ID: {connected_user_id}, Closeness: {final_closeness}, Shared Values: {shared_values}")

            # Sort connections by closeness and add to results per user
            results[user_id] = sorted(connections, key=lambda x: -x["closeness"])

        except Exception as e:
            print(f"Error retrieving connections for user {user_id}: {e}")
            results[user_id] = {"error": str(e)}

    return {"user_connections": results}

# Export the router for integration into the main FastAPI app
multi_user_analysis = router
