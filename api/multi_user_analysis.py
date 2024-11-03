from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from utils.opensearch_client import client, router
from utils.column_weights import load_column_weights
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

        # Retrieve user data by `id` field
        try:
            print(f"Fetching data for user_id: {user_id}")
            user_data_query = {"query": {"term": {"id": user_id}}}
            user_search = client.search(index=user_index, body=user_data_query)

            if not user_search['hits']['hits']:
                print(f"User with id {user_id} not found in index {user_index}.")
                continue

            user_data = user_search['hits']['hits'][0]["_source"]
            user_vector = user_data.get("vector", [])

            # Check and convert user_vector to ensure it's a list of floats
            if isinstance(user_vector, str):
                print(f"Converting user_vector from string to list for user_id: {user_id}")
                user_vector = eval(user_vector)  # Be cautious with eval in production
            user_vector = [float(x) for x in user_vector]

            print(f"user_vector for user_id {user_id} is now: {user_vector[:10]}...")  # Print first 10 values

            # Ensure `user_vector` is a list of the correct length
            if not isinstance(user_vector, list) or len(user_vector) != 898:
                print(f"User vector for user_id {user_id} is missing or has incorrect dimensions.")
                continue

        except Exception as e:
            print(f"Error retrieving data for user with id {user_id}: {e}")
            continue

        # Perform KNN search using OpenSearch KNN plugin
        knn_query = {
            "size": request.k,
            "query": {
                "knn": {
                    "field": "vector",
                    "query_vector": user_vector,
                    "k": request.k,
                    "num_candidates": request.k * 2  # Adjust based on accuracy/performance needs
                }
            }
        }

        try:
            print(f"Performing KNN search for user_id: {user_id}")
            cluster_response = client.search(index=user_index, body=knn_query)
            cluster_users = cluster_response['hits']['hits']
            print(f"Found {len(cluster_users)} users in cluster for user_id {user_id}")

            closest_users = []
            for connected_user in cluster_users:
                connected_user_data = connected_user["_source"]
                connected_user_id = connected_user_data.get("id")

                # Exclude the requested user_id from results
                if connected_user_id == user_id:
                    continue

                # Get similarity score from OpenSearch's KNN plugin
                similarity_score = connected_user["_score"]
                print(f"Similarity score for connected_user_id {connected_user_id}: {similarity_score}")

                connected_user_vector = connected_user_data.get("vector", [])
                if not isinstance(connected_user_vector, list) or len(connected_user_vector) != 898:
                    print(f"Skipping connected_user_id {connected_user_id} due to invalid vector.")
                    continue

                # Extract shared values
                shared_values = {
                    column_names[i]: user_vector[i]
                    for i in range(len(user_vector))
                    if user_vector[i] == connected_user_vector[i] and column_weights.get(column_names[i], 0.0) != 0.0
                }
                num_shared_values = len(shared_values)
                print(f"Shared values for connected_user_id {connected_user_id}: {shared_values}")

                # Calculate a final closeness score
                shared_value_score = sum(column_weights.get(key, 1) for key in shared_values) / sum(column_weights.values())
                final_closeness = 0.7 * similarity_score + 0.3 * shared_value_score
                final_closeness = min(final_closeness * 100, 100)

                # Apply minimum closeness filter
                if final_closeness >= request.min_closeness:
                    closest_users.append({
                        "user_id": connected_user_id,
                        "closeness": round(final_closeness, 2),
                        "user_name": connected_user_data.get("user_name", "N/A"),
                        "shared_values": shared_values,
                        "num_shared_values": num_shared_values,
                        "earliest_date_of_sharing": connected_user_data.get("share_date", datetime.now().isoformat())
                    })

            results.append({user_id: closest_users})

        except Exception as e:
            print(f"Error retrieving KNN results for user with id {user_id}: {e}")
            continue

    return {"connected_users_per_user": results}


# Export the router for integration into the main FastAPI app
multi_user_analysis = router
