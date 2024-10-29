from pydantic import BaseModel
from utils.opensearch_client import client, router, OS_KNN_INDEX
from utils.column_weights import load_column_weights
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
    results = []

    for user_id in request.user_ids:
        user_index = request.index

        # Retrieve user data by `id` field (instead of _id)
        try:
            # Use OpenSearch query to find the document by `id` field
            user_data_query = {
                "query": {
                    "term": {"id": user_id}
                }
            }
            user_search = client.search(index=user_index, body=user_data_query)
            if not user_search['hits']['hits']:
                print(f"User with id {user_id} not found in index {user_index}.")
                continue

            user_data = user_search['hits']['hits'][0]["_source"]
        except Exception as e:
            print(f"Error retrieving data for user with id {user_id}: {e}")
            continue

        user_cluster_id = user_data.get("cluster")
        if user_cluster_id is None:
            print(f"User with id {user_id} does not have a cluster ID.")
            continue

        # Preprocess user data to get vector representation
        user_vector = preprocess_data(pd.DataFrame([user_data]), column_weights).iloc[0].tolist()

        # Query for all users in the same cluster
        cluster_query = {
            "size": 100,
            "query": {
                "term": {"cluster": user_cluster_id}
            }
        }
        cluster_response = client.search(index=user_index, body=cluster_query)
        cluster_users = cluster_response['hits']['hits']
        num_users_in_cluster = len(cluster_users)

        # Closeness calculation for each user in the same cluster
        for connected_user in cluster_users:
            connected_user_data = connected_user["_source"]
            connected_user_id = connected_user_data.get("id")

            # Skip if it's the same user
            if connected_user_id == user_id:
                continue

            # Preprocess connected user data to get feature vector
            connected_user_vector = preprocess_data(pd.DataFrame([connected_user_data]), column_weights).iloc[0].tolist()

            # Calculate closeness as cosine similarity
            closeness_score = np.dot(user_vector, connected_user_vector) / (
                    np.linalg.norm(user_vector) * np.linalg.norm(connected_user_vector))

            # Calculate additional weight based on shared values
            shared_values = [key for key in user_data.keys() if user_data[key] == connected_user_data.get(key) and key in column_weights]
            shared_value_score = sum(column_weights[key] for key in shared_values) / sum(column_weights.values())

            # Final weighted closeness score
            final_closeness = 0.7 * closeness_score + 0.3 * shared_value_score

            # Filter by minimum closeness
            if final_closeness >= request.min_closeness:
                results.append({
                    "user_id": connected_user_id,
                    "num_users_in_cluster": num_users_in_cluster,
                    "closeness": round(final_closeness * 100, 2),
                    "user_name": connected_user_data.get("user_name", "N/A"),
                    "shared_values": shared_values,
                    "num_shared_values": len(shared_values),
                    "earliest_date_of_sharing": connected_user_data.get("share_date", datetime.now().isoformat())
                })

    # Sort all results by closeness score in descending order
    results = sorted(results, key=lambda x: x['closeness'], reverse=True)

    return {"connected_users": results}

# Export the router for integration into the main FastAPI app
multi_user_analysis_router = router
