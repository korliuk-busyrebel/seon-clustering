from pydantic import BaseModel
from utils.opensearch_client import client, router, OS_KNN_INDEX
from utils.column_weights import load_column_weights
from services.preprocessing import preprocess_data
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

        # Retrieve user data by `id` field (not `_id`)
        try:
            user_data_query = {"query": {"term": {"id": user_id}}}
            user_search = client.search(index=user_index, body=user_data_query)

            if not user_search['hits']['hits']:
                print(f"User with id {user_id} not found in index {user_index}.")
                continue

            user_data = user_search['hits']['hits'][0]["_source"]
            user_vector = preprocess_data(pd.DataFrame([user_data]), column_weights).iloc[0].tolist()

            # Ensure the vector is exactly 898 dimensions
            if len(user_vector) < 898:
                user_vector.extend([0] * (898 - len(user_vector)))
            elif len(user_vector) > 898:
                user_vector = user_vector[:898]

            print(f"User vector for ID {user_id}: {user_vector[:10]}... (truncated)")

        except Exception as e:
            print(f"Error retrieving data for user with id {user_id}: {e}")
            continue

        user_cluster_id = user_data.get("cluster")
        if user_cluster_id is None:
            print(f"User with id {user_id} does not have a cluster ID.")
            continue

        # Retrieve all users in the same cluster
        cluster_query = {
            "size": 100,
            "query": {
                "term": {"cluster": user_cluster_id}
            }
        }
        cluster_response = client.search(index=user_index, body=cluster_query)
        cluster_users = cluster_response['hits']['hits']
        num_users_in_cluster = len(cluster_users)
        print(f"Number of users in cluster {user_cluster_id}: {num_users_in_cluster}")

        # Closeness calculation for each user in the cluster
        for connected_user in cluster_users:
            connected_user_data = connected_user["_source"]
            connected_user_id = connected_user_data.get("id")

            if connected_user_id == user_id:
                continue

            # Preprocess connected user data to get feature vector
            connected_user_vector = preprocess_data(pd.DataFrame([connected_user_data]), column_weights).iloc[
                0].tolist()

            if len(connected_user_vector) < 898:
                connected_user_vector.extend([0] * (898 - len(connected_user_vector)))
            elif len(connected_user_vector) > 898:
                connected_user_vector = connected_user_vector[:898]

            # Calculate closeness as cosine similarity
            closeness_score = np.dot(user_vector, connected_user_vector) / (
                    np.linalg.norm(user_vector) * np.linalg.norm(connected_user_vector))

            shared_values = [
                key for key in user_data.keys()
                if user_data[key] == connected_user_data.get(key) and key in column_weights
            ]
            shared_value_score = sum(column_weights[key] for key in shared_values) / sum(column_weights.values())

            final_closeness = 0.7 * closeness_score + 0.3 * shared_value_score

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
                print(
                    f"Connected User ID: {connected_user_id}, Closeness: {final_closeness}, Shared Values: {shared_values}")

    # Sort all results by closeness score in descending order
    results = sorted(results, key=lambda x: x['closeness'], reverse=True)
    print(f"Final connected users list: {results}")

    return {"connected_users": results}


# Export the router for integration into the main FastAPI app
multi_user_analysis = router
