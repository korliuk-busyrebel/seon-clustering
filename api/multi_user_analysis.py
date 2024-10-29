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
    user_index = request.index

    for user_id in request.user_ids:
        try:
            user_data_query = {"query": {"term": {"id": user_id}}}
            user_search = client.search(index=user_index, body=user_data_query)

            if not user_search['hits']['hits']:
                raise HTTPException(status_code=404, detail=f"User {user_id} not found in index {user_index}.")

            user_data = user_search['hits']['hits'][0]["_source"]
            user_vector = preprocess_data(pd.DataFrame([user_data]), column_weights).iloc[0].tolist()

            if len(user_vector) < 898:
                user_vector.extend([0] * (898 - len(user_vector)))
            elif len(user_vector) > 898:
                user_vector = user_vector[:898]

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

            response = client.search(index=user_index, body=knn_query)
            for hit in response['hits']['hits']:
                connected_user_data = hit["_source"]
                shared_values, num_shared_values = extract_shared_values(user_data, connected_user_data)
                closeness = calculate_closeness(shared_values, column_weights)

                if closeness >= request.min_closeness:
                    results.append({
                        "user_id": connected_user_data.get("id"),
                        "closeness": round(closeness * 100, 2),
                        "user_name": connected_user_data.get("user_name", "N/A"),
                        "shared_values": shared_values,
                        "num_shared_values": num_shared_values,
                        "earliest_date_of_sharing": connected_user_data.get("share_date")
                    })

        except Exception as e:
            print(f"Error retrieving data for user with id {user_id}: {e}")

    results = sorted(results, key=lambda x: -x['closeness'])
    return {"connected_users": results}

# Export the router for integration into the main FastAPI app
multi_user_analysis = router
