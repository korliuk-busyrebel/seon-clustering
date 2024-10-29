from pydantic import BaseModel
from utils.opensearch_client import client, router, OS_KNN_INDEX
from utils.column_weights import load_column_weights
from datetime import datetime

# Load column weights for closeness calculations
column_weights = load_column_weights('/app/utils/column_weights.json')


# Define request model
class MultiUserRequest(BaseModel):
    user_ids: list
    min_closeness: float = 0.5  # Minimum closeness threshold


@router.post("/multi-user-analysis/")
async def multi_user_analysis(request: MultiUserRequest):
    results = []

    for user_id in request.user_ids:
        # Search for each user in OpenSearch
        query = {"size": 10, "query": {"match": {"user_id": user_id}}}
        user_results = client.search(index=OS_KNN_INDEX, body=query)

        for hit in user_results['hits']['hits']:
            source = hit['_source']
            connected_user_id = source['user_id']
            connection_data = {key: source[key] for key in column_weights if key in source}

            # Calculate closeness based on shared values and weights
            shared_values = [key for key, value in connection_data.items() if value]
            closeness_score = sum(column_weights[key] for key in shared_values) / sum(column_weights.values())

            if closeness_score >= request.min_closeness:
                results.append({
                    "user_id": connected_user_id,
                    "num_users_in_cluster": len(user_results['hits']['hits']),
                    "closeness": closeness_score,
                    "user_name": source.get("user_name"),
                    "shared_values": shared_values,
                    "num_shared_values": len(shared_values),
                    "earliest_date_of_sharing": source.get("share_date", datetime.now().isoformat())
                })

    # Sort results by closeness score
    results = sorted(results, key=lambda x: x['closeness'], reverse=True)

    return {"connected_users": results}


# Export the router for integration into the main FastAPI app
multi_user_analysis_router = router
