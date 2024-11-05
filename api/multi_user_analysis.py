from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from utils.qdrant_client import get_qdrant_client
from utils.column_weights import load_column_weights
from datetime import datetime
import numpy as np

router = APIRouter()
qdrant_client = get_qdrant_client()

# Load column weights for closeness calculations
column_weights = load_column_weights('/app/utils/column_weights.json')
column_names = list(column_weights.keys())  # Get the list of feature names in the correct order

class MultiUserAnalysisRequest(BaseModel):
    user_ids: List[str]
    min_closeness: float = 0.5  # Minimum closeness threshold
    k: int = 10  # Default number of nearest neighbors
    collection_name: str  # Qdrant collection name

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

@router.post("/qdrant/multi-user-analysis/")
async def multi_user_analysis(request: MultiUserAnalysisRequest):
    results = {}

    try:
        for user_id in request.user_ids:
            # Retrieve user vector from Qdrant
            query_vector = qdrant_client.get_vector(
                collection_name=request.collection_name, vector_id=user_id
            )
            if not query_vector:
                print(f"User with id {user_id} not found in collection {request.collection_name}.")
                continue

            # Perform similarity search in Qdrant
            response = qdrant_client.search(
                collection_name=request.collection_name,
                query_vector=query_vector,
                limit=request.k,
            )

            closest_users = []
            for hit in response:
                connected_user_data = hit["payload"]
                connected_user_id = connected_user_data.get("id")

                # Exclude the requested user_id from results
                if connected_user_id == user_id:
                    continue

                # Retrieve similarity score
                similarity_score = hit["score"]

                # Retrieve and validate connected user vector
                connected_user_vector = connected_user_data.get("vector", [])
                if not isinstance(connected_user_vector, list) or len(connected_user_vector) != len(query_vector):
                    print(f"Skipping connected_user_id {connected_user_id} due to invalid vector.")
                    continue

                # Calculate shared values
                shared_values = get_shared_values(query_vector, connected_user_vector, column_names, column_weights)
                num_shared_values = len(shared_values)

                # Calculate final closeness score
                shared_value_score = sum(column_weights.get(key, 1) for key in shared_values) / sum(column_weights.values())
                final_closeness = 0.7 * similarity_score + 0.3 * shared_value_score
                final_closeness = min(final_closeness * 100, 100)  # Ensure it's between 0-100

                # Apply minimum closeness filter
                if final_closeness >= request.min_closeness:
                    closest_users.append({
                        "user_id": connected_user_id,
                        "similarity": round(similarity_score, 2),
                        "closeness": round(final_closeness, 2),
                        "user_name": connected_user_data.get("user_name", "N/A"),
                        "shared_values": shared_values,
                        "num_shared_values": num_shared_values,
                        "earliest_date_of_sharing": connected_user_data.get("share_date", datetime.now().isoformat())
                    })

            results[user_id] = sorted(closest_users, key=lambda x: -x["closeness"])

        return {"connected_users_per_user": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to perform multi-user analysis: {e}")

qdrant_multi_user_analysis = router
