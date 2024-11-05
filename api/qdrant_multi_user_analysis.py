from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from utils.qdrant_client import get_qdrant_client

router = APIRouter()
qdrant_client = get_qdrant_client()

class MultiUserAnalysisRequest(BaseModel):
    user_ids: List[str]
    k: int = 10
    collection_name: str

@router.post("/qdrant/multi-user-analysis/")
async def multi_user_analysis(request: MultiUserAnalysisRequest):
    analysis_results = {}

    try:
        for user_id in request.user_ids:
            query_vector = qdrant_client.get_vector(
                collection_name=request.collection_name, vector_id=user_id
            )
            response = qdrant_client.search(
                collection_name=request.collection_name,
                query_vector=query_vector,
                limit=request.k,
            )

            analysis_results[user_id] = [
                {
                    "user_id": hit["payload"]["id"],
                    "score": hit["score"],
                    "details": hit["payload"]
                }
                for hit in response
            ]

        return {"analysis": analysis_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to perform multi-user analysis: {e}")

qdrant_multi_user_analysis = router
