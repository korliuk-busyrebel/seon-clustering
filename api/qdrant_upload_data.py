from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
from utils.qdrant_client import get_qdrant_client
import pandas as pd
from io import StringIO
from utils.column_weights import load_column_weights

router = APIRouter()
qdrant_client = get_qdrant_client()

class UploadDataRequest(BaseModel):
    collection_name: str

@router.post("/qdrant/upload-data/")
async def upload_data(request: UploadDataRequest, file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode("utf-8")))

    column_weights = load_column_weights("/app/utils/column_weights.json")
    vectors = [
        {
            "id": str(row["id"]),
            "vector": [row[col] * column_weights.get(col, 1) for col in df.columns if column_weights.get(col, 0) > 0],
            "payload": row.to_dict(),
        }
        for _, row in df.iterrows()
    ]

    vector_size = len(vectors[0]["vector"])
    try:
        qdrant_client.upload_collection(
            collection_name=request.collection_name,
            vectors=vectors,
            vector_size=vector_size,
            payload=1
        )
        return {"message": f"Data uploaded to Qdrant collection '{request.collection_name}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload data: {e}")

qdrant_upload_data = router
