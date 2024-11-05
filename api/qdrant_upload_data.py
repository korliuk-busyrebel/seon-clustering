from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import Optional
from utils.qdrant_client import get_qdrant_client
from utils.column_weights import load_column_weights
from io import StringIO
import pandas as pd
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
qdrant_client = get_qdrant_client()

def prepare_vectors(df: pd.DataFrame, column_weights: dict):
    """Preprocesses data based on column weights and prepares vectors for Qdrant."""
    logger.info("Starting data preprocessing...")
    df.fillna(0, inplace=True)  # Fill NaNs with 0s
    vectors = []

    for i, row in df.iterrows():
        vector = [row[col] * column_weights.get(col, 1) for col in df.columns if column_weights.get(col, 0) > 0]
        vectors.append({
            "id": str(row["id"]),
            "vector": vector,
            "payload": row.to_dict()
        })

        if (i + 1) % 500 == 0:  # Log every 500 rows processed
            logger.info(f"Processed {i + 1} rows for vector preparation.")

    logger.info("Data preprocessing completed.")
    return vectors

@router.post("/qdrant/upload-data/")
async def upload_data(collection_name: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode("utf-8")))

    # Load column weights
    column_weights = load_column_weights("/app/utils/column_weights.json")

    # Prepare vectors
    vectors = prepare_vectors(df, column_weights)
    vector_size = len(vectors[0]["vector"])

    start_time = time.time()
    uploaded_count = 0
    skipped_docs = 0

    # Upload each vector individually to Qdrant
    try:
        for i, vector in enumerate(vectors):
            try:
                qdrant_client.upload_collection(
                    collection_name=collection_name,
                    vectors=[vector["vector"]],
                    vector_size=vector_size,
                    payload=[vector["payload"]],
                )
                uploaded_count += 1
            except Exception as e:
                logger.error(f"Failed to upload document {vector['id']} to Qdrant: {e}")
                skipped_docs += 1

            # Log progress every 500 records
            if (i + 1) % 500 == 0:
                elapsed_time = time.time() - start_time
                percent_complete = (i + 1) / len(vectors) * 100
                logger.info(f"Uploaded {i + 1} / {len(vectors)} ({percent_complete:.2f}%) - Elapsed time: {elapsed_time:.2f} seconds.")

        elapsed_time = time.time() - start_time
        logger.info(f"Upload to Qdrant completed. Total uploaded: {uploaded_count}, Skipped documents: {skipped_docs}, Time taken: {elapsed_time:.2f} seconds.")
        return {"message": f"Data uploaded to Qdrant collection '{collection_name}', skipped documents: {skipped_docs}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload data: {e}")

# Export the router for use in the main app
qdrant_upload_data = router
