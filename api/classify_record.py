from fastapi import APIRouter
import pandas as pd
from services.preprocessing import preprocess_data
from services.opensearch_client import classify_record_opensearch
from utils.column_weights import load_column_weights

router = APIRouter()

@router.post("/classify-record/")
async def classify_record(record: dict):
    df = pd.DataFrame([record])
    column_weights = load_column_weights('./column_weights.json')
    df_preprocessed = preprocess_data(df, column_weights)

    cluster = classify_record_opensearch(df_preprocessed)
    return {"cluster": cluster}

# Export the router for use in the main app
classify_record = router
