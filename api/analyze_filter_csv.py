from fastapi import APIRouter, File, UploadFile, BackgroundTasks
import pandas as pd
from io import StringIO
from utils.opensearch_client import client, OS_INDEX
from utils.column_weights import load_column_weights
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a router
router = APIRouter()


def analyze_and_filter_csv(file_contents: bytes, weights_path: str, index_name: str):
    """
    Analyzes and filters CSV data based on column weights, retaining only columns with non-zero weights.
    Inserts filtered records into OpenSearch.
    """
    # Load CSV data into a DataFrame
    df = pd.read_csv(StringIO(file_contents.decode('utf-8')))
    logger.info("CSV file successfully loaded.")

    # Load column weights
    column_weights = load_column_weights(weights_path)

    # Filter columns based on weights
    filtered_columns = [col for col, weight in column_weights.items() if weight > 0]
    df_filtered = df[filtered_columns]
    logger.info(f"Filtered columns based on weights. Retained columns: {filtered_columns}")

    # Insert filtered data into OpenSearch in batches
    batch_size = 500
    total_records = len(df_filtered)
    skipped_records = 0

    for start in range(0, total_records, batch_size):
        batch = df_filtered.iloc[start:start + batch_size].to_dict(orient="records")

        for idx, record in enumerate(batch):
            success = False
            for attempt in range(3):  # Retry up to 3 times
                try:
                    client.index(index=index_name, id=start + idx, body=record)
                    success = True
                    break
                except Exception as e:
                    logger.error(f"Error indexing document {start + idx} in {index_name}: {e}")
                    time.sleep(1)  # Wait before retrying
            if not success:
                skipped_records += 1
                logger.warning(f"Document {start + idx} in {index_name} skipped after 3 failed attempts.")

    logger.info(f"Finished indexing filtered data. Total records: {total_records}, Skipped records: {skipped_records}")


@router.post("/analyze-filter-csv/")
async def analyze_filter_csv(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Read file contents
    contents = await file.read()

    # Define the path to the column weights file
    weights_path = '/app/utils/column_weights.json'
    index_name = OS_INDEX

    # Add the filtering and indexing task to the background
    background_tasks.add_task(analyze_and_filter_csv, contents, weights_path, index_name)

    return {"message": "CSV analysis and filtering started in background."}


# Export the router for use in the main app
analyze_filter_csv = router
