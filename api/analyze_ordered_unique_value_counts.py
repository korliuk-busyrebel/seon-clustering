from fastapi import APIRouter, File, UploadFile, BackgroundTasks
import pandas as pd
from io import StringIO
from utils.opensearch_client import client, OS_INDEX
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a router
router = APIRouter()


def extract_ordered_unique_value_counts(df: pd.DataFrame) -> dict:
    """
    Extracts unique values with their counts and order for each column in the DataFrame and returns as a dictionary.
    """
    unique_values_ordered = {}
    for column in df.columns:
        # Get unique values and their counts in descending order
        value_counts = df[column].value_counts(dropna=True).reset_index()
        value_counts.columns = ['value', 'count']  # Rename columns for clarity

        # Add order and structure data for each unique value
        ordered_values = [
            {"order": idx + 1, "value": row['value'], "count": row['count']}
            for idx, row in value_counts.iterrows()
        ]

        # Store the ordered list for the column
        unique_values_ordered[column] = ordered_values
    return unique_values_ordered


def index_ordered_unique_value_counts(ordered_value_counts: dict, index_name: str):
    """
    Inserts a dictionary of ordered unique value counts into the specified OpenSearch index.
    """
    try:
        client.index(index=index_name, body=ordered_value_counts)
        logger.info(f"Ordered unique value counts successfully indexed in '{index_name}'.")
    except Exception as e:
        logger.error(f"Error indexing ordered unique value counts in '{index_name}': {e}")


@router.post("/analyze-ordered-unique-value-counts/")
async def analyze_ordered_unique_value_counts(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Read file contents
    contents = await file.read()

    # Load CSV data into DataFrame
    df = pd.read_csv(StringIO(contents.decode('utf-8')))
    logger.info("CSV file successfully loaded for ordered unique value counts extraction.")

    # Extract ordered unique values and their counts for each column
    ordered_value_counts = extract_ordered_unique_value_counts(df)
    logger.info("Ordered unique value counts extracted from CSV.")

    # Define OpenSearch index for ordered unique value counts
    index_name = "ordered-unique-value-counts-index"

    # Ensure the index exists or create it
    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name, body={"settings": {"index": {"knn": False}}})
        logger.info(f"Created OpenSearch index '{index_name}' for ordered unique value counts.")

    # Add indexing task to background
    background_tasks.add_task(index_ordered_unique_value_counts, ordered_value_counts, index_name)

    return {"message": "Ordered unique value counts extraction and indexing started in background."}


# Export the router for use in the main app
analyze_ordered_unique_value_counts = router
