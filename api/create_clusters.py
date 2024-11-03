from fastapi import APIRouter, File, UploadFile, BackgroundTasks
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN  # for DBSCAN clustering algorithm
from io import StringIO
from services.clustering import find_optimal_dbscan, assign_noise_points
from services.dimensionality_reduction import reduce_dimensions_optimal
from services.evaluation import evaluate_clustering
from services.preprocessing import preprocess_data
from utils.opensearch_client import client, router, OS_KNN_INDEX, REDUCED_INDEX, OS_INDEX, OS_RAW_INDEX
from utils.column_weights import load_column_weights
import mlflow

# Define a router
router = APIRouter()

# Function to handle the clustering process in the background
def process_clusters(df: pd.DataFrame):
    # Load column weights and preprocess the data
    column_weights = load_column_weights('/app/utils/column_weights.json')
    df_preprocessed = preprocess_data(df, column_weights)

    # Dynamically find optimal eps and min_samples
    optimal_eps, optimal_min_samples = find_optimal_dbscan(df_preprocessed)

    # Start an MLflow run
    with mlflow.start_run() as run:
        mlflow.log_param("eps", optimal_eps)
        mlflow.log_param("min_samples", optimal_min_samples)

        # Apply DBSCAN clustering
        clustering_model = DBSCAN(eps=optimal_eps, min_samples=optimal_min_samples)
        clusters = clustering_model.fit_predict(df_preprocessed)

        # Assign noise points (-1) to nearest cluster
        clusters = assign_noise_points(df_preprocessed, clusters)

        # Add clusters to the original dataframe
        df['cluster'] = clusters

        # Save to OpenSearch in batches
        batch_size = 1000

        # Store raw data
        for i in range(0, len(df), batch_size):
            batch = df[i:i + batch_size].to_dict(orient="records")
            for index, doc in enumerate(batch):
                client.index(index=OS_RAW_INDEX, id=i + index, body=doc)

        # Process reduced dimensions and store them in OpenSearch
        df_reduced = reduce_dimensions_optimal(df_preprocessed)
        df_reduced = pd.DataFrame(df_reduced, columns=[f"dim_{i + 1}" for i in range(df_reduced.shape[1])])

        # Store reduced dimension data
        for i in range(0, len(df_reduced), batch_size):
            batch = df_reduced[i:i + batch_size]
            batch_dicts = [
                {
                    **row.to_dict(),
                    "cluster": df['cluster'].iloc[i + index]
                }
                for index, row in batch.iterrows()
            ]
            for index, doc in enumerate(batch_dicts):
                client.index(index=REDUCED_INDEX, id=i + index, body=doc)

        # Prepare and store KNN vectors
        for i in range(0, len(df), batch_size):
            batch = df_preprocessed[i:i + batch_size]
            knn_docs = [
                {
                    "id": str(df['id'].iloc[i + idx]),
                    "vector": row.tolist(),
                    "cluster": df['cluster'].iloc[i + idx]
                }
                for idx, row in batch.iterrows()
            ]
            for index, doc in enumerate(knn_docs):
                client.index(index=OS_KNN_INDEX, id=i + index, body=doc)

        # Calculate and log evaluation metrics
        silhouette_avg, ch_score, db_score = evaluate_clustering(df_preprocessed, clusters)
        mlflow.log_metric("silhouette_score", silhouette_avg)
        mlflow.log_metric("calinski_harabasz_score", ch_score)
        mlflow.log_metric("davies_bouldin_score", db_score)
        mlflow.sklearn.log_model(clustering_model, "dbscan_model")


@router.post("/create-clusters/")
async def create_clusters(file: UploadFile = File(...), background_tasks: BackgroundTasks):
    contents = await file.read()

    # Load data in chunks to avoid memory issues with very large files
    df = pd.read_csv(StringIO(contents.decode('utf-8')), chunksize=50000)
    df = pd.concat(df)

    # Add the clustering task to the background
    background_tasks.add_task(process_clusters, df)

    return {"message": "Cluster creation started in background."}


# Export the router for use in the main app
create_clusters = router
