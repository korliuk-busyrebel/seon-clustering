from fastapi import APIRouter, File, UploadFile
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


@router.post("/create-clusters/")
async def create_clusters(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))

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

        # Save clusters to the dataframe
        df['cluster'] = clusters

        # Log model input example for signature
        input_example = df_preprocessed.head(1)
        mlflow.sklearn.log_model(clustering_model, "dbscan_model", input_example=input_example)

        # Calculate the number of dimensions based on the preprocessed data
        dimension = df_preprocessed.shape[1]

        # Create an OpenSearch index for clusters with dynamically calculated dimensions
        index_body = {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "vector": {"type": "knn_vector", "dimension": dimension},
                    "id": {"type": "keyword"},  # Document metadata fields
                }
            }
        }
        # Check if the index already exists, and if not, create it
        if not client.indices.exists(index=OS_KNN_INDEX):
            client.indices.create(index=OS_KNN_INDEX, body=index_body)

        # Optimal dimensionality reduction using PCA + UMAP
        df_reduced = reduce_dimensions_optimal(df_preprocessed)

        # Convert the NumPy array to a DataFrame for iterrows
        df_reduced = pd.DataFrame(df_reduced, columns=[f"dim_{i + 1}" for i in range(df_reduced.shape[1])])

        # Store original (raw) data in a validation index
        if not client.indices.exists(index=OS_RAW_INDEX):
            client.indices.create(index=OS_RAW_INDEX)

        for index, row in df.iterrows():
            raw_doc = row.to_dict()
            client.index(index=OS_RAW_INDEX, id=index, body=raw_doc)

        # Store original data with clusters in OpenSearch for processed data
        for index, row in df.iterrows():
            processed_doc = row.to_dict()
            client.index(index=OS_INDEX, id=index, body=processed_doc)

        # Store reduced-dimension data with cluster labels in OpenSearch
        for index, row in df_reduced.iterrows():
            reduced_doc = row.to_dict()
            reduced_doc['cluster'] = df['cluster'].iloc[index]
            client.index(index=REDUCED_INDEX, id=index, body=reduced_doc)

        # Prepare and add 'vector' and 'id' fields using preprocessed data
        for idx, row in df.iterrows():
            # Generate the vector for KNN search from preprocessed data
            unique_id = str(row['id'])
            vector = df_preprocessed.iloc[idx].values.tolist()  # Convert to list format
            cluster = row['cluster']
            # Create a minimal document with only 'id' and 'vector'
            knn_doc = {
                "id": unique_id,
                "vector": vector,
                "cluster": cluster
            }

            # Index the document with the minimal fields
            client.index(index=OS_KNN_INDEX, id=idx, body=knn_doc)

        return {"message": f"KNN index '{OS_KNN_INDEX}' prepared with {dimension} dimensions."}

        # Calculate evaluation metrics
        silhouette_avg, ch_score, db_score = evaluate_clustering(df_preprocessed, clusters)

        # Log metrics in MLflow
        mlflow.log_metric("silhouette_score", silhouette_avg)
        mlflow.log_metric("calinski_harabasz_score", ch_score)
        mlflow.log_metric("davies_bouldin_score", db_score)

        # Log the DBSCAN model in MLflow
        mlflow.sklearn.log_model(clustering_model, "dbscan_model")

    return {
        "message": f"Clusters created with final eps={optimal_eps}, min_samples={optimal_min_samples}, and stored in {OS_INDEX}. Reduced dimension data stored in {REDUCED_INDEX}. Raw data stored in {OS_RAW_INDEX}.",
        "silhouette_score": silhouette_avg,
        "calinski_harabasz_score": ch_score,
        "davies_bouldin_score": db_score
    }


# Export the router for use in the main app
create_clusters = router
