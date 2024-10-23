from fastapi import FastAPI, File, UploadFile
import pandas as pd
from io import StringIO
from config import client, OS_INDEX, REDUCED_INDEX
from preprocessing import load_column_weights, preprocess_data
from clustering import find_optimal_dbscan, assign_noise_points, reduce_dimensions_optimal, evaluate_clustering
from mlflow_utils import start_mlflow_run, log_metrics_and_model, end_mlflow_run
from sklearn.cluster import DBSCAN

app = FastAPI()


@app.post("/create-clusters/")
async def create_clusters(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))
    column_weights = load_column_weights('/app/column_weights.json')
    df_preprocessed = preprocess_data(df, column_weights)

    optimal_eps, optimal_min_samples = find_optimal_dbscan(df_preprocessed)

    run = start_mlflow_run("clustering_experiment", {"eps": optimal_eps, "min_samples": optimal_min_samples})

    clustering_model = DBSCAN(eps=optimal_eps, min_samples=optimal_min_samples)
    clusters = clustering_model.fit_predict(df_preprocessed)

    clusters = assign_noise_points(df_preprocessed, clusters)
    df['cluster'] = clusters

    input_example = df_preprocessed.head(1)
    df_reduced = reduce_dimensions_optimal(df_preprocessed)

    for index, row in df.iterrows():
        client.index(index=OS_INDEX, id=index, body=row.to_dict())

    for index, row in df_reduced.iterrows():
        row['cluster'] = df['cluster'].iloc[index]
        client.index(index=REDUCED_INDEX, id=index, body=row.to_dict())

    silhouette_avg, ch_score, db_score = evaluate_clustering(df_preprocessed, clusters)

    log_metrics_and_model(clustering_model, "dbscan_model", input_example, {
        "silhouette_score": silhouette_avg,
        "calinski_harabasz_score": ch_score,
        "davies_bouldin_score": db_score
    })

    end_mlflow_run()

    return {
        "message": "Clusters created.",
        "silhouette_score": silhouette_avg,
        "calinski_harabasz_score": ch_score,
        "davies_bouldin_score": db_score
    }
