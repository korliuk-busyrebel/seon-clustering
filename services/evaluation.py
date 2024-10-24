from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def evaluate_clustering(df, clusters):
    silhouette_avg = silhouette_score(df, clusters)
    ch_score = calinski_harabasz_score(df, clusters)
    db_score = davies_bouldin_score(df, clusters)

    return silhouette_avg, ch_score, db_score
