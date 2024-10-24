from opensearchpy import OpenSearch
import os

client = OpenSearch(
    hosts=[{'host': os.getenv("OS_HOST", "localhost"), 'port': int(os.getenv("OS_PORT", 9200))}],
    http_auth=(os.getenv("OS_USERNAME", "admin"), os.getenv("OS_PASSWORD", "admin")),
    use_ssl=os.getenv("OS_SCHEME", "http") == 'https',
    verify_certs=False,
    scheme=os.getenv("OS_SCHEME", "http"),
    timeout=60
)


def save_clusters(df, clusters):
    for index, row in df.iterrows():
        # Convert the row to a dictionary
        doc = row.to_dict()

        # Add the cluster information to the document
        doc['cluster'] = int(clusters[index])  # Ensure the cluster value is an integer

        # Index the document in OpenSearch
        client.index(index=os.getenv("OS_INDEX", "clustered_data"), id=index, body=doc)

def save_reduced_data(df_reduced, clusters):
    for index, row in df_reduced.iterrows():
        doc = row.to_dict()
        doc['cluster'] = clusters[index]
        client.index(index=os.getenv("OS_REDUCED_INDEX", "clustered_data_visual"), id=index, body=doc)

def classify_record_opensearch(df_preprocessed):
    # Convert the preprocessed record to a list (vector) for k-NN search
    vector = df_preprocessed.values[0].tolist()

    # Define the k-NN query
    k_nn_query = {
        "size": 1,  # Retrieve the closest neighbor
        "_source": False,  # Optionally, control which fields are returned
        "knn": {
            "field": "vector",  # Field where you stored your vector data
            "query_vector": vector,  # Vector for comparison
            "k": 1,  # Number of nearest neighbors to retrieve
            "num_candidates": 100  # Optional: number of top candidates to consider
        }
    }

    # Perform k-NN search in OpenSearch to find the nearest cluster
    response = client.search(index=os.getenv("OS_INDEX", "clustered_data"), body=k_nn_query)
    nearest_neighbor = response['hits']['hits'][0]

    # Retrieve the cluster information from the nearest neighbor
    predicted_cluster = nearest_neighbor['_id']  # Or any other relevant cluster field

    return predicted_cluster
