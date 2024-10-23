from opensearchpy import OpenSearch
from config import OS_HOST, OS_PORT, OS_SCHEME, OS_USERNAME, OS_PASSWORD

# Initialize OpenSearch client with basic authentication
client = OpenSearch(
    hosts=[{'host': OS_HOST, 'port': OS_PORT}],
    http_auth=(OS_USERNAME, OS_PASSWORD),
    use_ssl=(OS_SCHEME == 'https'),
    verify_certs=False,
    scheme=OS_SCHEME,
    timeout=60
)

def save_to_opensearch(df, df_reduced):
    for index, row in df.iterrows():
        doc = row.to_dict()
        client.index(index='clustered_data', id=index, body=doc)

    for index, row in df_reduced.iterrows():
        doc = row.to_dict()
        doc['cluster'] = df['cluster'].iloc[index]
        client.index(index='clustered_data_visual', id=index, body=doc)

def classify_record_knn(record):
    # Load column weights
    column_weights = load_column_weights('/app/column_weights.json')

    # Convert the incoming record to a DataFrame for preprocessing
    record_data = pd.DataFrame([record])

    # Preprocess the record
    record_preprocessed = preprocess_data(record_data, column_weights)

    # Convert the preprocessed record into a list (vector)
    vector = record_preprocessed.values[0].tolist()

    # Define the k-NN query in OpenSearch
    k_nn_query = {
        "size": 1,  # Retrieve the closest neighbor
        "_source": ["cluster"],  # Specify that we want to retrieve only the cluster field
        "knn": {
            "field": "vector",  # The vector field where the document vector is stored
            "query_vector": vector,  # Vector for comparison
            "k": 1,  # Number of nearest neighbors to retrieve
            "num_candidates": 100  # Optional: the number of top candidates to consider
        }
    }

    # Perform the k-NN search in OpenSearch
    response = client.search(index='clustered_data', body=k_nn_query)

    # Retrieve the closest neighbor's cluster
    if response['hits']['hits']:
        predicted_cluster = response['hits']['hits'][0]['_source']['cluster']
    else:
        predicted_cluster = None  # Handle case where no neighbor is found

    return {"cluster": predicted_cluster}
