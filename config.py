import os
from opensearchpy import OpenSearch

# Get environment variables for OpenSearch connection and index
OS_HOST = os.getenv("OS_HOST", "localhost")
OS_PORT = os.getenv("OS_PORT", 9200)
OS_INDEX = os.getenv("OS_INDEX", "clustered_data")
REDUCED_INDEX = os.getenv("OS_REDUCED_INDEX", "clustered_data_visual")
OS_SCHEME = os.getenv("OS_SCHEME", "http")
OS_USERNAME = os.getenv("OS_USERNAME", "admin")
OS_PASSWORD = os.getenv("OS_PASSWORD", "admin")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-tracking")

# Set path to Git executable if needed
os.environ["GIT_PYTHON_GIT_EXECUTABLE"] = "/usr/bin/git"
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("MLFLOW_TRACKING_USERNAME", "your-username")
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_TRACKING_PASSWORD", "your-password")

# Initialize OpenSearch client
client = OpenSearch(
    hosts=[{'host': OS_HOST, 'port': int(OS_PORT)}],
    http_auth=(OS_USERNAME, OS_PASSWORD),
    use_ssl=(OS_SCHEME == 'https'),
    verify_certs=False,
    scheme=OS_SCHEME,
    timeout=60
)
