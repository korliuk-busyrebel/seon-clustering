import os

# Load OpenSearch Configuration
OS_HOST = os.getenv("OS_HOST", "localhost")
OS_PORT = int(os.getenv("OS_PORT", 9200))
OS_INDEX = os.getenv("OS_INDEX", "clustered_data")
REDUCED_INDEX = os.getenv("OS_REDUCED_INDEX", "clustered_data_visual")
OS_SCHEME = os.getenv("OS_SCHEME", "http")
OS_USERNAME = os.getenv("OS_USERNAME", "admin")
OS_PASSWORD = os.getenv("OS_PASSWORD", "admin")

# Load MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-tracking")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "your-username")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "your-password")

# Set Git executable if needed
os.environ["GIT_PYTHON_GIT_EXECUTABLE"] = os.getenv("GIT_PYTHON_GIT_EXECUTABLE", "/usr/bin/git")
os.environ["GIT_PYTHON_REFRESH"] = os.getenv("GIT_PYTHON_REFRESH", "quiet")

# Environment variables for MLflow authentication
os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
