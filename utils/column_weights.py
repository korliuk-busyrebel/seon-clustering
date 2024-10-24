import json

def load_column_weights(weights_file_path):
    with open(weights_file_path, 'r') as f:
        return json.load(f)
