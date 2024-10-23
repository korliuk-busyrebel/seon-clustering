import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json

# Load column weights from an external JSON file
def load_column_weights(weights_file_path):
    with open(weights_file_path, 'r') as f:
        return json.load(f)

# Apply column weights during preprocessing
def preprocess_data(df, column_weights):
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Fill missing values for numeric columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Encode categorical columns
    for col in categorical_cols:
        df[col] = df[col].fillna('missing')  # Replace NaN with 'missing'
        le = LabelEncoder()
        try:
            df[col] = le.fit_transform(df[col].astype(str))
        except Exception as e:
            print(f"Error encoding column {col}: {e}")

    # Combine the numeric and encoded categorical columns into a single DataFrame
    all_columns = list(numeric_cols) + list(categorical_cols)
    df_processed = df[all_columns]

    # Scale the entire dataset
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_processed)

    # Apply column weights
    for idx, col in enumerate(all_columns):
        if col in column_weights and column_weights[col] > 0.0:
            df_scaled[:, idx] *= column_weights[col]

    return pd.DataFrame(df_scaled, columns=all_columns)
