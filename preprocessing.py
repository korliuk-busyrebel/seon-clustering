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

    # Apply StandardScaler to numeric columns
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[numeric_cols])

    # Convert the scaled array back to a DataFrame
    df_scaled = pd.DataFrame(df_scaled, columns=numeric_cols)

    # Apply column weights, but only for those with weight greater than 0
    for idx, col in enumerate(numeric_cols):
        if col in column_weights and column_weights[col] > 0.0:
            df_scaled[col] *= column_weights[col]

    # Convert boolean-like columns to proper boolean type
    df_scaled = convert_to_boolean(df_scaled)

    return df_scaled


def convert_to_boolean(df):
    """Convert columns that are supposed to be boolean into proper booleans."""

    # Loop through each column in the DataFrame
    for col in df.columns:
        # Check if the column contains only 0 and 1, or True and False
        if df[col].dropna().isin([0, 1]).all():  # Check for 0 and 1 values
            # Convert column to boolean
            df[col] = df[col].astype(bool)
        elif df[col].dropna().isin([True, False]).all():  # Check for True/False values
            df[col] = df[col].astype(bool)

    return df
