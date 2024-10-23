import json
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_column_weights(weights_file_path):
    with open(weights_file_path, 'r') as f:
        return json.load(f)


def preprocess_data(df, column_weights):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include(['object'])).columns

    df[numeric_cols] = df[numeric_cols].fillna(0)

    for col in categorical_cols:
        df[col] = df[col].fillna('missing')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    all_columns = list(numeric_cols) + list(categorical_cols)
    df_processed = df[all_columns]

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_processed)

    for idx, col in enumerate(all_columns):
        if col in column_weights and column_weights[col] > 0.0:
            df_scaled[:, idx] *= column_weights[col]

    return pd.DataFrame(df_scaled, columns=all_columns)
