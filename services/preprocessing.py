from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

def preprocess_data(df, column_weights):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    df[numeric_cols] = df[numeric_cols].fillna(0)

    for col in categorical_cols:
        df[col] = df[col].fillna('missing')
        le = LabelEncoder()
        try:
            df[col] = le.fit_transform(df[col].astype(str))
        except Exception as e:
            print(f"Error encoding column {col}: {e}")

    df_processed = df[numeric_cols.tolist() + categorical_cols.tolist()]
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_processed)

    for idx, col in enumerate(df_processed.columns):
        if col in column_weights and column_weights[col] > 0:
            df_scaled[:, idx] *= column_weights[col]

    return pd.DataFrame(df_scaled, columns=df_processed.columns)
