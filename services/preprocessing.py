import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def preprocess_data(df, column_weights):
    """
    Preprocesses data by filling NaNs, encoding categorical columns, and scaling based on column weights.
    Logs detailed steps, including which columns are processed.
    """
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    logger.info(f"Found numeric columns: {list(numeric_cols)}")
    logger.info(f"Found categorical columns: {list(categorical_cols)}")

    # Fill NaN values in numeric columns with 0
    logger.info("Filling NaN values in numeric columns with 0...")
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Encode categorical columns and log each transformation
    for col in categorical_cols:
        df[col] = df[col].fillna('missing')
        le = LabelEncoder()
        try:
            df[col] = le.fit_transform(df[col].astype(str))
            logger.info(f"Encoded column '{col}' with LabelEncoder. Unique values: {le.classes_}")
        except Exception as e:
            logger.error(f"Error encoding column '{col}': {e}")

    # Combine processed numeric and categorical columns
    df_processed = df[numeric_cols.tolist() + categorical_cols.tolist()]
    logger.info("Scaling combined data...")

    # Scale combined data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_processed)

    # Apply column weights to scaled data and log the affected columns
    logger.info("Applying column weights...")
    weighted_columns = []
    for idx, col in enumerate(df_processed.columns):
        if col in column_weights and column_weights[col] > 0:
            df_scaled[:, idx] *= column_weights[col]
            weighted_columns.append(col)

    logger.info(f"Columns with applied weights: {weighted_columns}")

    return pd.DataFrame(df_scaled, columns=df_processed.columns)
