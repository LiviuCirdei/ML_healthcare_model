import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data(filepath):
    """Load data from a CSV file."""
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)

def preprocess_data(df, is_train=True, age_map=None):
    """
    Preprocess the data by mapping age ranges and encoding the target column.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        is_train (bool): Whether the data is for training.
        age_map (dict): Mapping for age ranges to numerical values.
    
    Returns:
        pd.DataFrame or tuple: Preprocessed dataframe or (X, y) for training.
    """
    if age_map is None:
        age_map = {
            '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
            '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
            '[80-90)': 85, '[90-100)': 95
        }
    
    if 'age' in df.columns:
        df['age'] = df['age'].map(age_map)
    else:
        logging.warning("Column 'age' not found in the dataframe.")
    
    if is_train and 'readmitted' in df.columns:
        df['readmitted'] = df['readmitted'].map({'no': 0, 'yes': 1})
        if df['readmitted'].isnull().any():
            logging.warning("Unexpected values found in 'readmitted' column.")
        X = df.drop(columns=['readmitted'])
        y = df['readmitted']
        return X, y
    return df

def split_data(df, test_size=0.2, random_state=42):
    """Split the dataframe into training and testing sets."""
    return train_test_split(df, test_size=test_size, random_state=random_state)

def save_data(df, filepath):
    """Save a dataframe to a CSV file."""
    try:
        df.to_csv(filepath, index=False)
        logging.info(f"✅ Data saved to {filepath}")
    except Exception as e:
        logging.error(f"Failed to save data to {filepath}: {e}")
        raise

def split_and_save(input_path, train_path, test_path, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets and save them to CSV files.
    
    Args:
        input_path (str): Path to the input CSV file.
        train_path (str): Path to save the training data.
        test_path (str): Path to save the testing data.
        test_size (float): Proportion of the data to include in the test split.
        random_state (int): Random seed for reproducibility.
    """
    df = load_data(input_path)
    train_df, test_df = split_data(df, test_size=test_size, random_state=random_state)
    save_data(train_df, train_path)
    save_data(test_df, test_path)

if __name__ == "__main__":
    split_and_save(
        input_path="data/hospital_readmissions.csv",
        train_path="data/train_hospital_readmissions.csv",
        test_path="data/test_hospital_readmissions.csv"
    )
