import pandas as pd
from monitoring_logging.logging_setup import setup_logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

logger = setup_logging()

def read_data_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        logger.info(f"CSV file {file_path} read successfully.")
        return df
    except Exception as e:
        logger.error(f"Error reading CSV file {file_path}: {e}")
        return None
    
def preprocess_data(data):
    try:
        data = pd.get_dummies(data, columns=['Geography', 'Gender'], dtype=int)

        scaler = MinMaxScaler()
        data[['CreditScore',
              'Age',
              'Tenure',
              'Balance',
              'NumOfProducts',
              'EstimatedSalary']] = scaler.fit_transform(data[['CreditScore',
                                                                'Age',
                                                                'Tenure',
                                                                'Balance',
                                                                'NumOfProducts',
                                                                'EstimatedSalary']])
        
        data.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)
        logger.info("Data preprocessing completed successfully.")
        return data
    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        return None
    
def split_data(df, target_column):
    try:
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info("Data split into features and target successfully.")
        return  X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        return None, None
    
    
    
    