from src.train import train_model
from src.inference import test_model
from data.data_handeler import read_data_csv, preprocess_data, split_data
from monitoring_logging.monitoring import set_experiment
from mlflow import start_run, end_run
from models.logistic_regression import save_logistic_regression_model
from models.svm import save_svm_model
from monitoring_logging.logging_setup import setup_logging
exp = set_experiment()
import pandas as pd

def main():
    logger = setup_logging()
    logger.info("Starting the main function...")
    # Read the data
    df = read_data_csv("data/Churn_Modelling.csv")
    df = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = split_data(df, "Exited")
    logger.info(X_train.iloc[0])
    
    
    
    start_run(experiment_id=exp.experiment_id, run_name="Logistic Regression Model")
    
    log_model = train_model("logistic_regression", X_train, y_train, 
                        penalty="l1", C=0.9, solver="liblinear")
    accuracy, mse = test_model("logistic_regression", log_model, X_test, y_test)
    end_run()
    # Save the model
    save_logistic_regression_model(log_model, "app/logistic_regression_model.pkl")
    logger.debug(f"logistic regression model saved")
    
        
    start_run(experiment_id=exp.experiment_id, run_name="svm Model")
    svm_model = train_model("svm", X_train, y_train,
                            kernel="rbf", C=15.0, gamma="auto")
    
    accuracy, mse = test_model("svm", svm_model, X_test, y_test)
    end_run()
    # Save the model
    save_svm_model(svm_model, "app/svm_model.pkl")
    logger.debug(f"svm model saved")
    
if __name__ == "__main__":
    main()