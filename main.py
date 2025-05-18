from src.train import train_model
from src.inference import test_model
from data.data_handeler import read_data_csv, preprocess_data, split_data
from monitoring_logging.monitoring import set_experiment
from mlflow import start_run, end_run
exp = set_experiment()
import pandas as pd

def main():
    # Read the data
    df = read_data_csv("data/Churn_Modelling.csv")
    df = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = split_data(df, "Exited")
    
    
    start_run(experiment_id=exp.experiment_id, run_name="Logistic Regression Model")
    
    log_model = train_model("logistic_regression", X_train, y_train, 
                        penalty="l1", C=0.9, solver="liblinear")
    accuracy, mse = test_model("logistic_regression", log_model, X_test, y_test)
    end_run()
        
    start_run(experiment_id=exp.experiment_id, run_name="svm Model")
    svm_model = train_model("svm", X_train, y_train,
                            kernel="rbf", C=15.0, gamma="auto")
    
    accuracy, mse = test_model("svm", svm_model, X_test, y_test)
    end_run()
    
if __name__ == "__main__":
    main()