from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from monitoring_logging.logging_setup import setup_logging
from mlflow import log_metric, log_params, log_artifact, set_tags, models, sklearn, log_param
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

logging = setup_logging()

def train_logistic_regression_model(X_train, y_train,  **kwargs):
    """
    Train a Logistic Regression model.
    """
    try:
        log_params(kwargs)
        model = LogisticRegression(random_state=42, **kwargs)
        model.fit(X_train, y_train)
        sig = models.infer_signature(X_train, model.predict(X_train))
        sklearn.log_model(model, "logistic_regression_model", signature=sig, input_example=X_train.iloc[0:2])
        logging.info("Logistic Regression model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Error training Logistic Regression model: {e}")
        return None
    
    
def evaluate_logistic_regression_model(model, X_test, y_test):
    """
    Evaluate the Logistic Regression model.
    """
    try:
        predictions = model.predict(X_test)
        accuracy = (predictions == y_test).mean()
        mean_squared_error_value = mean_squared_error(y_test, predictions)
        
        logging.info(f"Logistic Regression model mean squared error: {mean_squared_error_value}")
        logging.info(f"Logistic Regression model accuracy: {accuracy}")
        
        log_metric("mean_squared_error", mean_squared_error_value)
        log_metric("accuracy", accuracy)
        log_metric("MSE", mean_squared_error_value)
        log_param("model_type", "Logistic Regression")
    
        conf_mat = confusion_matrix(y_test, predictions, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_mat, display_labels=model.classes_
        )
        
        conf_mat_disp.plot()
        plt.savefig("confusion_matrix.png")
        log_artifact("confusion_matrix.png")
        set_tags({"model_type": "Logistic Regression", 
                  "accuracy": accuracy,
                  "mean_squared_error": mean_squared_error_value
                  })

        
        return accuracy, mean_squared_error_value
    except Exception as e:
        logging.error(f"Error evaluating Logistic Regression model: {e}")
        return None
    
    
def save_logistic_regression_model(model, path):
    """
    Save the trained Logistic Regression model to the specified path.
    """
    try:
        import joblib
        joblib.dump(model, path)
        logging.info(f"Logistic Regression model saved to {path}")
    except Exception as e:
        logging.error(f"Error saving Logistic Regression model: {e}")