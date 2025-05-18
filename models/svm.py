from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
from monitoring_logging.logging_setup import setup_logging
from mlflow import log_metric, log_params, log_artifact, set_tags, models, sklearn, log_param
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

logging = setup_logging()

def train_svm_model(X_train, y_train,  **kwargs):
    """
    Train a SVC model.
    """
    try:
        log_params(kwargs)
        model = SVC(random_state=42, **kwargs)
        model.fit(X_train, y_train)
        sig = models.infer_signature(X_train, model.predict(X_train))
        sklearn.log_model(model, "SVC", signature=sig, input_example=X_train.iloc[0:2])
        logging.info("SVC model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Error training SVC model: {e}")
        return None
    
    
def evaluate_svm_model(model, X_test, y_test):
    """
    Evaluate the SVC model.
    """
    try:
        predictions = model.predict(X_test)
        accuracy = (predictions == y_test).mean()
        mean_squared_error_value = mean_squared_error(y_test, predictions)
        
        logging.info(f"SVC model mean squared error: {mean_squared_error_value}")
        logging.info(f"SVC model accuracy: {accuracy}")
        
        log_metric("mean_squared_error", mean_squared_error_value)
        log_metric("accuracy", accuracy)
        log_metric("MSE", mean_squared_error_value)
        log_param("model_type", "SVC")
    
        conf_mat = confusion_matrix(y_test, predictions, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_mat, display_labels=model.classes_
        )
        
        conf_mat_disp.plot()
        plt.savefig("confusion_matrix.png")
        log_artifact("confusion_matrix.png")
        set_tags({"model_type": "SVC", 
                  "accuracy": accuracy,
                  "mean_squared_error": mean_squared_error_value
                  })

        
        return accuracy, mean_squared_error_value
    except Exception as e:
        logging.error(f"Error evaluating SVC model: {e}")
        return None
    
    
def save_svm_model(model, path):
    """
    Save the trained SVC model to the specified path.
    """
    try:
        import joblib
        joblib.dump(model, path)
        logging.info(f"SVC model saved to {path}")
    except Exception as e:
        logging.error(f"Error saving SVC model: {e}")