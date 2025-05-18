from models.logistic_regression import train_logistic_regression_model
from models.svm import train_svm_model

from monitoring_logging.logging_setup import setup_logging


def train_model(model_type, X_train, y_train, **kwargs):
    """
    Train a model based on the specified model type.
    """
    logger = setup_logging()
    
    logger.info(f"Training {model_type} model...")
    if model_type == "logistic_regression":
        return train_logistic_regression_model(X_train, y_train, **kwargs)
    elif model_type == "svm":
        return train_svm_model(X_train, y_train, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    
    