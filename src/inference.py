from monitoring_logging.logging_setup import setup_logging
from models.logistic_regression import evaluate_logistic_regression_model
from models.svm import evaluate_svm_model


def test_model(model_type, model, X_test, y_test):
    """
    Test a model based on the specified model type.
    """
    logger = setup_logging()
    logger.info(f"Testing {model_type} model...")
    if model_type == "logistic_regression":
        return evaluate_logistic_regression_model(model, X_test, y_test)
    elif model_type == "svm":
        return evaluate_svm_model(model, X_test, y_test)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")