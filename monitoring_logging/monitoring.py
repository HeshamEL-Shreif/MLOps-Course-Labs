import mlflow
from .logging_setup import setup_logging

def monitoring(name):
    """
    Set up MLflow tracking and logging.
    """
    logger = setup_logging()
    try:

        mlflow.set_tracking_uri("http://127.0.0.1:5001")
    
        exp = mlflow.set_experiment(name)
        logger.info(f"Experiment created with ID: {exp.experiment_id}")
        logger.info(f"Experiment ID: {exp.experiment_id}")
    
        return exp
    
    except Exception as e:
        logger.error(f"Error setting up MLflow: {e}")
        return None
    
exp = None

def set_experiment(name="Churn Prediction Experiment"):
    """
    Set the experiment ID for MLflow.
    """
    global exp
    if exp is None:
        exp = monitoring(name)
    else:
        logger = setup_logging()
        logger.info(f"Experiment ID already set: {exp.experiment_id}")
        
    return exp