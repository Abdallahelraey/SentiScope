import mlflow
from SentiScope.logging import logger
from typing import Any, Dict, Optional
from functools import wraps
from SentiScope.entity import MLflowConfig

class MLflowTracker:
    """
    A modular MLflow tracking component that can be imported and used across different modules
    without interfering with their core logic.
    """
    def __init__(self, config: MLflowConfig):
        self.config = config
        self.experiment_name = self.config.experiment_name
        self.run_name = self.config.run_name
        self.tracking_uri = self.config.tracking_uri
        self.run = None
        
        try:
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
                logger.info(f"Setting MLflow tracking URI to: {self.tracking_uri}")
                
                # Test connection before proceeding
                self._test_connection()
            
            # Get or create experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is not None:
                self.experiment_id = experiment.experiment_id
                logger.info(f"Found existing experiment: {self.experiment_name} with ID: {self.experiment_id}")
            else:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created new experiment: {self.experiment_name} with ID: {self.experiment_id}")

        except ConnectionError as e:
            logger.error(f"Failed to connect to MLflow tracking server at {self.tracking_uri}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error initializing MLflow tracker: {str(e)}")
            raise

    def _test_connection(self) -> bool:
        """Test connection to MLflow server"""
        try:
            mlflow.get_tracking_uri()
            # Try to list experiments as a connection test
            mlflow.search_experiments()
            logger.info("Successfully connected to MLflow server")
            return True
        except Exception as e:
            raise ConnectionError(f"Cannot connect to MLflow server: {str(e)}")

    def start_run(self, run_name: str, nested: bool = False):
        """Start a new MLflow run if none is active or end the active one."""
        if run_name:
            self.run_name = run_name
        try:
            # Check for an active run
            active_run = mlflow.active_run()
            if active_run and not nested:
                logger.warning(f"An active run with ID {active_run.info.run_id} exists. Ending it now.")
                mlflow.end_run()
            
            self.run = mlflow.start_run(
                experiment_id=self.experiment_id, 
                run_name=self.run_name,
                nested=nested  
            )
            logger.info(f"Started new MLflow run with ID: {self.run.info.run_id}")
            return self.run.info.run_id
        except Exception as e:
            logger.error(f"Error starting MLflow run: {str(e)}")
            raise



    def log_params(self, params: dict):
        """Log parameters to MLflow"""
        try:
            if self.run is None:
                raise RuntimeError("No active MLflow run. Call start_run() first.")
            mlflow.log_params(params)
            logger.debug(f"Logged parameters: {params}")
        except Exception as e:
            logger.error(f"Error logging parameters: {str(e)}")
            raise

    def log_metrics(self, metrics: dict):
        """Log metrics to MLflow"""
        try:
            if self.run is None:
                raise RuntimeError("No active MLflow run. Call start_run() first.")
            mlflow.log_metrics(metrics)
            logger.debug(f"Logged metrics: {metrics}")
        except Exception as e:
            logger.error(f"Error logging metrics: {str(e)}")
            raise

    def log_model(self, model: Any, artifact_path: str):
        """Log ML model to MLflow"""
        try:
            if self.run is None:
                raise RuntimeError("No active MLflow run. Call start_run() first.")
            mlflow.sklearn.log_model(sk_model=model, artifact_path=artifact_path)
            logger.info(f"Logged model to artifact path: {artifact_path}")
        except Exception as e:
            logger.error(f"Error logging model: {str(e)}")
            raise

    def log_artifact(self, artifact_path: str, destination_path: str = None):
        """Log artifact to MLflow"""
        try:
            if self.run is None:
                raise RuntimeError("No active MLflow run. Call start_run() first.")
            mlflow.log_artifact(artifact_path, destination_path)
            logger.info(f"Logged artifact from {artifact_path} to {destination_path or 'default path'}")
        except Exception as e:
            logger.error(f"Error logging artifact: {str(e)}")
            raise

    def end_run(self):
        """End the current MLflow run"""
        try:
            if self.run:
                mlflow.end_run()
                logger.info("Ended MLflow run")
                self.run = None
        except Exception as e:
            logger.error(f"Error ending MLflow run: {str(e)}")
            raise
