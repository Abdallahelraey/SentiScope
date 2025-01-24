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
        self.artifact_location = self.config.artifact_location
        self.run = None
        
        try:
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
                logger.info(f"Set MLflow tracking URI to: {self.tracking_uri}")
                self._test_connection()
            
            # Get or create experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment:
                self.experiment_id = experiment.experiment_id
                logger.info(f"Existing experiment found: {self.experiment_name} (ID: {self.experiment_id})")
            else:
                self.experiment_id = mlflow.create_experiment(
                    name=self.experiment_name,
                    artifact_location=self.artifact_location
                )
                logger.info(f"Created new experiment: {self.experiment_name} (ID: {self.experiment_id})")

        except ConnectionError as e:
            logger.error(f"MLflow connection failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            raise

    def _test_connection(self) -> bool:
        """Validate connection to MLflow tracking server"""
        try:
            mlflow.search_experiments()
            logger.info("Successfully connected to MLflow server")
            return True
        except Exception as e:
            raise ConnectionError(f"MLflow connection test failed: {str(e)}")

    def start_run(self, run_name: str = None, nested: bool = False) -> str:
        """Start a new MLflow run"""
        try:
            self.run_name = run_name or self.run_name
            active_run = mlflow.active_run()
            
            if active_run and not nested:
                logger.warning(f"Ending active run {active_run.info.run_id}")
                mlflow.end_run()
            
            self.run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=self.run_name,
                nested=nested
            )
            logger.info(f"Started run: {self.run.info.run_id}")
            return self.run.info.run_id
        except Exception as e:
            logger.error(f"Run start failed: {str(e)}")
            raise

    def log_params(self, params: dict):
        """Log parameters to current run"""
        self._validate_active_run()
        try:
            mlflow.log_params(params)
            logger.debug(f"Logged parameters: {params}")
        except Exception as e:
            logger.error(f"Parameter logging failed: {str(e)}")
            raise

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """Log metrics to current run"""
        self._validate_active_run()
        try:
            mlflow.log_metrics(metrics, step=step)
            logger.debug(f"Logged metrics: {metrics}")
        except Exception as e:
            logger.error(f"Metrics logging failed: {str(e)}")
            raise

    def log_model(self, model: Any, artifact_path: str):
        """Log ML model artifact"""
        self._validate_active_run()
        try:
            mlflow.sklearn.log_model(model, artifact_path)
            logger.info(f"Model logged to: {artifact_path}")
        except Exception as e:
            logger.error(f"Model logging failed: {str(e)}")
            raise

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log file artifact"""
        self._validate_active_run()
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.info(f"Artifact logged: {local_path}")
        except Exception as e:
            logger.error(f"Artifact logging failed: {str(e)}")
            raise

    def end_run(self):
        """Terminate current run"""
        try:
            if self.run:
                mlflow.end_run()
                logger.info(f"Ended run: {self.run.info.run_id}")
                self.run = None
        except Exception as e:
            logger.error(f"Run termination failed: {str(e)}")
            raise

    def register_model(self, model_uri: str, model_name: str):
        """Register model in MLflow registry"""
        self._validate_active_run()
        try:
            mlflow.register_model(model_uri, model_name)
            logger.info(f"Registered model: {model_name}")
        except Exception as e:
            logger.error(f"Model registration failed: {str(e)}")
            raise

    def transition_model_stage(self, model_name: str, stage: str = "Production"):
        """Transition model to specified stage"""
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            version = client.get_latest_versions(model_name, stages=["None"])[0].version
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            logger.info(f"Model {model_name} v{version} -> {stage}")
        except Exception as e:
            logger.error(f"Stage transition failed: {str(e)}")
            raise

    def load_model(self, model_name: str, stage: str = "Production"):
        """Load model from MLflow registry"""
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            # model = client.get_model_by_name(model_name)
            model_version = client.get_latest_versions(model_name, stages=[stage])[0]
            model_uri = f"models:/{model_name}/{model_version.version}"
            loaded_model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded model: {model_name} v{model_version.version}")
            return loaded_model
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")



    def _validate_active_run(self):
        """Check for active run before logging operations"""
        if not mlflow.active_run():
            raise RuntimeError("No active MLflow run. Call start_run() first.")

    def __enter__(self):
        """Context manager entry"""
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.end_run()
