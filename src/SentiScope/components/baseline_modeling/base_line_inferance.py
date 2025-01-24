from typing import Any, List, Dict
import numpy as np
import pandas as pd
from SentiScope.config.configuration import ConfigurationManager
from SentiScope.components.mlops.tracking import MLflowTracker
from SentiScope.components.baseline_modeling.baseline_sentiment_pipeline import SentimentPipeline
from SentiScope.logging import logger


class ModelPredictionPipeline:
    """
    A class to manage model loading, prediction, and logging processes.
    
    This class encapsulates the workflow of loading a production model,
    making predictions, and logging metrics using MLflow tracking.
    """
    
    def __init__(self, 
                 mlflow_tracker: MLflowTracker, 
                 model_name: str = "logistic_regression", 
                 model_stage: str = "Production"
                 ):
        """
        Initialize the prediction pipeline.
        
        Args:
            config_manager (ConfigurationManager, optional): Configuration manager. 
                Defaults to creating a new instance if not provided.
            model_name (str, optional): Name of the model to load. Defaults to "IrisClassifier".
            model_stage (str, optional): Stage of the model to load. Defaults to "Production".
        """
        
        # Initialize MLflow tracker
        self.mlflow_tracker = mlflow_tracker
        
        # Model parameters
        self.model_name = model_name
        self.model_stage = model_stage
        
        # Placeholder for loaded model and predictions
        self.production_model = None
        self.predictions = None
    
    def load_model(self) -> Any:
        """
        Load the production model from MLflow.
        
        Returns:
            Any: Loaded machine learning model
        """
        try:
            self.production_model = self.mlflow_tracker.load_model(
                self.model_name, 
                self.model_stage
            )
            logger.info(f"Successfully loaded {self.model_name} model from {self.model_stage} stage")
            return self.production_model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions using the loaded model.
        
        Args:
            X_test (np.ndarray): Input data for predictions
        
        Returns:
            np.ndarray: Model predictions
        """
        if self.production_model is None:
            self.load_model()
        
        try:
            self.predictions = self.production_model.predict(X_test)
            return self.predictions
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def log_sample_predictions(self, num_samples: int = 5) -> Dict[str, Any]:
        """
        Log sample predictions as metrics.
        
        Args:
            num_samples (int, optional): Number of samples to log. Defaults to 5.
        
        Returns:
            Dict[str, Any]: Logged sample predictions
        """
        if self.predictions is None:
            raise ValueError("No predictions available. Call predict() first.")
        
        # Ensure we don't exceed available predictions
        num_samples = min(num_samples, len(self.predictions))
        
        # Convert sample predictions to list
        sample_preds = self.predictions[:num_samples].tolist()
        
        # Create a dictionary of sample predictions for logging
        sample_metrics = {
            f"sample_prediction_{i}": pred 
            for i, pred in enumerate(sample_preds)
        }
        
        # Log metrics
        self.mlflow_tracker.log_metrics(sample_metrics)
        
        logger.info(f"Logged {num_samples} sample predictions")
        return sample_metrics

# Example usage
def main():
    """
    Example demonstration of the ModelPredictionPipeline class.
    
    Note: This is a placeholder and should be replaced with actual data and context.
    """
    try:
        config = ConfigurationManager()
        mlflow_config = config.get_mlflow_config() 
        InferancePipeline_tracker = MLflowTracker(config=mlflow_config)
        InferancePipeline_tracker.start_run("InferancePipeline")
        # Create pipeline instance
        pipeline = ModelPredictionPipeline(mlflow_tracker=InferancePipeline_tracker)
   
        config = ConfigurationManager()
        model_development_config = config.get_model_development_config()
        sentiment_pipeline = SentimentPipeline(config=model_development_config, mlflow_tracker= InferancePipeline_tracker)
   
        _, X_test, _, _ = sentiment_pipeline.prepare_data()
        # Load model and make predictions
        pipeline.predict(X_test)
        
        # Log sample predictions
        sample_metrics = pipeline.log_sample_predictions()
        
        print("Sample prediction metrics:", sample_metrics)
        InferancePipeline_tracker.end_run()
    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")

if __name__ == "__main__":
    main()