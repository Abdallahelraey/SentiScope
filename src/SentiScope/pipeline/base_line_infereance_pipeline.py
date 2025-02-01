from SentiScope.config.configuration import ConfigurationManager
from SentiScope.components.data_pipeline.data_transformation import FeatureTransformer
from SentiScope.components.mlops.tracking import MLflowTracker
from SentiScope.logging import logger
from typing import Optional

class BaseLineInferancePipeline:
    def __init__(self, mlflow_tracker: MLflowTracker):
        self.mlflow_tracker = mlflow_tracker

    def main(self, 
             model_name: str, 
             stage: str, 
             data, 
             vectorizer_path: Optional[str] = None, 
             label_encoder_path: Optional[str] = None):
        
        config = ConfigurationManager()
        feature_transform_config = config.get_feature_transform_config()
        feature_transformer = FeatureTransformer(config=feature_transform_config, mlflow_tracker=self.mlflow_tracker, create_timestamp=False)
        
        # Load the model from MLflow
        production_model = self.mlflow_tracker.load_model(model_name, stage)

        # Preprocess data, automatically handling missing paths
        transformed_data = feature_transformer.preprocess_for_prediction(data, vectorizer_path, label_encoder_path)

        # Predict using the loaded model
        predictions = self.mlflow_tracker.predict(production_model, transformed_data)
        
        # Log sample predictions
        sample_metrics = self.mlflow_tracker.log_sample_predictions(predictions)

        # Convert predictions back to original labels
        original_labels = feature_transformer.label_encoder.inverse_transform(predictions)

        logger.info(f"Sample predictions: {original_labels}")
        return original_labels
