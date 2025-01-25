from SentiScope.config.configuration import ConfigurationManager
from SentiScope.components.data_pipeline.data_transformation import FeatureTransformer
from SentiScope.components.mlops.tracking import MLflowTracker
from SentiScope.logging import logger


class BaseLineInferancePipeline:
    def __init__(self, mlflow_tracker: MLflowTracker):
        self.mlflow_tracker = mlflow_tracker
    def main(self, model_name, stage, data, vectorizer_path, label_encoder_path):
        config = ConfigurationManager()
        feature_transform_config = config.get_feature_transform_config()
        feature_transformer = FeatureTransformer(config=feature_transform_config, mlflow_tracker= self.mlflow_tracker)
        production_model = self.mlflow_tracker.load_model(model_name, stage)
        transformed_data = feature_transformer.preprocess_for_prediction(data,vectorizer_path, label_encoder_path)
        predictions = self.mlflow_tracker.predict(production_model, transformed_data)
        samplee_metrics = self.mlflow_tracker.log_sample_predictions(predictions)
        original_labels = feature_transformer.label_encoder.inverse_transform(predictions)
        logger.info(f"Sample predictions: {original_labels}")
        return samplee_metrics
