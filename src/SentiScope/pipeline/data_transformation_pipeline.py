from SentiScope.config.configuration import ConfigurationManager
from SentiScope.components.data_pipeline.data_transformation import FeatureTransformer
from SentiScope.components.mlops.tracking import MLflowTracker
from SentiScope.logging import logger


class DataTransformerPipeline:
    def __init__(self, mlflow_tracker: MLflowTracker):
        self.mlflow_tracker =mlflow_tracker
    def main(self):
        config = ConfigurationManager()
        feature_transform_config = config.get_feature_transform_config()
        transformer = FeatureTransformer(config=feature_transform_config, mlflow_tracker= self.mlflow_tracker)
        output_path = transformer.transform_and_save()
        logger.info(f"Feature transformation completed. Output saved at: {output_path}")

