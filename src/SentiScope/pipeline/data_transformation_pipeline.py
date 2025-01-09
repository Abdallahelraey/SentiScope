from SentiScope.config.configuration import ConfigurationManager
from SentiScope.components.data_transformation import FeatureTransformer
from SentiScope.logging import logger


class DataTransformerPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        feature_transform_config = config.get_feature_transform_config()
        transformer = FeatureTransformer(config=feature_transform_config)
        output_path = transformer.transform_and_save()
        logger.info(f"Feature transformation completed. Output saved at: {output_path}")

