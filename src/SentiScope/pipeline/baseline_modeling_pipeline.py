from SentiScope.config.configuration import ConfigurationManager
from SentiScope.components.baseline_modeling.baseline_sentiment_pipeline import SentimentPipeline
from SentiScope.logging import logger


class BaseLineModelingPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        model_development_config = config.get_model_development_config()
        pipeline = SentimentPipeline(config=model_development_config)
        results = pipeline.train_models()


