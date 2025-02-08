from SentiScope.config.configuration import ConfigurationManager
from SentiScope.components.baseline_modeling.baseline_sentiment_pipeline import SentimentPipeline
from SentiScope.components.mlops.tracking import MLflowTracker
from SentiScope.logging import logger


class BaseLineModelingPipeline:
    def __init__(self, mlflow_tracker: MLflowTracker):
        self.mlflow_tracker =mlflow_tracker
    def main(self):
        config = ConfigurationManager()
        model_development_config = config.get_model_development_config()
        pipeline = SentimentPipeline(config=model_development_config, mlflow_tracker= self.mlflow_tracker)
        results = pipeline.train_models()


