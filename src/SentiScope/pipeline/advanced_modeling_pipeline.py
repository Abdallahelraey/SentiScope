from SentiScope.config.configuration import ConfigurationManager
from SentiScope.components.advanced_modeling.roberta_model import TransformerSentiment
from SentiScope.components.mlops.tracking import MLflowTracker
from SentiScope.logging import logger
import pandas as pd

class TransformerModelPipeline:
    def __init__(self, mlflow_tracker: MLflowTracker):
        self.mlflow_tracker =mlflow_tracker
    def main(self, data = None):
        config_manager = ConfigurationManager()
        transformer_config = config_manager.get_transformer_config()
        Transformer_Sentiment = TransformerSentiment(config=transformer_config, mlflow_tracker= self.mlflow_tracker)
        results = Transformer_Sentiment.predict_sentiment(data)
        return results


