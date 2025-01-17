from SentiScope.config.configuration import ConfigurationManager
from SentiScope.components.roberta_model import TransformerSentiment
from SentiScope.logging import logger
import pandas as pd

class TransformerModelPipeline:
    def __init__(self):
        pass
    def main(self):
        config_manager = ConfigurationManager()
        transformer_config = config_manager.get_transformer_config()
        Transformer_Sentiment = TransformerSentiment(config=transformer_config)
        data_path = transformer_config.data_file_path
        data = pd.read_csv(data_path).head(50)
        results = Transformer_Sentiment.predict_dataframe_sentiment(data)


