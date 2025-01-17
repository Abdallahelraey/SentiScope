from SentiScope.config.configuration import ConfigurationManager
from SentiScope.components.data_ingestion import DataIngestion
from SentiScope.logging import logger


class DataIngestionPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        DataFrame = data_ingestion.Ingest_data_uri()
        # DataFrame.head(10)

