from SentiScope.config.configuration import ConfigurationManager
from SentiScope.components.data_pipeline.data_ingestion import DataIngestion
from SentiScope.components.mlops.tracking import MLflowTracker
from SentiScope.logging import logger


class DataIngestionPipeline:
    def __init__(self, mlflow_tracker: MLflowTracker):
        self.mlflow_tracker =mlflow_tracker
    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config, mlflow_tracker= self.mlflow_tracker)
        DataFrame = data_ingestion.Ingest_data_uri()
        # DataFrame.head(10)

