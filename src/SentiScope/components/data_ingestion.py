from SentiScope.utils.data_utils import download_data, unzip_data,load_data_to_dataframe
from SentiScope.config.configuration import ConfigurationManager
from SentiScope.entity import DataIngestionConfig
from SentiScope.constants import *
from SentiScope.utils.file_utils import *
from SentiScope.utils.config_utils import *



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.data_uri = self.config.source_URL
        self.root_dir = self.config.root_dir
        self.local_data_file = self.config.local_data_file
        self.unzip_dir = self.config.unzip_dir

    def Ingest_data_uri(self):
        try:
            logger.info(f"Downloading data from {self.data_uri}")
            download_data(self.data_uri, self.local_data_file) 
            logger.info(f"Data downloaded to {self.local_data_file}")

            logger.info(f"Unzipping data to {self.unzip_dir}")
            unzip_data(self.local_data_file, self.unzip_dir)
            logger.info(f"Data unzipped successfully")

            logger.info(f"Loading data from {self.local_data_file}")
            df = load_data_to_dataframe(self.local_data_file)
            logger.info(f"Data loaded into DataFrame")

            return df
        
        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise 
