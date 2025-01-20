"""
DataIngestion module for SentiScope

This module provides a class for ingesting data from a URI, unzipping it, and loading it into a Pandas DataFrame.

Attributes:
    DataIngestionConfig: The configuration class for data ingestion.

Classes:
    DataIngestion: A class that handles data ingestion from a URI.

Methods:
    ingest_data_uri(self) -> pd.DataFrame: Downloads data from a URI, unzips it, and loads it into a Pandas DataFrame.

Exceptions:
    Exception: Raised if an error occurs during data ingestion.
"""

from SentiScope.utils.data_utils import (
    download_data,
    unzip_data,
    load_data_to_dataframe,
)
from SentiScope.entity import DataIngestionConfig
from SentiScope.logging import logger
from SentiScope.components.mlops.tracking import MLflowTracker
from datetime import datetime
from pathlib import Path
import json
import os

class DataIngestion:
    def __init__(self, config: DataIngestionConfig, mlflow_tracker: MLflowTracker):
        self.config = config
        self.data_uri = self.config.source_URL
        
        self.mlflow_tracker = mlflow_tracker
        self.mlflow_tracker.start_run(run_name="Data_Ingestion",nested=True)
        logger.info("data Ingestion mlflow_tracker initialized successfully.")
        
        # Create timestamp directory
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.root_dir = Path(self.config.root_dir)
        self.output_dir = self.root_dir / self.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Update paths to use timestamp directory
        # Just use the filename from the config, not the full path
        self.local_data_file = self.output_dir / Path(self.config.local_data_file).name
        self.unzip_dir = self.output_dir / "unzipped"
        
        self.mlflow_tracker.log_params({
                "data_uri": str(self.data_uri),
                "local_data_file": str(self.local_data_file)
            })
        
        logger.info(f"Initialized DataIngestion with output directory: {self.output_dir}")

    def Ingest_data_uri(self):
        try:
            # Ensure parent directories exist
            self.local_data_file.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Downloading data from {self.data_uri}")
            download_data(self.data_uri, str(self.local_data_file))  # Convert Path to string
            logger.info(f"Data downloaded to {self.local_data_file}")
            
            # Log download metrics
            file_size = os.path.getsize(self.local_data_file)
            self.mlflow_tracker.log_metrics({
                "downloaded_file_size_bytes": file_size,
                "downloaded_file_size_mb": file_size / (1024 * 1024)
            })

            logger.info(f"Unzipping data to {self.unzip_dir}")
            self.unzip_dir.mkdir(parents=True, exist_ok=True)  # Ensure unzip directory exists
            unzip_data(str(self.local_data_file), str(self.unzip_dir))  # Convert Paths to strings
            logger.info(f"Data unzipped successfully")

            # Log unzip metrics
            unzipped_size = sum(f.stat().st_size for f in self.unzip_dir.glob('**/*') if f.is_file())
            self.mlflow_tracker.log_metrics({
                "unzipped_total_size_bytes": unzipped_size,
                "unzipped_total_size_mb": unzipped_size / (1024 * 1024)
            })


            logger.info(f"Loading data from {self.local_data_file}")
            df = load_data_to_dataframe(str(self.local_data_file))  # Convert Path to string
            logger.info(f"Data loaded into DataFrame")
            

            # Save metadata about the ingestion
            metadata = {
                'timestamp': self.timestamp,
                'data_source': str(self.data_uri),
                'local_data_file': str(self.local_data_file),
                'unzip_dir': str(self.unzip_dir),
                'data_shape': df.shape if hasattr(df, 'shape') else None
            }
            
            with open(self.output_dir / 'ingestion_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=4)
            
            return df
        
        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise
        finally:
            self.mlflow_tracker.end_run()