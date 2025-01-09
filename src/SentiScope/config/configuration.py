from SentiScope.constants import (CONFIG_FILE_PATH,
                                  PARAMS_FILE_PATH)
from SentiScope.utils.file_utils import (create_directories,
                                            get_size)
from SentiScope.utils.config_utils import (read_yaml,
                                           Settings,
                                           get_settings)
from SentiScope.entity import (DataIngestionConfig,
                               DataProfilerConfig,
                               FeatureTransformConfig)
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path
import json


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_data_profiler_config(self) -> DataProfilerConfig:
        config = self.config.data_profileing

        create_directories([config.root_dir])

        data_profileing_config = DataProfilerConfig(
            root_dir=config.root_dir,
            data_file=config.data_file,
            profile_folder= config.profile_folder,
            profile_file= config.profile_file,
            text_column = config.text_column,
            sentiment_column = config.sentiment_column
        )

        return data_profileing_config
    
    def get_latest_report(self) -> Dict:
        """Locate the latest report.json file based on the timestamp folder."""
        config = self.config.data_profileing
        profiling_dir = Path(config.root_dir)

        # Get all subdirectories in data_profiling
        timestamp_dirs = [d for d in profiling_dir.iterdir() if d.is_dir()]
        
        if not timestamp_dirs:
            raise FileNotFoundError("No timestamp folders found in data_profiling.")

        # Sort directories by name (assuming timestamp format)
        latest_dir = sorted(timestamp_dirs, key=lambda x: x.name, reverse=True)[0]
        report_path = latest_dir / "report.json"

        if not report_path.exists():
            raise FileNotFoundError(f"report.json not found in {latest_dir}.")

        # Load the report.json file
        with open(report_path, "r") as f:
            report_data = json.load(f)

        return report_data

    def get_feature_transform_config(self) -> FeatureTransformConfig:
        config = self.config.feature_transformation
        report_data = self.get_latest_report()

        create_directories([config.root_dir])

        timestamp = report_data["timestamp"]
        # data_file_path = Path(str(config.data_file).format(timestamp=timestamp))
        data_file_path = Path(config.data_file_path).joinpath(f"{timestamp}", config.data_file)
 

        feature_transform_config = FeatureTransformConfig(
            root_dir=config.root_dir,
            data_file=config.data_file,
            data_file_path = data_file_path,
            features_dir=config.features_dir,
            text_column=config.text_column,
            sentiment_column=config.sentiment_column,
            train_size=config.train_size,
            random_state=config.random_state,
            vectorizer_type=config.vectorizer_type,
            max_features=config.max_features,
            ngram_range=tuple(config.ngram_range),
            word2vec_params=config.word2vec_params if hasattr(config, 'word2vec_params') else None
        )

        return feature_transform_config