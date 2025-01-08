from SentiScope.constants import (CONFIG_FILE_PATH,
                                  PARAMS_FILE_PATH)
from SentiScope.utils.file_utils import (create_directories,
                                            get_size)
from SentiScope.utils.config_utils import (read_yaml,
                                           Settings,
                                           get_settings)
from SentiScope.entity import (DataIngestionConfig,
                               DataProfilerConfig)

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