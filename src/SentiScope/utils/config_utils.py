from pydantic_settings import BaseSettings, SettingsConfigDict
from box.exceptions import BoxValueError
import yaml
from SentiScope.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any



@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        logger.error(f"Error loading yaml file:{path_to_yaml}")
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    
class Settings(BaseSettings):

    APP_NAME: str
    APP_VERSION: str
    class Config:
        env_file = ".env"

def get_settings():
    return Settings()


class Settings(BaseSettings):

    APP_NAME: str
    APP_VERSION: str
    class Config:
        env_file = ".env"

def get_settings():
    return Settings()