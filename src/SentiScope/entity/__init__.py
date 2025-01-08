from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataProfilerConfig:
    root_dir: Path
    data_file: Path
    profile_folder: Path
    profile_file: Path
    text_column: str
    sentiment_column: str