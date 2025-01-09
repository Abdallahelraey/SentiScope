from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

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
    

@dataclass(frozen=True)
class FeatureTransformConfig:
    root_dir: Path
    data_file_path: Path
    data_file: Path
    features_dir: Path
    text_column: str
    sentiment_column: str
    train_size: float
    random_state: int
    vectorizer_type: str  # 'tfidf', 'bow', or 'word2vec'
    max_features: int
    ngram_range: Tuple[int, int]
    word2vec_params: Optional[Dict] = None