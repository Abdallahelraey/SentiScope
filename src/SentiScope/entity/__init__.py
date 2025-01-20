from dataclasses import dataclass
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any

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

@dataclass(frozen=True)
class ModelDevelopmentConfig:
    root_dir: Path
    data_files_path: Path

    
@dataclass
class TrainingResult:
    """Stores the results of model training and evaluation"""
    model_name: str
    metrics: Dict[str, Any]
    predictions: np.ndarray
    parameters: Dict[str, Any]
    trained_model: Any  # Store the trained model instance
    


@dataclass
class TransformerModelConfig:
    """Configuration class for transformer-based model settings"""
    root_dir: Path
    data_file_path: Path
    model_name: str
    text_column: str
    label_column: str
    max_length: int
    batch_size: int
    num_labels: int
    labels: List[str]  
    
    
@dataclass(frozen=True)
class MLflowConfig:
    root_dir: Path
    experiment_name: str
    run_name: str
    tracking_uri: str
    artifact_location: Optional[str]
    default_tags: Dict[str, str]
    dynamic_tags: Dict[str, bool]
    logging: Dict[str, bool]
    basemodel: Dict[str, str]
    advancedmodel: Dict[str, str]