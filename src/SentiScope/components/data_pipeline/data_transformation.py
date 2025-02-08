import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
import joblib
import json
from datetime import datetime
from SentiScope.logging import logger
from SentiScope.entity import FeatureTransformConfig
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from SentiScope.components.mlops.tracking import MLflowTracker
from SentiScope.config.configuration import ConfigurationManager
class FeatureTransformer:
    def __init__(self, config: FeatureTransformConfig, mlflow_tracker: MLflowTracker, create_timestamp: bool = True):
        """
        Initialize the FeatureTransformer with configuration settings.
        
        Parameters:
        config (FeatureTransformConfig): Configuration object containing transformation parameters
        """
        logger.info("Initializing FeatureTransformer...")
        self.config = config
        self.path = self.config.data_file_path
        self.df = pd.read_csv(self.path)
        self.output_dir = Path(self.config.root_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if create_timestamp:
            # Create output directories
            self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = Path(self.config.root_dir) / self.timestamp
            
        self.mlflow_tracker = mlflow_tracker
        self.mlflow_tracker.start_run(run_name="Data Transformation",nested=True)
        logger.info("data transformation mlflow_tracker initialized successfully.")
                       
        # Initialize encoders and vectorizers
        self.label_encoder = LabelEncoder()
        self.vectorizer = self.config.vectorizer_type
        self.word2vec_model = None
        
        logger.info("FeatureTransformer initialized successfully.")

    def _split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data into training and testing sets and save them as CSV files.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Train and test dataframes
        """
        logger.info("Splitting data into train and test sets...")
        
        # Check class distribution if stratification is requested
        if self.config.sentiment_column:
            class_counts = self.df[self.config.sentiment_column].value_counts()
            min_samples = class_counts.min()
            
            if min_samples < 2:
                logger.warning(f"Found class(es) with less than 2 samples. Disabling stratification.")
                stratify = None
            else:
                stratify = self.df[self.config.sentiment_column]
        else:
            stratify = None
        
        train_df, test_df = train_test_split(
            self.df,
            train_size=self.config.train_size,
            random_state=self.config.random_state,
            stratify=stratify
        )
        
        # Save train and test splits as CSV files
        train_path = self.output_dir / 'train_split.csv'
        test_path = self.output_dir / 'test_split.csv'
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"Train set size: {len(train_df)}, saved to: {train_path}")
        logger.info(f"Test set size: {len(test_df)}, saved to: {test_path}")
        
        return train_df, test_df

    def _initialize_vectorizer(self):
        """
        Initialize the appropriate vectorizer based on configuration.
        """
        logger.info(f"Initializing {self.config.vectorizer_type} vectorizer...")
        if self.config.vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=self.config.max_features,
                ngram_range=self.config.ngram_range
            )
        elif self.config.vectorizer_type == 'bow':
            self.vectorizer = CountVectorizer(
                max_features=self.config.max_features,
                ngram_range=self.config.ngram_range
            )
        elif self.config.vectorizer_type == 'word2vec':
            # Word2Vec will be initialized during transformation
            pass
        else:
            raise ValueError(f"Unsupported vectorizer type: {self.config.vectorizer_type}")

    def _transform_text_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform text data into numerical features using the specified method.
        """
        logger.info("Transforming text features...")
        
        if self.config.vectorizer_type in ['tfidf', 'bow']:
            # Transform using TF-IDF or Bag-of-Words
            X_train = self.vectorizer.fit_transform(train_df[self.config.text_column])
            X_test = self.vectorizer.transform(test_df[self.config.text_column])
            
            # Save vectorizer
            vectorizer_path = self.output_dir / f'{self.config.vectorizer_type}_vectorizer.joblib'
            joblib.dump(self.vectorizer, vectorizer_path)
            
            # Log vectorizer as an artifact
            self.mlflow_tracker.log_artifact(str(vectorizer_path), "vectorizer")
            
        elif self.config.vectorizer_type == 'word2vec':
            # Initialize and train Word2Vec model
            texts = train_df[self.config.text_column].apply(str.split).values
            self.word2vec_model = Word2Vec(
                sentences=texts,
                vector_size=self.config.word2vec_params.get('vector_size', 100),
                window=self.config.word2vec_params.get('window', 5),
                min_count=self.config.word2vec_params.get('min_count', 1),
                workers=self.config.word2vec_params.get('workers', 4)
            )
            
            # Transform texts to vectors by averaging word vectors
            X_train = np.array([
                np.mean([self.word2vec_model.wv[word] 
                        for word in text.split() 
                        if word in self.word2vec_model.wv], axis=0)
                for text in train_df[self.config.text_column]
            ])
            X_test = np.array([
                np.mean([self.word2vec_model.wv[word]
                        for word in text.split()
                        if word in self.word2vec_model.wv], axis=0)
                for text in test_df[self.config.text_column]
            ])
            
            # Save Word2Vec model
            word2vec_path = self.output_dir / 'word2vec_model.model'
            self.word2vec_model.save(str(word2vec_path))
            
            # Log Word2Vec model as an artifact
            self.mlflow_tracker.log_artifact(str(word2vec_path), "word2vec_model")
            
        logger.info("Text feature transformation completed.")
        return X_train, X_test

    def _transform_labels(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform labels using LabelEncoder.
        """
        if not self.config.sentiment_column:
            logger.info("No sentiment column specified. Skipping label transformation.")
            return None, None
            
        logger.info("Transforming labels...")
        y_train = self.label_encoder.fit_transform(train_df[self.config.sentiment_column])
        y_test = self.label_encoder.transform(test_df[self.config.sentiment_column])
        
        # Save label encoder
        label_encoder_path = self.output_dir / 'label_encoder.joblib'
        joblib.dump(self.label_encoder, label_encoder_path)
        
        # Log label encoder as an artifact
        self.mlflow_tracker.log_artifact(str(label_encoder_path), "label_encoder")
        
        return y_train, y_test
    
    def transform_and_save(self) -> str:
        """
        Execute the complete transformation pipeline and save results.
        
        Returns:
            str: Path to the output directory
        """
        try:
            logger.info("Starting feature transformation pipeline...")
            
            # Log configuration parameters
            self.mlflow_tracker.log_params({
                'vectorizer_type': self.config.vectorizer_type,
                'max_features': self.config.max_features,
                'ngram_range': self.config.ngram_range,
                'train_size': self.config.train_size,
                'random_state': self.config.random_state
            })
            
            # Split data and save CSV files
            train_df, test_df = self._split_data()
            
            # Log train and test split sizes
            self.mlflow_tracker.log_metrics({
                'train_set_size': len(train_df),
                'test_set_size': len(test_df)
            })
            
            # Initialize vectorizer
            self._initialize_vectorizer()
            
            # Transform features
            X_train, X_test = self._transform_text_features(train_df, test_df)
            
            # Transform labels
            y_train, y_test = self._transform_labels(train_df, test_df)
            
            # Save transformed data
            np.save(self.output_dir / 'X_train.npy', X_train)
            np.save(self.output_dir / 'X_test.npy', X_test)
            if y_train is not None and y_test is not None:
                np.save(self.output_dir / 'y_train.npy', y_train)
                np.save(self.output_dir / 'y_test.npy', y_test)
                
            # Log transformed data shapes
            self.mlflow_tracker.log_metrics({
                'X_train_shape': X_train.shape[0],
                'X_test_shape': X_test.shape[0],
                'y_train_shape': y_train.shape[0] if y_train is not None else 0,
                'y_test_shape': y_test.shape[0] if y_test is not None else 0
            })
            
            
            # Save configuration and metadata
            metadata = {
                'timestamp': self.timestamp,
                'config': {
                    'vectorizer_type': self.config.vectorizer_type,
                    'max_features': self.config.max_features,
                    'ngram_range': self.config.ngram_range,
                    'train_size': self.config.train_size,
                    'random_state': self.config.random_state
                },
                'data_shapes': {
                    'X_train': X_train.shape,
                    'X_test': X_test.shape,
                    'y_train': y_train.shape if y_train is not None else None,
                    'y_test': y_test.shape if y_test is not None else None
                },
                'split_files': {
                    'train_split': 'train_split.csv',
                    'test_split': 'test_split.csv'
                }
            }
            
            metadata_path = self.output_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
                
            # Log metadata as an artifact
            self.mlflow_tracker.log_artifact(str(metadata_path), "metadata")
            
            # Log train and test split files as artifacts
            self.mlflow_tracker.log_artifact(str(self.output_dir / 'train_split.csv'), "data_splits")
            self.mlflow_tracker.log_artifact(str(self.output_dir / 'test_split.csv'), "data_splits")
            
            logger.info(f"Feature transformation completed. Results saved to: {self.output_dir}")
            return str(self.output_dir)
            
        except Exception as e:
            logger.error(f"Error during feature transformation: {str(e)}")
            raise
        finally:
            # End the MLflow run
            self.mlflow_tracker.end_run()
            
    def get_transformer_models_path(self):
        """
        Get the trained vectorizer and label encoder models.
        
        Returns:
            Tuple[Path, Path]: Paths to the trained vectorizer and label encoder models.
        """
        report_data = ConfigurationManager().get_latest_report_baselinemodels()

        if "timestamp" not in report_data:
            raise KeyError("Missing 'timestamp' in metadata.json")

        timestamp = report_data["timestamp"]

        # Ensure the timestamp is a valid folder inside output_dir
        model_dir = self.output_dir / timestamp

        # Construct file paths
        vectorizer_path = model_dir / f'{self.config.vectorizer_type}_vectorizer.joblib'
        label_encoder_path = model_dir / 'label_encoder.joblib'

        # Validate file existence
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

        if not label_encoder_path.exists():
            raise FileNotFoundError(f"Label encoder file not found: {label_encoder_path}")

        return vectorizer_path, label_encoder_path
            
            
    def preprocess_for_prediction(self, 
                                data: Union[pd.DataFrame, List[str]], 
                                vectorizer_path: Optional[str] = None,
                                label_encoder_path: Optional[str] = None) -> np.ndarray:
        logger.info("Preprocessing data for prediction...")

        # Use default paths if none are provided
        if vectorizer_path is None or label_encoder_path is None:
            default_vectorizer_path, default_label_encoder_path = self.get_transformer_models_path()
            vectorizer_path = vectorizer_path or default_vectorizer_path
            label_encoder_path = label_encoder_path or default_label_encoder_path

        # Convert input to DataFrame if it's a list of strings
        if isinstance(data, list):
            data = pd.DataFrame({self.config.text_column: data})

        # Load the vectorizer
        vectorizer = joblib.load(vectorizer_path)

        # Transform the text data
        X_pred = vectorizer.transform(data[self.config.text_column])

        # Optionally load label encoder
        if label_encoder_path is not None:
            self.label_encoder = joblib.load(label_encoder_path)

        logger.info("Data preprocessing for prediction completed.")
        return X_pred