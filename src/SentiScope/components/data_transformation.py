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

class FeatureTransformer:
    def __init__(self, config: FeatureTransformConfig):
        """
        Initialize the FeatureTransformer with configuration settings.
        
        Parameters:
        config (FeatureTransformConfig): Configuration object containing transformation parameters
        """
        logger.info("Initializing FeatureTransformer...")
        self.config = config
        self.path = self.config.data_file_path
        self.df = pd.read_csv(self.path)
        
        # Create output directories
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(self.config.root_dir) / self.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize encoders and vectorizers
        self.label_encoder = LabelEncoder()
        self.vectorizer = self.config.vectorizer_type
        self.word2vec_model = None
        
        logger.info("FeatureTransformer initialized successfully.")

    def _split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data into training and testing sets.
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
        
        logger.info(f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")
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
            joblib.dump(self.vectorizer, self.output_dir / f'{self.config.vectorizer_type}_vectorizer.joblib')
            
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
            self.word2vec_model.save(str(self.output_dir / 'word2vec_model.model'))
            
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
        joblib.dump(self.label_encoder, self.output_dir / 'label_encoder.joblib')
        
        return y_train, y_test

    def transform_and_save(self) -> str:
        """
        Execute the complete transformation pipeline and save results.
        
        Returns:
        str: Path to the output directory
        """
        try:
            logger.info("Starting feature transformation pipeline...")
            
            # Split data
            train_df, test_df = self._split_data()
            
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
                }
            }
            
            with open(self.output_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Feature transformation completed. Results saved to: {self.output_dir}")
            return str(self.output_dir)
            
        except Exception as e:
            logger.error(f"Error during feature transformation: {str(e)}")
            raise
