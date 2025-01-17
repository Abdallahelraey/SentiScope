from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List
from sklearn.metrics import classification_report
import numpy as np
from scipy.sparse import issparse
from scipy import sparse
import os
import json
from datetime import datetime
from SentiScope.logging import logger
from datetime import datetime
from SentiScope.entity import TrainingResult

class TrainingManager:
    """Handles model training and evaluation"""
    
    def __init__(self, output_dir: Path):
        self.training_history: List[TrainingResult] = []
        self.output_dir = output_dir
        logger.info(f"Initialized TrainingManager with output directory: {output_dir}")
    
    def _validate_data_split(self, data_split):
        """Validate the data split dictionary"""
        logger.info("Validating data split...")
        
        if not isinstance(data_split, dict):
            msg = f"data_split must be a dictionary, got {type(data_split)}"
            logger.error(msg)
            raise TypeError(msg)
            
        required_keys = ["X_train", "X_test", "y_train", "y_test"]
        
        missing = [key for key in required_keys if key not in data_split]
        if missing:
            msg = f"Missing required keys in data_split: {missing}"
            logger.error(msg)
            raise ValueError(msg)
        
        n_train_samples = (data_split["X_train"].shape[0] if not issparse(data_split["X_train"]) 
                          else data_split["X_train"].shape[0])
        n_test_samples = (data_split["X_test"].shape[0] if not issparse(data_split["X_test"]) 
                         else data_split["X_test"].shape[0])
        
        if n_train_samples != len(data_split["y_train"]):
            msg = (f"Mismatch in training set dimensions: X_train has {n_train_samples} "
                  f"samples but y_train has {len(data_split['y_train'])} samples")
            logger.error(msg)
            raise ValueError(msg)
        
        if n_test_samples != len(data_split["y_test"]):
            msg = (f"Mismatch in test set dimensions: X_test has {n_test_samples} "
                  f"samples but y_test has {len(data_split['y_test'])} samples")
            logger.error(msg)
            raise ValueError(msg)
        
        logger.info("Data split validation completed successfully")
    
    def train_and_evaluate(self, model, model_name: str, data_split) -> TrainingResult:
        """Train a model and evaluate its performance"""
        try:
            logger.info(f"Starting training and evaluation for {model_name}")
            
            # Validate the data split
            self._validate_data_split(data_split)
            
            # Verify model has required methods
            required_methods = ['train', 'predict', 'get_params']
            for method in required_methods:
                if not hasattr(model, method):
                    msg = f"Model lacks required method: {method}"
                    logger.error(msg)
                    raise AttributeError(msg)
            
            # Train the model
            logger.info(f"Training {model_name}...")
            model.train(data_split["X_train"], data_split["y_train"])
            
            # Make predictions
            logger.info(f"Making predictions for {model_name}...")
            predictions = model.predict(data_split["X_test"])
            
            # Calculate metrics
            logger.info(f"Calculating metrics for {model_name}...")
            metrics = classification_report(
                data_split["y_test"], 
                predictions, 
                output_dict=True
            )
            
            # Create result object with trained model
            result = TrainingResult(
                model_name=model_name,
                metrics=metrics,
                predictions=predictions,
                parameters=model.get_params(),
                trained_model=model
            )
            
            # Save results to output directory
            model_dir = self.output_dir / model_name
            model_dir.mkdir(exist_ok=True)
            
            # Save predictions
            np.save(model_dir / 'predictions.npy', predictions)
            
            # Save metrics and parameters
            with open(model_dir / 'results.json', 'w') as f:
                json.dump({
                    'metrics': metrics,
                    'parameters': model.get_params()
                }, f, indent=4)
            
            self.training_history.append(result)
            
            logger.info(f"Successfully completed training and evaluation for {model_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error in train_and_evaluate for {model_name}: {str(e)}")
            raise