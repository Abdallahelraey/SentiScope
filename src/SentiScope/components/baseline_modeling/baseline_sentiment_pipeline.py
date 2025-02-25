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
from SentiScope.entity import ModelDevelopmentConfig
from SentiScope.components.mlops.tracking import MLflowTracker
from SentiScope.components.baseline_modeling.training_manager import TrainingManager
from SentiScope.components.baseline_modeling.logistic_regression_model import LogisticRegressionModel

import joblib
class SentimentPipeline:
    """Coordinates all components of the sentiment analysis system"""
    
    def __init__(self, config: ModelDevelopmentConfig, mlflow_tracker: MLflowTracker):
        self.config = config
        self.data_files_path = self.config.data_files_path
        
        # Create timestamped output directory
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(self.config.root_dir) / self.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create models directory
        self.models_dir = self.output_dir / 'models'
        self.models_dir.mkdir(exist_ok=True)
        
        self.mlflow_tracker = mlflow_tracker
        logger.info("Model baseline mlflow_tracker initialized successfully.")        
        
        self.training_manager = TrainingManager(self.output_dir, self.mlflow_tracker)
        self.models = {
            'logistic_regression': LogisticRegressionModel()
        }
        
        logger.info(f"Initialized SentimentPipeline with output directory: {self.output_dir}")
    
    def prepare_data(self):
        """Load and prepare data for training"""
        try:
            logger.info("Loading prepared data from artifacts...")
            
            # Construct paths to the data files
            x_train_path = self.data_files_path / "X_train.npy"
            x_test_path = self.data_files_path / "X_test.npy"
            y_train_path = self.data_files_path / "y_train.npy"
            y_test_path = self.data_files_path / "y_test.npy"

            # Load the data
            X_train_npy = np.load(x_train_path, allow_pickle=True)
            X_test_npy = np.load(x_test_path, allow_pickle=True)
            y_train = np.load(y_train_path, allow_pickle=True)
            y_test = np.load(y_test_path, allow_pickle=True)
            
            # Convert to sparse matrices
            X_train_sparse = sparse.csr_matrix(X_train_npy.all())
            X_test_sparse = sparse.csr_matrix(X_test_npy.all())

            logger.info("Data successfully loaded and converted to sparse format")
            
            # Save data info to metadata
            metadata = {
                'timestamp': self.timestamp,
                'data_shapes': {
                    'X_train': X_train_sparse.shape,
                    'X_test': X_test_sparse.shape,
                    'y_train': y_train.shape,
                    'y_test': y_test.shape
                },
                'data_source': {
                    'X_train': str(x_train_path),
                    'X_test': str(x_test_path),
                    'y_train': str(y_train_path),
                    'y_test': str(y_test_path)
                }
            }
            
            metadata_path = self.output_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            # Log metadata as an artifact
            self.mlflow_tracker.log_artifact(str(metadata_path), "metadata")
            
            return X_train_sparse, X_test_sparse, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def save_model(self, model, name: str):
        """Save a trained model to disk"""
        try:
            model_path = self.models_dir / f"{name}.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Saved model {name} to {model_path}")
            
            # Log the saved model as an artifact
            self.mlflow_tracker.log_artifact(str(model_path), "models")
            return str(model_path)
        except Exception as e:
            logger.error(f"Error saving model {name}: {str(e)}")
            raise
    
    def train_models(self):
        """Train all registered models"""
        # Start a new MLflow run for the training pipeline
        self.mlflow_tracker.start_run(run_name="Model_Training", nested=True)
        
        try:
            logger.info("Starting model training pipeline")
            
            x_train, x_test, y_train, y_test = self.prepare_data()
            data_split = {
                "X_train": x_train,
                "X_test": x_test,
                "y_train": y_train,
                "y_test": y_test
            }
            
            results = {}
            model_paths = {}
            for name, model in self.models.items():
                logger.info(f"Training model: {name}")
                # Train and evaluate the model
                training_result = self.training_manager.train_and_evaluate(
                    model, name, data_split
                )
                results[name] = training_result
                
                # Save the trained model from the training result
                model_path = self.save_model(training_result.trained_model, name)
                model_paths[name] = model_path
                
                # Log the model to MLflow
                self.mlflow_tracker.log_model(training_result.trained_model, "models")
                
                # Construct the model URI dynamically
                run_id = self.mlflow_tracker.run.info.run_id  # Get the current run ID
                model_uri = f"runs:/{run_id}/models"
                
                # Register the model in MLflow Model Registry
                self.mlflow_tracker.register_model(model_uri=model_uri, model_name=name)
                
                # Check if this model is the one specified in the config
                if name == self.mlflow_tracker.config.basemodel.default_model_name: 
                    # Transition the model to the "Production" stage
                    self.mlflow_tracker.transition_model_stage(name)
                    logger.info(f"Model '{name}' registered and transitioned to production.")
                else:
                    logger.info(f"Model '{name}' registered but not transitioned to production.")

            
            # Save final summary
            summary = {
                'timestamp': self.timestamp,
                'models_trained': list(self.models.keys()),
                'model_paths': model_paths,
                'results': {
                    name: {
                        'metrics': result.metrics,
                        'parameters': result.parameters,
                        'model_path': model_paths[name]
                    }
                    for name, result in results.items()
                }
            }
            
            summary_path = self.output_dir / 'training_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=4)
            
            # Log the training summary as an artifact
            self.mlflow_tracker.log_artifact(str(summary_path), "training_summary")
            
            logger.info("Model training pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in train_models: {str(e)}")
            raise
        finally:
            # End the MLflow run
            self.mlflow_tracker.end_run()