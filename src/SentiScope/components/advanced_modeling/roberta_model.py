from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from SentiScope.logging import logger
from SentiScope.entity import TransformerModelConfig
from SentiScope.components.mlops.tracking import MLflowTracker

class TransformerSentiment:
    def __init__(self, config: TransformerModelConfig, mlflow_tracker: MLflowTracker):
        logger.info("Initializing TransformerSentiment...")
        self.config = config
        self.MODEL = self.config.model_name
        
        self.mlflow_tracker = mlflow_tracker
        self.mlflow_tracker.start_run(run_name="Roberta model", nested=True)
        logger.info("TransformerModel mlflow_tracker initialized successfully.")
        
        # Log model configuration
        self.mlflow_tracker.log_params({
            'model_name': self.MODEL,
            'labels': self.config.labels
        })        
        
        logger.info(f"Loading tokenizer and model for {self.MODEL}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL)
        self.labels = self.config.labels
        logger.info("Tokenizer and model loaded successfully.")
        
    def predict_single_sentiment(self, text):
        """
        Predict sentiment for a single text input
        Returns: tuple (sentiment, confidence)
        """  
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        # Get model predictions
        outputs = self.model(**inputs)
        scores = outputs.logits.detach().numpy()
        probabilities = torch.nn.functional.softmax(torch.tensor(scores), dim=-1)
        
        # Determine the sentiment
        sentiment_index = np.argmax(probabilities.numpy(), axis=1)[0]
        sentiment = self.labels[sentiment_index]
        confidence = probabilities[0][sentiment_index].item()

        # Log the prediction
        self.mlflow_tracker.log_metrics({
            'confidence': confidence
        })

        return sentiment, confidence

    def predict_dataframe_sentiment(self, df):
        """
        Predict sentiment for all texts in a DataFrame
        Args:
            df: pandas DataFrame
        Returns: DataFrame with added sentiment and confidence columns
        """
        try:
            logger.info("Predicting sentiment for DataFrame...")
            
            # Create new columns for results
            df['sentiment'] = ''
            df['confidence'] = 0.0
            
            # Process each text in the DataFrame
            for idx in df.index:
                text = df.loc[idx, self.config.text_column]
                sentiment, confidence = self.predict_single_sentiment(text)
                df.loc[idx, 'sentiment'] = sentiment
                df.loc[idx, 'confidence'] = confidence
            
            logger.info("Sentiment prediction for DataFrame completed.")
            
            # Log the DataFrame with predictions as an artifact
            output_path = Path(self.config.root_dir, "predictions.csv")
            df.to_csv(output_path, index=False)
            self.mlflow_tracker.log_artifact(output_path, "predictions")
            
            return df
        except Exception as e:
            logger.error(f"Error during sentiment prediction: {str(e)}")
            raise  
        finally:
            # End the MLflow run
            self.mlflow_tracker.end_run()

    def predict_sentiment(self, input_data):
        
        # Check input type and route to appropriate prediction method
        if isinstance(input_data, str):
            # Single text prediction
            return self.predict_single_sentiment(input_data)
        
        elif isinstance(input_data, pd.DataFrame):
            # DataFrame prediction
            return self.predict_dataframe_sentiment(input_data)
        
        else:
            # Raise error for unsupported input type
            raise TypeError(f"Unsupported input type: {type(input_data)}. "
                            "Input must be a string or pandas DataFrame.")