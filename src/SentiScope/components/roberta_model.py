from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import pandas as pd
from SentiScope.logging import logger
from SentiScope.entity import TransformerModelConfig


class TransformerSentiment:
    def __init__(self, config: TransformerModelConfig):
        logger.info("Initializing TransformerSentiment...")
        self.config = config
        self.MODEL = self.config.model_name
        
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

        return sentiment, confidence

    def predict_dataframe_sentiment(self, df):
        """
        Predict sentiment for all texts in a DataFrame
        Args:
            df: pandas DataFrame
        Returns: DataFrame with added sentiment and confidence columns
        """
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
        return df
