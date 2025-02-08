"""
SentimentDataProfiler

This class provides tools for profiling, analyzing, and reporting on sentiment analysis datasets. It supports 
preprocessing text data, analyzing text features, and generating visualizations and reports. The primary use case 
is for datasets containing textual data and optional sentiment labels.

Attributes:
-----------
config : DataProfilerConfig
    Configuration object containing dataset paths, column names, and output directory details.
path : Path
    Path to the dataset file.
df : pd.DataFrame
    Loaded dataset as a pandas DataFrame.
text_column : str
    Name of the column containing textual data.
sentiment_column : str, optional
    Name of the column containing sentiment labels.
stop_words : set
    Set of English stopwords used during text preprocessing.
lemmatizer : WordNetLemmatizer
    Lemmatizer for reducing words to their base forms.
stemmer : PorterStemmer
    Stemmer for reducing words to their root forms.
timestamp : str
    Timestamp when the profiler is initialized, used for organizing output files.
output_dir : Path
    Path to the directory where reports and visualizations are saved.

Methods:
--------
_read_csv_file(file_path: Path) -> pd.DataFrame:
    Reads and validates the CSV file containing the dataset.

_validate_columns() -> None:
    Ensures the required text and sentiment columns are present in the dataset.

_preprocess_text(text: str) -> List[str]:
    Processes a single text entry by lowercasing, tokenizing, removing stopwords, and stemming.

_get_initial_statistics() -> Dict[str, Any]:
    Calculates basic statistics about the dataset, such as row/column counts and text lengths.

_analyze_text_features() -> Dict[str, Any]:
    Analyzes text features, such as word frequency and vocabulary size, and generates a word cloud.

_analyze_sentiment_distribution() -> Optional[Dict[str, Any]]:
    Analyzes the distribution of sentiment labels and generates a sentiment distribution plot.

generate_report() -> str:
    Generates a comprehensive report on the dataset, including statistics, visualizations, and sentiment analysis 
    (if applicable). Saves the report as JSON and generates a README file.

Usage:
------
1. Initialize the profiler with a configuration object:
    config = DataProfilerConfig(data_file="path/to/data.csv", text_column="text", sentiment_column="sentiment")
    profiler = SentimentDataProfiler(config)

2. Generate a report:
    report_path = profiler.generate_report()

3. Access the generated report and visualizations in the output directory.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
from wordcloud import WordCloud
from typing import Dict, List, Optional, Any
import json
import os
from datetime import datetime
from pathlib import Path
from SentiScope.logging import logger
from SentiScope.entity import DataProfilerConfig
from SentiScope.components.mlops.tracking import MLflowTracker
class SentimentDataProfiler:
    def __init__(self, config: DataProfilerConfig, mlflow_tracker: MLflowTracker):
        """
        Initialize the SentimentDataProfiler with a data path and column names.
        
        Parameters:
        data_path (str): Path to the CSV file containing sentiment data
        text_column (str): Name of the column containing text data
        sentiment_column (str, optional): Name of the column containing sentiment labels
        """
        # Convert string path to Path object
        logger.info("Initializing SentimentDataProfiler...")
        self.config = config
        self.path = self.config.data_file
        logger.info(f"Reading CSV file from path: {self.path}")
        self.df = self._read_csv_file(self.path)
        self.text_column = self.config.text_column
        self.sentiment_column = self.config.sentiment_column
        
        self.mlflow_tracker = mlflow_tracker
        self.mlflow_tracker.start_run(run_name="Data Profiler",nested=True)
        logger.info("data profiler mlflow_tracker initialized successfully.")
        
        # Initialize NLTK components
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            logger.info("Required NLTK data found.")
        except LookupError:
            logger.info("Downloading required NLTK data...")
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # Create output directory structure
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(os.getcwd()) / self.config.profile_folder / self.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'images').mkdir(exist_ok=True)
        logger.info(f"Output directory created at {self.output_dir}")
        
        # Validate and process
        self._validate_columns()
        logger.info("Columns validated successfully.")
        self.df['processed_text'] = self.df[self.text_column].apply(self._preprocess_text)
        logger.info("Text preprocessing completed.")

    def _read_csv_file(self, file_path: Path) -> pd.DataFrame:
        """
        Read and validate the CSV file.
        
        Parameters:
        file_path (Path): Path to the CSV file
        
        Returns:
        pd.DataFrame: The loaded DataFrame
        """
        logger.info(f"Reading CSV file from {file_path}")
        try:
            df = pd.read_csv(file_path)
            logger.info("Successfully read the CSV file.")
            return df
        except FileNotFoundError:
            logger.error(f"Error: File not found at {file_path}")
            raise
        except pd.errors.ParserError:
            logger.error("Error: There might be a parsing issue with the CSV file!")
            try:
                # Attempt to read with more flexible parsing
                df = pd.read_csv(file_path, dtype=str)
                return df
            except Exception as e:
                logger.error(f"Failed to fix parsing errors: {e}")
                raise

    def _validate_columns(self) -> None:
        """Validate that the specified columns exist in the DataFrame."""
        logger.info("Validating columns in the DataFrame.")
        if self.text_column not in self.df.columns:
            logger.error(f"Text column '{self.text_column}' not found in DataFrame.")
            raise ValueError(f"Text column '{self.text_column}' not found in DataFrame")
        if self.sentiment_column and self.sentiment_column not in self.df.columns:
            logger.error(f"Sentiment column '{self.sentiment_column}' not found in DataFrame.")
            raise ValueError(f"Sentiment column '{self.sentiment_column}' not found in DataFrame")

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess a single text string."""
        if not isinstance(text, str):
            logger.warning("Non-string text encountered during preprocessing.")
            return []
        logger.debug(f"Preprocessing text: {text[:30]}...")
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [self.stemmer.stem(token) for token in tokens
                 if token.isalnum() and token not in self.stop_words]
        logger.debug(f"Processed tokens: {tokens[:10]}")
        return tokens

    def _get_initial_statistics(self) -> Dict[str, Any]:
        """Generate basic statistics about the dataset."""
        logger.info("Generating initial dataset statistics.")
        stats = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'dtypes': {k: str(v) for k, v in self.df.dtypes.to_dict().items()},  # Convert dtypes to strings for JSON
            'missing_values': self.df.isnull().sum().to_dict()
        }

        self.df['text_length'] = self.df[self.text_column].str.len()
        stats['text_length_stats'] = {
            'mean': int(self.df['text_length'].mean()),
            'median': int(self.df['text_length'].median()),
            'min': int(self.df['text_length'].min()),
            'max': int(self.df['text_length'].max())
        }
        logger.info("Initial statistics generated successfully.")
        return stats

    def _analyze_text_features(self) -> Dict[str, Any]:
        """Analyze text features and generate visualizations."""
        # Get all words from processed texts
        logger.info("Analyzing text features.")
        all_words = [word for text in self.df['processed_text'] for word in text]
        word_freq = Counter(all_words)
        logger.info("Text feature analysis completed.")
        vocab_stats = {
            'total_words': len(all_words),
            'unique_words': len(word_freq),
            'average_words_per_text': round(len(all_words) / len(self.df), 2),
            'most_common_words': dict(word_freq.most_common(20))
        }
        
        # Generate and save word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_words))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Text Data')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'images' / 'wordcloud.png')
        plt.close()
        logger.info(f"Word Cloud saved to: {self.output_dir}/'images'/'wordcloud.png'")
        return vocab_stats

    def _analyze_sentiment_distribution(self) -> Optional[Dict[str, Any]]:
        """Analyze sentiment distribution and generate visualization."""
        
        if not self.sentiment_column:
            logger.info("No sentiment column provided; skipping sentiment analysis.")
            return None
        logger.info("Analyzing sentiment distribution.")
        sentiment_stats = {
            'value_counts': self.df[self.sentiment_column].value_counts().to_dict(),
            'distribution_percentage': {k: round(v, 2) for k, v in 
                (self.df[self.sentiment_column].value_counts(normalize=True) * 100).to_dict().items()}
        }
        
        # Generate and save sentiment distribution plot
        plt.figure(figsize=(8, 6))
        sns.countplot(data=self.df, x=self.df[self.sentiment_column])
        plt.title('Sentiment Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'images' / 'sentiment_distribution.png')
        plt.close()
        logger.info("Sentiment distribution analysis completed.")
        logger.info(f"Sentiment distribution plot saved to: {self.output_dir} / 'images' / 'sentiment_distribution.png' ")
        return sentiment_stats
    
    def save_dataframe(self, filename: str = "processed_data.csv") -> str:
        """
        Save the processed DataFrame to the output directory as a CSV file.
        
        Parameters:
        -----------
        filename : str, optional
            The name of the output file. Default is "processed_data.csv".
        
        Returns:
        --------
        str
            Path to the saved CSV file.
        """
        file_path = self.output_dir / filename
        try:
            self.df.to_csv(file_path, index=False)
            logger.info(f"DataFrame successfully saved to {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Failed to save DataFrame: {e}")
            raise
        return file_path


    def generate_report(self) -> str:
        """
        Generate and save a comprehensive profile report.
        
        Returns:
        str: Path to the generated report directory
        """
        try:
            processed_file_path = self.save_dataframe()
            
            # Log the processed data as an artifact
            self.mlflow_tracker.log_artifact(processed_file_path, "processed_data")
            
            
            logger.info("Generating profile report.")
            # Generate report components
            report = {
                'timestamp': self.timestamp,
                'dataset_info': {
                    'text_column': self.text_column,
                    'sentiment_column': self.sentiment_column
                },
                'initial_statistics': self._get_initial_statistics(),
                'text_analysis': self._analyze_text_features()
            }
            
            if self.sentiment_column:
                report['sentiment_analysis'] = self._analyze_sentiment_distribution()
            
            # Save report as JSON
            report_path = self.output_dir / 'report.json'
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=4, ensure_ascii=False)
            logger.info(f"Report generated successfully at {report_path}.")
            
            # Log the report as an artifact
            self.mlflow_tracker.log_artifact(str(report_path), "report")
            
            # Generate a README with file descriptions
            readme_content = f"""Data Profile Report
            Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

            Files in this directory:
            1. report.json - Complete analysis report in JSON format
            2. images/wordcloud.png - Word cloud visualization of text data"""

            if self.sentiment_column:
                readme_content += "\n3. images/sentiment_distribution.png - Distribution of sentiment labels"
                
            readme_path = self.output_dir / 'README.txt'

            with open(readme_path, 'w') as f:
                f.write(readme_content)
            logger.info("README file created successfully.")
            
            # Log the README as an artifact
            self.mlflow_tracker.log_artifact(str(readme_path), "readme")
            
            # Log visualizations as artifacts
            self.mlflow_tracker.log_artifact(str(self.output_dir / 'images' / 'wordcloud.png'), "images")
            if self.sentiment_column:
                self.mlflow_tracker.log_artifact(str(self.output_dir / 'images' / 'sentiment_distribution.png'), "images")
            
            # Log key statistics as metrics
            initial_stats = report['initial_statistics']
            self.mlflow_tracker.log_metrics({
                'total_rows': initial_stats['total_rows'],
                'total_columns': initial_stats['total_columns'],
                'mean_text_length': initial_stats['text_length_stats']['mean'],
                'median_text_length': initial_stats['text_length_stats']['median']
            })
            
            text_analysis = report['text_analysis']
            self.mlflow_tracker.log_metrics({
                'total_words': text_analysis['total_words'],
                'unique_words': text_analysis['unique_words'],
                'average_words_per_text': text_analysis['average_words_per_text']
            })
            
            if self.sentiment_column:
                sentiment_analysis = report['sentiment_analysis']
                for sentiment, count in sentiment_analysis['value_counts'].items():
                    self.mlflow_tracker.log_metrics({f"sentiment_count_{sentiment}": count})
            return str(self.output_dir)
        finally:
            # End the MLflow run
            self.mlflow_tracker.end_run()