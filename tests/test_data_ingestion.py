
import pytest
import pandas as pd
from pathlib import Path
from src.SentiScope.components.data_ingestion import DataIngestion
from src.SentiScope.config.configuration import ConfigurationManager
from src.SentiScope.entity import DataIngestionConfig



class TestDataIngestion:
    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration for testing."""
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        return data_ingestion_config
    
    @pytest.fixture
    def data_ingestion(self, sample_config):
        """Initialize data ingestion with sample configuration."""
        return DataIngestion(config=sample_config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'text': ['This is positive', 'This is negative', 'This is neutral'],
            'sentiment': ['positive', 'negative', 'neutral']
        })
    
    def test_data_ingestion_initialization(self, data_ingestion):
        """Test if DataIngestion is initialized correctly."""
        assert isinstance(data_ingestion, DataIngestion)
        assert data_ingestion.config is not None
    