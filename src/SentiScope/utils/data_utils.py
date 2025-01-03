
import pandas as pd
import requests
import zipfile
import os
from SentiScope.logging import logger

def download_data(url, filename):
  """
  Downloads data from the given URL and saves it to the specified filename.

  Args:
    url: The URL of the data to download.
    filename: The filename to save the downloaded data to.
  """
  try:
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes

    with open(filename, 'wb') as f:
      f.write(response.content)
    logger.info(f"Data downloaded successfully to {filename}")

  except requests.exceptions.RequestException as e:
    logger.error(f"Error downloading data: {e}")

def unzip_data(zip_file, extract_dir):
  """
  Unzips the given zip file to the specified directory.

  Args:
    zip_file: The path to the zip file.
    extract_dir: The directory to extract the contents of the zip file to.
  """
  try:
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
      zip_ref.extractall(extract_dir)
    logger.info(f"Data unzipped successfully to {extract_dir}")

  except zipfile.BadZipFile as e:
    logger.error(f"Error unzipping data: {e}")

def load_data_to_dataframe(filepath):
  """
  Loads data from the given filepath into a pandas DataFrame.

  Args:
    filepath: The path to the data file.

  Returns:
    A pandas DataFrame containing the loaded data.
  """
  try:
    df = pd.read_csv(filepath)
    return df

  except FileNotFoundError:
    logger.error(f"Error loading data: File not found at {filepath}")
    return None

  except pd.errors.ParserError as e:
    logger.erro(f"Error loading data: {e}")
    return None

