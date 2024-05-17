import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.DiamondPricePrediction.exception import CustomException
from src.DiamondPricePrediction.logger import logging
from dataclasses import dataclass
from pathlib import Path
import os
import sys


class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Started Data Ingestion")

        try:
            data = pd.read_csv(
                Path(os.path.join("notebooks/data", "Diamonds_Prices.csv")))
            logging.info("Read dataset as a DataFrame")

            os.makedirs(os.path.dirname(os.path.join(
                self.ingestion_config.raw_data_path)), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Saved raw data in artifacts directory")

            logging.info("Here, Perform train test split")
            train_data, test_data = train_test_split(data, test_size=0.25)
            logging.info("train test split completed")

            data.to_csv(self.ingestion_config.train_data_path, index=False)
            data.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Data ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("exception occurred during the data ingestion stage")
            raise CustomException(e, sys)
