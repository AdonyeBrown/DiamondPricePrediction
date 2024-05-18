import os
import sys
import pandas as pd
from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.exception import CustomException
from src.DiamondPricePrediction.components.data_ingestion import DataIngestion
from src.DiamondPricePrediction.components.data_transformation import DataTransformation
from src.DiamondPricePrediction.components.model_trainer import ModelTrainer


obj = DataIngestion()
train_data_path, test_data_path = obj.initiate_data_ingestion()


data_transformation = DataTransformation()
train_arr, test_arr = data_transformation.initialize_data_transformation(
    train_data_path, test_data_path)

model_trainer_obj = ModelTrainer()
model_trainer_obj.initate_model_training(train_arr, test_arr)