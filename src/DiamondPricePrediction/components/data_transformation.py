import os
import sys
import pandas as pd
import numpy as np

from src.DiamondPricePrediction.exception import CustomException
from src.DiamondPricePrediction.logger import logging
from dataclasses import dataclass

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.DiamondPricePrediction.utils.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):

        try:
            logging.info("Data Transformation Initiated")

            # Define which features should be encoded and which should be scaled
            categorical_cols = ["cut", "color", "clarity"]
            numerical_cols = ["carat", "depth", "table", "x", "y", "z"]

            # Ranking for the categorical features(Ordinal)
            cut_categories = ["Ideal", "Premium", "Very Good", "Good", "Fair"]
            color_categories = ["D", "E", "F", "G", "H", "I", "J"]
            clarity_categories = ["IF", "VVS1", "VVS2",
                                  "VS1", "VS2", "SI1", "SI2", "I1"]

            logging.info("Pipeline Initiated")

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("ordinalencoder", OrdinalEncoder(
                        categories=[cut_categories, color_categories, clarity_categories]))
                ]
            )

            # Transform the columns
            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", num_pipeline, numerical_cols),
                    ("categorical_pipeline", cat_pipeline, categorical_cols)
                ]
            )

            return preprocessor

        except Exception as e:
            logging.info("exception occurred duringthe data ingestion stage")
            raise CustomException(e, sys)

    def initialize_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data complete")
            logging.info(
                f"Train DataFrame Head : \n{train_df.head().to_string()}")
            logging.info(

                f"Test DataFrame Head : \n{test_df.head().to_string()}")

            preprocessing_obj = self.get_data_transformation()

            # Dropping target and id features from train_df and test_df

            target_column_name = "price"
            drop_columns = [target_column_name, "Unnamed: 0"]

            input_feature_train_df = train_df.drop(
                columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(
                columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            # tranform the columns

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df)

            logging.info(
                "Applying preprocessing object on the training and test dataset")

            train_arr = np.c_[input_feature_train_arr,
                              np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,
                             np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessor Pickled file saved")

            return (
                train_arr,
                test_arr
            )

        except Exception as e:
            logging.info(
                "exception occurred during the data transformation stage")
            raise CustomException(e, sys)
