import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import (
    DataTransformation,
    DataTransformationConfig,
)


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion method or component")

        try:
            df = pd.read_csv("notebook\data\stud.csv")
            logging.info("Read the dataset as dataframe")

            ## First - Create Base directory
            os.makedirs(
                os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True
            )

            ## Second - Put all data to Raw data CSV and store it in raw.csv
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated!")
            ## Third - Create train test split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            ## Fourth - Put train data in tran csv and store it in train.csv
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )

            ## Fifth - Put test data in tran csv and store it in test.csv
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            logging.info("Ingestion of data is completed!")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_ingestion_obj = DataIngestion()
    train_data_path, test_data_path = data_ingestion_obj.initiate_data_ingestion()

    data_transforation_obj = DataTransformation()
    data_transforation_obj.initiate_data_transformation(train_data_path, test_data_path)
