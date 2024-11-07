import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import Utility
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation


@dataclass
class ModuleTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModuleTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Splitting train and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            models = {
                "Random Forrest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XG Boost Regressor": XGBRegressor(),
                "Cat Boost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            model_report: dict = Utility.evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
            )

            logging.info(f"model_report : {model_report}")

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            logging.info(f"best_model_score : {best_model_score}")

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            logging.info(f"best_model_name : {best_model_name}")

            best_model = models[best_model_name]
            logging.info(f"best_model : {best_model}")

            if best_model_name < "0.6":
                raise CustomException("No best model found!")

            logging.info("best_model found on training and testing dataset")

            Utility.save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)

            r2score = r2_score(y_test, predicted)
            return r2score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    data_ingestion_obj = DataIngestion()
    train_data_path, test_data_path = data_ingestion_obj.initiate_data_ingestion()

    data_transformation_obj = DataTransformation()
    train_arr, test_arr, preprocessor_obj_file_path = (
        data_transformation_obj.initiate_data_transformation(
            train_data_path, test_data_path
        )
    )

    model_trainer_obj = ModelTrainer()
    r2_score = model_trainer_obj.initiate_model_trainer(
        train_arr, test_arr, preprocessor_obj_file_path
    )

    print("r2_score : ", r2_score)

    print("Hello")
