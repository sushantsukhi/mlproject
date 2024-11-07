import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score


class Utility:

    def save_object(file_path, obj):
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            with open(file_path, "wb") as file_obj:
                dill.dump(obj, file_obj)
            logging.info("File dumped..")
        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_model(X_train, y_train, X_test, y_test, models):
        try:
            report = {}

            for i in range(len(list(models))):
                model = list(models.values())[i]

                model.fit(X_train, y_train)  # Train Model

                y_train_pred = model.predict(X_train)
                train_model_score = r2_score(y_train, y_train_pred)

                y_test_pred = model.predict(X_test)
                test_model_score = r2_score(y_test, y_test_pred)

                report[list(models.keys())[i]] = train_model_score

            return report
        except:
            pass
