import logging

import os
import pickle

import mlflow.sklearn
from zenml import step
import pandas as pd
from model.model_dev import XGBClassifierModel, LogisticRegressionModel, HyperparameterTuner
from typing_extensions import Annotated
from steps.config import ModelConfig
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from model.LinearVectorizeModel import LinearVectorizeModel
from zenml.client import Client
import mlflow

experiment_tracker = Client().active_stack.experiment_tracker

@step(name="train_step", experiment_tracker=experiment_tracker.name, enable_cache=False)
def train_model(
        X_train: pd.Series,
        X_test: pd.Series,
        y_train: pd.Series,
        y_test: pd.Series,
        vectorizer: CountVectorizer,
        le: LabelEncoder,
        config: ModelConfig,
) -> Annotated[LinearVectorizeModel, "Model"]:
    """
    Args:
        X_train: pd.Series
        X_test: pd.Series
        y_train: pd.Series
        y_test: pd.Series
        vectorizer: CountVectorizer
        le: LabelEncoder
        config: ModelConfig
    Returns:
        model: LinearVectorizeModel
    """

    try:
        if os.path.exists("save_model/model.pkl"):
            with open("save_model/model.pkl", "rb") as file:
                lvm = pickle.load(file)
            logging.info("Model Loaded Successfully...")

        else:
            def selectModel(name: str):
                selector = {
                    "log_reg": LogisticRegressionModel(),
                    "xgb": XGBClassifierModel()
                }

                return selector.get(name)

            logging.info("Training Model...")

            # Transform documents using vectorizer
            X_train_bow = vectorizer.transform(X_train)
            X_test_bow = vectorizer.transform(X_test)

            # Encode labels using label encoder
            y_train_bow = le.transform(y_train)
            y_test_bow = le.transform(y_test)

            model = selectModel(config.model_name)
            tuner = HyperparameterTuner(model, X_train_bow, X_test_bow, y_train_bow, y_test_bow)

            if model:
                if config.optimize:
                    logging.info("Running Fine-Tune Optimization")
                    best_params = tuner.optimize()
                else:
                    best_params = config.params_dict

                logging.info(f"Training Model {config.model_name} With Params {best_params}")
                trained_model = model.train(X_train_bow, y_train_bow, **best_params)
                lvm = LinearVectorizeModel(trained_model, vectorizer, le)
                logging.info("Model Trained Successfully...")
            else:
                raise ValueError(f"Model {config.model_name} is not supported")

        mlflow.sklearn.log_model(lvm, "model")
        return lvm

    except Exception as e:
        logging.error(e)
        raise e
