import logging

import mlflow
import pandas as pd

from model.model_eval import MSE, RMSE, R2Score
from model.LinearVectorizeModel import LinearVectorizeModel
from zenml.client import Client
from typing_extensions import Annotated
from typing import Tuple
from zenml import step

experiment_tracker = Client().active_stack.experiment_tracker


@step(name="eval_step", experiment_tracker=experiment_tracker.name, enable_cache=False)
def evaluation(
        model: LinearVectorizeModel, val: pd.DataFrame
) -> Tuple[Annotated[float, "r2_score"], Annotated[float, "rmse"]]:
    """
    Args:
        model: RegressorMixin
        val: pd.DataFrame
    Returns:
        r2_score: float
        rmse: float
    """

    try:
        X_test, y_test = val.text, val.sentiment
        logging.info("Running Evaluation...")

        y_test_bow = model.le.transform(y_test)
        predictions = model.predict(X_test)

        # Using the MSE class for mean squared error calculation
        mse = MSE().calculate_score(y_test_bow, predictions)
        mlflow.log_metric("mse", mse)

        # Using the R2Score class for R2 score calculation
        r2_score = R2Score().calculate_score(y_test_bow, predictions)
        mlflow.log_metric("r2_score", r2_score)

        # Using the RMSE class for root mean squared error calculation
        rmse = RMSE().calculate_score(y_test_bow, predictions)
        mlflow.log_metric("rmse", rmse)

        logging.info("Model Evaluated Successfully...")

        return r2_score, rmse

    except Exception as e:
        logging.error(e)
        raise e
