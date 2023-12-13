import logging
import pandas as pd
from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from model.processing_data import DataProcessing, DataSplitStrategy, DataPreprocessStrategy

@step(name="prepare_step", enable_cache=True)
def prepare_data(
        train: pd.DataFrame,
        val: pd.DataFrame
) -> Tuple[Annotated[pd.Series, "X_train"], Annotated[pd.Series, "X_test"], Annotated[pd.Series, "y_train"], Annotated[pd.Series, "y_test"], Annotated[pd.DataFrame, "processed_validation_df"]]:
    """
    Data preparing class which preprocesses the data and divides it into train and test data.

    Args:
        train: pd.DataFrame
        val: pd.DataFrame
    """
    try:
        process_strategy = DataPreprocessStrategy()
        split_strategy = DataSplitStrategy()

        processed_train = DataProcessing(data=train, strategy=process_strategy).handle_data()
        processed_val = DataProcessing(data=val, strategy=process_strategy).handle_data()
        X_train, X_test, y_train, y_test = DataProcessing(data=processed_train, strategy=split_strategy).handle_data()
        logging.info("Data Processed Successfully...")

        return X_train, X_test, y_train, y_test, processed_val

    except Exception as e:
        logging.error(e)
        raise e
