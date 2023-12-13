import logging

import pandas as pd
from zenml import step
from typing import Tuple
from typing_extensions import Annotated

class IngestData:
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def __init__(self) -> None:
        """Initialize the data ingestion class."""
        pass

    @staticmethod
    def get_data( ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train = pd.read_csv("data/reviews_training.csv", encoding='latin-1', usecols=[0, 1])
        val = pd.read_csv("data/reviews_validation.csv", encoding='latin-1', usecols=[0, 1])
        return train, val

@step(enable_cache=True, name="ingest_step")
def ingest_data() -> Tuple[Annotated[pd.DataFrame, "df_training"], Annotated[pd.DataFrame, "df_validation"]]:
    """
    :arg
        None
    :return:
        df: pd.DataFrame
    """
    try:
        train, val = IngestData.get_data()
        logging.info("Data Ingested Data Successfully...")
        return train, val
    except Exception as e:
        logging.error(f"Error On Data Ingest {e}")
        raise e

