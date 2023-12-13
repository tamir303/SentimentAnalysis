import logging
from typing import Union, Tuple
from abc import ABC, abstractmethod

import re
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    Abstract Class defining strategy for handling data
    """

    @abstractmethod
    def handle_data( self, data: pd.DataFrame ) -> Union[pd.DataFrame, Tuple[pd.Series, pd.Series, pd.Series, pd.Series]]:
        pass


class DataPreprocessStrategy(DataStrategy):
    """
    Data preprocessing strategy which preprocesses the data.
    """

    def handle_data( self, data: pd.DataFrame ) -> pd.DataFrame:
        try:
            data.columns = [ 'sentiment', 'text' ]
            data.dropna(inplace=True)
            data[ "text" ] = data.text.str.lower()  # lowercase
            data[ "text" ] = [ str(txt) for txt in data.text ]  # convert to str
            data[ "text" ] = data.text.apply(lambda x: re.sub('[^A-Za-z0-9 ]+', '', x))  # regex
            logging.info("Data Pre-Processing Successful")

            return data

        except Exception as e:
            logging.error(f"Error On DataPreprocessStrategy {e}")
            raise e

class DataSplitStrategy(DataStrategy):
    """
    Data dividing strategy which divides the data into train and test data.
    """

    def handle_data( self, data: pd.DataFrame ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Divides the data into train and test data.
        """

        try:
            # Extracting features (X) and target variable (y)
            X = data.text
            y = data.sentiment

            # Splitting the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            logging.info("Data Split Successful")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error(f"Error On DataSplitStrategy {e}")
            raise e


class DataProcessing:
    """
    Data processing class which preprocesses the data and divides it into train and test data.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        """Initializes the DataCleaning class with a specific strategy."""

        self.df = data
        self.strategy = strategy

    def handle_data( self ) -> Union[pd.DataFrame, Tuple[pd.Series, pd.Series, pd.Series, pd.Series]]:
        """Handle data based on the provided strategy"""

        return self.strategy.handle_data(self.df)
