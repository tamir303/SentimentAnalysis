import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Union

class LinearVectorizeModel(BaseEstimator, RegressorMixin):
    def __init__(self, model, vectorizer, le):
        self.model = model
        self.vectorizer = vectorizer
        self.le = le

    def predict(self, X):
        def converter(data):
            if isinstance(data, np.ndarray):
                return pd.Series(data)
            else:
                return data
        try:
            # Check if vectorizer is fitted
            if self.vectorizer is not None and hasattr(self.vectorizer, 'transform'):
                vect_X = self.vectorizer.transform(converter(X))
                predictions = self.model.predict(vect_X)
                return predictions
            else:
                raise ValueError("Vectorizer not fitted.")

        except AttributeError as attr_error:
            print(f"AttributeError in prediction: {attr_error}")
            print(f"Problematic input: {X}")
            raise attr_error

        except ValueError as value_error:
            print(f"ValueError in prediction: {value_error}")
            raise value_error

        except Exception as e:
            print(f"Error in prediction: {e}")
            raise e

