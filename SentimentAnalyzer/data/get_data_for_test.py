import pandas as pd
from steps.prepare_data import DataProcessing, DataPreprocessStrategy
import logging

def get_data_for_test(n_sample: int = 100):
    try:
        df = pd.read_csv("data/reviews_training.csv", encoding='latin-1', usecols=[ 0, 1 ])
        df = df.sample(100)
        processed_data = DataProcessing(data=df, strategy=DataPreprocessStrategy()).handle_data()
        processed_data.drop(['sentiment'], axis=1, inplace=True)
        result = processed_data.to_json(orient="split")
        return result
    except Exception as e:
        logging.error(e)
        raise e

