from typing import Tuple
from typing_extensions import Annotated
from zenml import step
import pandas as pd

import nltk
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

@step(enable_cache=True)
def get_encoders( X_train: pd.Series, y_train: pd.Series, ngram_span: tuple = (1, 4) ) \
        -> Tuple[ Annotated[CountVectorizer, "TextVectorizer"], Annotated[LabelEncoder, "LabelEncoder"] ]:
    """
    Args:
        X_train: pd.Series
        y_train: pd.Series
        ngram_span: tuple[int, int] = (1, 4)
    Returns:
        vectorizer: CountVectorizer
        le: LabelEncoder
    """
    nltk.download('punkt')
    vectorizer = CountVectorizer(tokenizer=word_tokenize, ngram_range=ngram_span)
    vectorizer.fit(X_train)
    le = LabelEncoder()
    le.fit(y_train)

    return vectorizer, le
