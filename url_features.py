import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin

class URLFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert input to pandas Series if it's a list
        if isinstance(X, list):
            X = pd.Series(X)
        return pd.DataFrame({
            'url_length': X.apply(lambda x: len(x)),
            'special_char_count': X.apply(lambda x: len(re.findall(r'[-?&=]', x))),
            'num_subdomains': X.apply(lambda x: len(re.findall(r'\.', x))),
            'has_ip': X.apply(lambda x: 1 if re.search(r'\d+\.\d+\.\d+\.\d+', x) else 0),
            'suspicious_words': X.apply(lambda x: 1 if re.search(r'free|login|offer|click', x.lower()) else 0)
        }, index=X.index)
