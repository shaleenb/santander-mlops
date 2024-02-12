import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, data, y=None):
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data_fe = data.copy()
        data_fe["sum"] = data[self.features].sum(axis=1)
        data_fe["min"] = data[self.features].min(axis=1)
        data_fe["max"] = data[self.features].max(axis=1)
        data_fe["mean"] = data[self.features].mean(axis=1)
        data_fe["std"] = data[self.features].std(axis=1)
        data_fe["skew"] = data[self.features].skew(axis=1)
        data_fe["kurtosis"] = data[self.features].kurtosis(axis=1)
        data_fe["median"] = data[self.features].median(axis=1)
        return data_fe
