from sklearn.base import BaseEstimator, TransformerMixin
from TaxiFareModel.utils import haversine_vectorized
import pandas as pd


class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """Extract the day of week (dow), the hour, the month and the year from a
    time column."""
    def __init__(self, time_column, time_zone_name='America/New_York'):
        self.time_column = time_column
        self.time_zone_name = time_zone_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'"""
        X2=X.copy()
        X2.index = pd.to_datetime(X[self.time_column])
        X2.index = X2.index.tz_convert(self.time_zone_name)
        X2["dow"] = X2.index.weekday
        X2["hour"] = X2.index.hour
        X2["month"] = X2.index.month
        X2["year"] = X2.index.year
        X2 = X2[['dow', 'hour', 'month', 'year']]
        return X2


class DistanceTransformer(BaseEstimator, TransformerMixin):
    """Compute the haversine distance between two GPS points."""

    def __init__(self,
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude"):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Returns a copy of the DataFrame X with only one column: 'distance'"""
        X2=X.copy()
        X2['distance'] = haversine_vectorized(X2, self.start_lat,self.start_lon,self.end_lat, self.end_lon)
        return pd.DataFrame(X2['distance'])
