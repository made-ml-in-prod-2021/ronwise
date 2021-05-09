from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline, make_union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from src.enities.feature_params import FeatureParams


class FeaturesTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, params: FeatureParams):

        self.numerical_features = params.numerical_features
        self.categorical_features = params.categorical_features
        self.categorical_pipeline = make_pipeline(
            SimpleImputer(missing_values=np.nan, strategy="most_frequent"),
            OneHotEncoder(sparse=False, drop='first'),
        )

        self.numerical_pipeline = make_pipeline(
            MinMaxScaler(),
        )

    def fit(self, X, y=None):
        self.categorical_pipeline.fit(X[self.categorical_features].values)
        self.numerical_pipeline.fit(X[self.numerical_features].values)
        return self

    def transform(self, X, y=None):
        categorical_features_transformed = self.categorical_pipeline.transform(
            X[self.categorical_features].values)
        numerical_features_transformed = self.numerical_pipeline.transform(
            X[self.numerical_features].values)
        df = np.hstack([numerical_features_transformed,
                       categorical_features_transformed])

        return pd.DataFrame(
            data=df, columns=self.numerical_features +
            ['x_' + str(x)
             for x in range(categorical_features_transformed.shape[1])]
        )


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    target = df[params.target_col]
    return target
