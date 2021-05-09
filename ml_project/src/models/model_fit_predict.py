import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.enities.train_params import TrainingParams

SklearnRegressionModel = Union[
    LogisticRegression,
    RandomForestClassifier,
    KNeighborsClassifier,
]


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> SklearnRegressionModel:

    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            criterion="gini",
            n_estimators=250,
            random_state=train_params.random_state,
        )

    elif train_params.model_type == "KNeighborsClassifier":
        model = KNeighborsClassifier(
            n_neighbors=10,
        )

    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression(
            C=1.0,
            solver="liblinear",
            random_state=train_params.random_state,
        )

    else:
        raise NotImplementedError()

    model.fit(features, target)
    return model


def predict_model(
    model: SklearnRegressionModel, features: pd.DataFrame
) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_model(
    predicts: np.ndarray, target: pd.Series
) -> Dict[str, float]:

    return {
        "accuracy": accuracy_score(target, predicts),
    }


def serialize_model(model: SklearnRegressionModel, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output


def deserialize_model(path: str) -> SklearnRegressionModel:
    with open(path, "rb") as f:
        return pickle.load(f)
