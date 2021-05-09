
import sys
import logging
import logging.config
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import yaml

import pandas as pd

from src.data.make_dataset import read_data, split_train_val_data
from src.enities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)

from src.features.build_features import extract_target, FeaturesTransformer
from src.models.model_fit_predict import (
    train_model,
    serialize_model,
    predict_model,
    evaluate_model,
)

APPLICATION_NAME = "train_pipeline"
DEFAULT_DATASET_PATH = "ml_project/data/raw/heart.csv"
DEFAULT_CONFIG_PATH = 'ml_project/configs/train_config.yaml'
DEFAULT_LOGGING_CONFIG_FILEPATH = "ml_project/configs/logging.conf.yml"

logger = logging.getLogger(APPLICATION_NAME)


def callback_parser(arguments):
    """ Calling process function wth received arguments """

    return train_pipeline(arguments.dataset_filepath,
                          arguments.config_filepath, )


def prepare_val_features_for_predict(
    train_features: pd.DataFrame,
    val_features: pd.DataFrame
):
    # small hack to work with categories
    train_features, val_features = train_features.align(
        val_features, join="left", axis=1
    )
    val_features = val_features.fillna(0)
    return val_features


def train_pipeline(dataset_filepath, config_filepath):
    """
    Process train pipeline
    """

    # reading config, loading dataset
    logger.info(
        f"Reading train pipeline params from {config_filepath}")

    train_params = read_training_pipeline_params(config_filepath)

    logger.info(f"Start train pipeline with params: {train_params}")
    data = read_data(dataset_filepath)
    logger.info(f"Data.shape is {data.shape}")
    train_df, val_df = split_train_val_data(
        data, train_params.splitting_params
    )
    logger.info(f"Train_df.shape is {train_df.shape}")
    logger.info(f"Val_df.shape is {val_df.shape}")

    # build features
    features = FeaturesTransformer(train_params.feature_params)
    features.fit(train_df)

    train_features = features.transform(train_df)
    train_target = extract_target(train_df, train_params.feature_params)
    logger.info(f"Train_features.shape is {train_features.shape}")

    val_features = features.transform(val_df)
    val_target = extract_target(val_df, train_params.feature_params)
    val_features_prepared = prepare_val_features_for_predict(
        train_features, val_features
    )
    logger.info(f"Val_features.shape is {val_features_prepared.shape}")

    # train model
    model = train_model(
        train_features,
        train_target,
        train_params.train_params
    )

    # get predictions
    predicts = predict_model(model, val_features_prepared)

    # evaluate prediction
    metrics = evaluate_model(predicts, val_target)

    if train_params.metric_path is not None:
        with open(train_params.metric_path, "w") as metric_file:
            json.dump(metrics, metric_file)
        logger.info(f"Metrics is {metrics}")

    if train_params.output_model_path is not None:
        path_to_model = serialize_model(
            model, train_params.output_model_path)
        logger.info(f"Model is serialized to {path_to_model}")
    logger.info(f"Training is completed.")
    return path_to_model, metrics


def setup_parser(parser):
    """ Setting up parser """

    parser.add_argument(
        "-d", "--dataset",
        default=DEFAULT_DATASET_PATH,
        metavar='DATASET',
        dest="dataset_filepath",
        help="path to dataset to load, default path is %(default)s",
    )

    parser.add_argument(
        "-c", "--config",
        default=DEFAULT_CONFIG_PATH,
        metavar='CONFIG',
        dest="config_filepath",
        help="path to config file to load, default path is %(default)s",
    )

    parser.set_defaults(callback=callback_parser)


def setup_logging():
    """ Setting up logging configuration """
    with open(DEFAULT_LOGGING_CONFIG_FILEPATH) as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))


def main():
    """ Main function to setup the parser and run """

    setup_logging()

    parser = ArgumentParser(
        prog="train_pipeline",
        description="Script to train model on heart disease data https://www.kaggle.com/ronitf/heart-disease-uci/code",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    setup_parser(parser)
    arguments = parser.parse_args()

    arguments.callback(arguments)


if __name__ == "__main__":
    main()
