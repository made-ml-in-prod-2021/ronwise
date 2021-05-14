# ml_project


## homework 1 for MADE course on ML in production 2021


## Setting up environment:


Requirements: `Python 3.7, conda`

Installation: `pip install .`

## Tests:


python -m pytest . -v --cov --cov-fail-under=80

## EDA report


Jupyter notebook: 
`notebooks/1_0_nr_initial_data_exploration.ipynb`

## Training


* config 1: 

    `python ml_project/train_pipeline.py -d ml_project/data/raw/heart.csv -c ml_project/configs/config_1/train_config.yaml`

* config 2:

    `python ml_project/train_pipeline.py -d ml_project/data/raw/heart.csv -c ml_project/configs/config_2/train_config.yaml`

## Prediction

* config 1: 

    `python ml_project/predict_pipeline.py -d ml_project/data/raw/heart.csv -c ml_project/configs/config_1/train_config.yaml`

* config 2:

    `python ml_project/predict_pipeline.py -d ml_project/data/raw/heart.csv -c ml_project/configs/config_2/train_config.yaml`