import sys
import pytest
from textwrap import dedent

import pandas as pd

from src.enities.split_params import SplittingParams

from src.data.make_dataset import (
    read_data,
    split_train_val_data,
)

DATASET_TINY_STR = dedent("""\
age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target
63,1,3,145,233,1,0,150,0,2.3,0,0,1,1
37,1,2,130,250,0,1,187,0,3.5,0,0,2,1
41,0,1,130,204,0,0,172,0,1.4,2,0,2,1
56,1,1,120,236,0,1,178,0,0.8,2,0,2,1
57,0,0,120,354,0,1,163,1,0.6,2,0,2,1
57,1,0,140,192,0,1,148,0,0.4,1,0,1,1
56,0,1,140,294,0,0,153,0,1.3,1,0,2,1
44,1,1,120,263,0,1,173,0,0,2,0,3,1
52,1,2,172,199,1,1,162,0,0.5,2,0,3,1
57,1,2,150,168,0,1,174,0,1.6,2,0,2,1
""")


def test_class_splitting_params():
    split_params = SplittingParams()
    assert split_params.val_size == 0.2
    assert split_params.random_state == 42


@pytest.fixture()
def tiny_dataset_fio(tmpdir):
    fio = tmpdir.join("dataset_tiny.csv")
    fio.write(DATASET_TINY_STR, "w")
    return fio


def test_can_read_data(tiny_dataset_fio, caplog):
    caplog.set_level("DEBUG")
    with caplog.at_level("DEBUG"):
        df_loaded = read_data(tiny_dataset_fio)
        data_local = {
            "age": [63, 37, 41, 56, 57, 57, 56, 44, 52, 57, ],
            "sex": [1, 1, 0, 1, 0, 1, 0, 1, 1, 1, ],
            "cp": [3, 2, 1, 1, 0, 0, 1, 1, 2, 2, ],
            "trestbps": [145, 130, 130, 120, 120, 140, 140, 120, 172, 150, ],
            "chol": [233, 250, 204, 236, 354, 192, 294, 263, 199, 168, ],
            "fbs": [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, ],
            "restecg": [0, 1, 0, 1, 1, 1, 0, 1, 1, 1, ],
            "thalach": [150, 187, 172, 178, 163, 148, 153, 173, 162, 174, ],
            "exang": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, ],
            "oldpeak": [2.3, 3.5, 1.4, 0.8, 0.6, 0.4, 1.3, 0, 0.5, 1.6, ],
            "slope": [0, 0, 2, 2, 2, 1, 1, 2, 2, 2, ],
            "ca": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            "thal": [1, 2, 2, 2, 2, 1, 2, 3, 3, 2, ],
            "target": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ]
        }
        df_local = pd.DataFrame(data=data_local)
        assert df_local.equals(df_loaded)


def test_split_train_val_data(tiny_dataset_fio, caplog):
    caplog.set_level("DEBUG")
    with caplog.at_level("DEBUG"):
        df_loaded = read_data(tiny_dataset_fio)
        split_params = SplittingParams()
        train_data, val_data = split_train_val_data(df_loaded, split_params)
        assert val_data.shape[0] == round(
            df_loaded.shape[0] * split_params.val_size)
