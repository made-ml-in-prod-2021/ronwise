from dataclasses import dataclass
from .split_params import SplittingParams
from .feature_params import FeatureParams
from .predict_params import PredictParams
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class PredictPipelineParams:
    input_data_path: str
    input_model_path: str
    output_predict_path: str
    splitting_params: SplittingParams
    predict_params: PredictParams
    feature_params: FeatureParams


PredictPipelineParamsSchema = class_schema(PredictPipelineParams)


def read_predict_pipeline_params(path: str) -> PredictPipelineParams:
    with open(path, "r") as input_stream:
        schema = PredictPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
