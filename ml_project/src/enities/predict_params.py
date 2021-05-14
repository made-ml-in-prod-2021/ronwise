from dataclasses import dataclass, field


@dataclass()
class PredictParams:
    model_type: str = field(default="RandomForestRegressor")
    random_state: int = field(default=42)
