from pathlib import Path
from typing import List, Literal
from pydantic import BaseModel


class EarlyStoppingConfig(BaseModel):
    patience: int
    start_from_epoch: int
    delta_retrieval: float
    delta_rating: float
    retrieval_metric: Literal[
        "val_factorized_top_k/top_1_categorical_accuracy",
        "val_factorized_top_k/top_5_categorical_accuracy",
        "val_factorized_top_k/top_10_categorical_accuracy",
        "val_factorized_top_k/top_50_categorical_accuracy",
        "val_factorized_top_k/top_100_categorical_accuracy"
    ]
    rating_metric: Literal["val_root_mean_squared_error"]


class LRScheduleConfig(BaseModel):
    monitor: str
    factor: float
    patience: int
    mode: Literal["min", "max"]
    min_delta: float
    cooldown: int
    min_lr: float


class CallbacksConfig(BaseModel):
    early_stopping: EarlyStoppingConfig
    lr_schedule: LRScheduleConfig


class TowerConfig(BaseModel):
    query: List[str]
    candidate: List[str]


class ModelConfig(BaseModel):
    max_epochs: int
    initial_learning_rate: float
    emb_weight: float
    query_layers: List[int]
    candidate_layers: List[int]
    rating_layers: List[int]
    dropout: float
    cross_layer: bool
    optimizer: Literal["Adagrad", "Adam"]


class Params(BaseModel):
    logs_path: str
    target: str
    time_feature: str
    batch_size: int
    seed: int
    tower: TowerConfig
    callbacks: CallbacksConfig
    model: ModelConfig


class HyperparametersConfig(BaseModel):
    initial_learning_rate: List[float]
    emb_weight: List[int]
    dropout: List[float]
    cross_layer: List[bool]
    optimizer: List[Literal["Adagrad", "Adam"]]
    query_layers: List[List[int]]
    candidate_layers: List[List[int]]
    output_layer: List[int]
    rating_layers: List[List[int]]


class OptunaConfig(BaseModel):
    study_name: str
    study_path: str
    retrieval_metric: Literal[
        "val_factorized_top_k/top_1_categorical_accuracy",
        "val_factorized_top_k/top_5_categorical_accuracy",
        "val_factorized_top_k/top_10_categorical_accuracy",
        "val_factorized_top_k/top_50_categorical_accuracy",
        "val_factorized_top_k/top_100_categorical_accuracy"
    ]
    hyperparameters: HyperparametersConfig


# Create dafault parameters file
params = {
    "logs_path": "",
    "target": "score",
    "time_feature": "time",
    "batch_size": 128,
    "seed": 42,
    "tower": {
        "query": [
            "query_feature1",
            "query_feature2",
            "target"
        ],
        "candidate": [
            "product_feature1",
            "product_feature2",
            "product_feature3"
        ]
    },
    "callbacks": {
        "early_stopping": {
            "patience": 3,
            "start_from_epoch": 5,
            "delta_retrieval": 0.01,
            "delta_rating": 0.01,
            # This can be changed among retrieval metrics
            "retrieval_metric": "val_factorized_top_k/top_50_categorical_accuracy",
            # This is fixed
            "rating_metric": "val_root_mean_squared_error"
        },
        "lr_schedule": {
            "monitor": "val_total_loss",
            "factor": 0.1,
            "patience": 1,
            "mode": "min",
            "min_delta": 10,
            "cooldown": 3,
            "min_lr": 1e-5
        }
    },
    "model": {
        "max_epochs": 50,
        "initial_learning_rate": 0.1,
        # embedding weight shared among all features where emb_size = (np.log2(input_dim) + 1) * emb_weigh
        "emb_weight": 16,
        # Units at the last layer have to be equal
        "query_layers": [128, 64, 32],
        "candidate_layers": [128, 128, 32],
        # Units at the last layer has to be equal to 1 for regression problem
        "rating_layers": [256, 128, 1],
        "dropout": 0.3,
        "cross_layer": False,
        "optimizer": "Adagrad"
    }
}

# Create optuna config file
optuna_config = {
    "study_name": "recommendation_engine",
    "study_path": "optuna/hp.log",
    "retrieval_metric": "val_factorized_top_k/top_50_categorical_accuracy",
    "hyperparameters": {
        "initial_learning_rate": [0.1, 0.01],
        "emb_weight": [4, 8, 12, 16],
        "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
        "cross_layer": [True, False],
        "optimizer": ["Adagrad", "Adam"],
        "query_layers": [
            [32, 64, 128],
            [32, 64, 128]
        ],
        "candidate_layers": [
            [32, 64, 128],
            [32, 64, 128]
        ],
        "output_layer": [8, 16, 32, 64],
        "rating_layers": [
            [8, 16, 32, 64, 128, 256],
            [8, 16, 32, 64, 128, 256]
        ]
    }
}


def create_params(path: str = None) -> None:
    """
    Create default parameters file named `params` in JSON format.
    This is the input and the backbone of the `rec_engine` package to work with.
    Set the parameters required for running the functions with `params` argument.

    Features below each `tower` are defined as lists of strings, where preprocessing step is given by a prefix separated by "-".
    Prefixes are:
    - `cat-` for string categorical
    - `int-` for integer categorical
    - `seq-` for sequential string categorical
    - `text-` for text features
    - `disc-` for discretization transformation of continuous numerical features
    - `norm-` for normalization transformation of continuous numerical features

    **NOTE: It is necessary to apply a transformation to be considered by the model, if not, it will be ignored.
    This is true for all features except the target feature, which is expected to be a rating value so it is not transformed.**

    Layers definition is given by a list of integers, where each integer is the number of units at each layer in order.


    Parameters
    ---
    path: str
        Path to the directory where the parameters file will be saved. Default to None
    """
    path = Path.cwd() if not path else Path(path)
    # Save default parameters
    with open(path / "params.json", "w") as f:
        f.write(Params(**params).model_dump_json(indent=4))

def create_optuna_config(path: str = None) -> None:
    """
    Create the optuna configuration file named `opt_config` in JSON format.
    It is needed for the Optuna optimization pipeline. Set the hyperparameters range for optimization.
    Each list indicates the possible values for each hyperparameter and Optuna will pick up one of them by chance.
    In layers definition, each list indicates the different possible units at each deep layer in the given order.
    It will be picked up to the number of layers by random, so a model can have equal or less deep layers than the ones provided.
    `output_layers` represent the units at the output layer in query and candidate tower. They must have the same dimension.

    To deactivate a parameter, only provide the actual value to be used in all models. Whereas layers definition accept empty lists.

    Parameters:
    ---
    path: str
        Path to the directory where the optuna configuration file will be saved. Default to None
    """
    path = Path.cwd() if not path else Path(path)
    with open(path / "opt_config.json", "w") as f:
        f.write(OptunaConfig(**optuna_config).model_dump_json(indent=4))
