from pathlib import Path
from pydantic import validate_call, ConfigDict

import tensorflow as tf
import optuna
from rec_engine.config import OptunaConfig, Params


from .model import RecommenderEngineModel
from .train_functions import train_model


# Hyperparameter Tuning
## Objective Function
# Objective function to be maximized.
class Objective:
    """
    Optuna Objective function to optimize hyperparameters.

    Define the hyperparameters in the `opt_config` input dict.
    The others parameters (those that are not optimized) are set in `parameters` input dict.

    Metrics to be optimized:
    - Retrieval Metric defined by `opt_config["retrieval_metric"]` is maximized.
    - Ranking Metric set as `val_root_mean_squared_error` is minimized.

    Attributes
    ---
    opt_config : dict
        Optuna configuration file. It is created by `rec_engine.config.create_optuna_config()`.
    parameters : dict
        Dictionary of model parameters. It is created by `rec_engine.config.create_params()` function.
    train : tf.data.Dataset
        Preprocessed training dataset.
    val : tf.data.Dataset
        Preprocessed validation dataset.
    candidates : tf.data.Dataset
        Preprocessed candidates dataset.
    feature_dim : dict[str, int]
        Features with vocabulary size to be used as input dimension for the embedding layer.

    Methods
    ---
    __call__(trial: optuna.trial)
        Trigger the optimization process
    """
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        opt_config: OptunaConfig,
        parameters: Params,
        train: tf.data.Dataset,
        val: tf.data.Dataset,
        candidates: tf.data.Dataset,
        feature_dim: dict[str, int]
    ):
        self.opt_config = opt_config.model_dump()
        self.parameters = parameters.model_dump()
        self.train = train
        self.val = val
        self.candidates = candidates
        self.feature_dim = feature_dim
        self.logs_path = Path(self.parameters["logs_path"])

    def _deep_layers(self, trial, name, units: list[list]) -> list:
        """
        Helper function to suggest deep layers arquitectures.

        Parameters
        ----------
        trial : optuna.trial
            Optuna trial
        units : list
            List of units
        Returns
        -------
        list
            Deep layers arquitectures
        """
        # If there are no units at layers defined, return an empty list
        if len(units) < 1:
            return []
        # Ensure at least one deep layer to be added
        n_layers = trial.suggest_int(f"{name}_layers", 1, len(units))

        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_categorical(f"{name}_units_l{i}", units[i]))
        return layers

    def __call__(self, trial) -> tuple[float, float]:
        # Clear clutter from previous TensorFlow graphs.
        tf.keras.backend.clear_session()

        # Define parameters and suggest values to be tuned and optimized
        self.parameters["logs_path"] = str(self.logs_path / "run_{:02d}".format(trial.number))

        self.parameters["model"]["initial_learning_rate"] = trial.suggest_categorical(
            "initial_learning_rate",
            self.opt_config["hyperparameters"]["initial_learning_rate"]
        )

        self.parameters["model"]["emb_weight"] = trial.suggest_categorical(
            "emb_weight", self.opt_config["hyperparameters"]["emb_weight"]
        )
        output_layer = trial.suggest_categorical(
            "output_layer", self.opt_config["hyperparameters"]["output_layer"]
        )
        self.parameters["model"]["query_layers"] = self._deep_layers(
            trial, "query", self.opt_config["hyperparameters"]["query_layers"]
        )
        self.parameters["model"]["query_layers"].append(output_layer)
        self.parameters["model"]["candidate_layers"] = self._deep_layers(
            trial, "candidate", self.opt_config["hyperparameters"]["candidate_layers"]
        )
        self.parameters["model"]["candidate_layers"].append(output_layer)
        self.parameters["model"]["rating_layers"] = self._deep_layers(
            trial, "rating", self.opt_config["hyperparameters"]["rating_layers"]
        )
        self.parameters["model"]["rating_layers"].append(1) # Add 1 unit to regression output layer
        self.parameters["model"]["dropout"] = trial.suggest_categorical(
            "dropout", self.opt_config["hyperparameters"]["dropout"]
        )
        self.parameters["model"]["cross_layer"] = trial.suggest_categorical(
            "cross_layer", self.opt_config["hyperparameters"]["cross_layer"]
        )
        self.parameters["model"]["optimizer"] = trial.suggest_categorical(
            "optimizer", self.opt_config["hyperparameters"]["optimizer"]
        )

        # Create model instance for baseline
        model = RecommenderEngineModel(
            params=self.parameters,
            candidates=self.candidates,
            preprocessing=False,  # Disable preprocessing
            feature_dim=self.feature_dim,
        )

        fitted_model = train_model(
            model=model,
            train= self.train,
            val= self.val,
            params=self.parameters,
            logging=True,
            verbose=False,
        )

        # Get metrics
        trial_results = {
            m: v[-1]
            for m, v in fitted_model.history.history.items()
            if m.startswith("val")
        }

        # Save metrics as attributes for further analysis
        for m, v in trial_results.items():
            trial.set_user_attr(m, v)

        # Return the objective values
        return (
            trial_results[self.opt_config["retrieval_metric"]],
            trial_results["val_root_mean_squared_error"],
        )

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def optimize(
        n_trials: int,
        opt_config: OptunaConfig,
        params: Params,
        train: tf.data.Dataset,
        val: tf.data.Dataset,
        candidates: tf.data.Dataset,
        feature_dim: dict[str, int]
    ) -> None:
    """
    Optimization Process with Optuna.

    Metrics to be optimized:
    - Retrieval Metric defined by `opt_config["retrieval_metric"]` is maximized.
    - Ranking Metric set as `val_root_mean_squared_error` is minimized.

    Parameters
    ---
    n_trials : int
        Number of trials to be run.
    opt_config : dict
        Optuna configuration file. It is created by `rec_engine.config.create_optuna_config()`.
    params : dict
        Dictionary of model parameters. It is created by `rec_engine.config.create_params()` function.
    train : tf.data.Dataset
        Preprocessed training dataset.
    val : tf.data.Dataset
        Preprocessed validation dataset.
    candidates : tf.data.Dataset
        Preprocessed candidates dataset.
    feature_dim : dict[str, int]
        Features with vocabulary size to be used as input dimension for the embedding layer.
    """
    params = params.model_dump()
    opt_config = opt_config.model_dump()

    # Load storage
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(opt_config["study_path"])),
    )

    # Create a study
    study = optuna.create_study(
        storage=storage,
        study_name=opt_config["study_name"],
        directions=["maximize", "minimize"],
        load_if_exists=True,
    )

    # Perform optimization
    objective = Objective(
        opt_config=opt_config,
        parameters=params,
        train=train,
        val=val,
        candidates=candidates,
        feature_dim=feature_dim
    )
    study.optimize(objective, n_trials=n_trials)
