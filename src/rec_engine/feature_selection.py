import copy
from itertools import zip_longest
from pydantic import validate_call, ConfigDict
from pathlib import Path

import tensorflow as tf
import numpy as np

from . import log
from .config import Params
from .train_functions import train_model
from .model import RecommenderEngineModel


class FeatureSelection():
    """
    Forward Feature selection pipeline.

    Attributes
    ---
    train : tf.data.Dataset
        The preprocessed training dataset.
    val : tf.data.Dataset
        The preprocessed validation dataset.
    candidates_ds: tf.data.Dataset
        The preprocessed candidates dataset.
    feature_dim : dict[str, int]
        The dictionary of feature dimensions.
    params : dict
        Dictionary of model parameters. It is created by `rec_engine.config.create_params()` function.
    baseline_features : dict[str, list]
        The dictionary of baseline features. Required fields:
        - `query`: [`target feature`, `id feature`]
        - `candidate`: [`id feature`]
        These are the minimum required features to build a baseline which follows a standard matrix factorization model.

    Methods
    ---
    run():
        Execute the feature selection pipeline.
    """
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
            self,
            train: tf.data.Dataset,
            val: tf.data.Dataset,
            candidates: tf.data.Dataset,
            feature_dim: dict,
            params: Params,
            baseline_features: dict[str, list],
        ):
        self.train = train
        self.val = val
        self.candidates = candidates
        self.feature_dim = feature_dim
        orig_params = params.model_dump()
        self.params = copy.deepcopy(orig_params)
        self.path = Path(orig_params["logs_path"])
        # Reset the model features
        self.params["tower"] = baseline_features

        # Remove the initial user and product features
        self.model_query_features = [
            e for e in orig_params["tower"]["query"]
            if e not in self.params["tower"]["query"]
        ]
        self.model_candidate_features = [
            e for e in orig_params["tower"]["candidate"]
            if e not in self.params["tower"]["candidate"]
        ]

        # Array to save metrics for each model
        self.results = []

        # Set the order of features
        self.features = list(zip_longest(
            self.model_query_features,
            self.model_candidate_features
        ))

        self.retrieval_metric_value = -np.inf
        self.rating_metric_value = np.inf

    def _train_model(self) -> dict:
        # Create model instance
        model = RecommenderEngineModel(
            params=self.params,
            candidates=self.candidates,
            preprocessing= False, # Disable preprocessing
            feature_dim= self.feature_dim, # Feature dimension is a must, if preprocessing is disabled
        )

        fitted_model = train_model(
            model = model,
            train= self.train,
            val= self.val,
            params= self.params
        )
        result = {k: v[-1] for k, v in fitted_model.history.history.items() if k.startswith("val")}
        result["selected"] = False
        current_retrieval_value = result["val_factorized_top_k/top_50_categorical_accuracy"]
        current_rating_value = result["val_root_mean_squared_error"]

        # Keep the feature only if it improves one of the metrics
        if (current_retrieval_value >= self.retrieval_metric_value or \
            current_rating_value <= self.rating_metric_value):
            self.retrieval_metric_value = current_retrieval_value
            self.rating_metric_value = current_rating_value
            result["selected"] = True # set flag to true if the feature will be added to the model
        return result

    def _add_feature(self, tower: str, feature: str):
        self.params["tower"][tower].append(feature)
        self.params["logs_path"] = str(self.path / f"add_{tower}_{feature}")
        log.info(f"[{self.i}/{self.total}] Added {feature} feature to the {tower.capitalize()} Tower. Fitting...")
        result = self._train_model()

        if not result["selected"]:
            log.info(f"[{self.i}/{self.total}] Feature named {feature} to {tower} tower doesn't improve the model.")
            self.params["tower"][tower].remove(feature) # Remove feature from the parameters
        return {"tower": tower, "feature": feature, **result}

    def run(self) -> list[dict]:
        """
        Step by step, it's selected one feature from the Query Tower followed by one from the Candidate Tower until all features are selected.
        Any feature that improves neither metric (retrieval nor rating metric) is discarded and removed from the model.
        """
        log.info("Running feature selection")
        self.i = 0
        self.total = len(self.model_query_features) + len(self.model_candidate_features)
        log.info(f"[{self.i}/{self.total}] Training baseline model")
        self.params["logs_path"] = str(self.path / f"baseline")
        self.results.append({"tower": "", "feature": "baseline", **self._train_model()})

        for query_feature, candidate_feature in self.features:
            # Add a feature to the user, train the model and save result
            if query_feature:
                self.i += 1
                self.results.append(self._add_feature("query", query_feature))

            # Add a feature to the product, train the model and save result
            if candidate_feature:
                self.i += 1
                self.results.append(self._add_feature("candidate", candidate_feature))
        return self.results
