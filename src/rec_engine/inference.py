from typing import Any
from datetime import datetime

import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np
from rec_engine.model import RecommenderEngineModel


class RecommendationEngine:
    """
    Recommendation Engine ready for Inference

    Attributes
    ---
    index : tfrs.layers.factorized_top_k.BruteForce
    model : RecommenderEngineModel
    candidates : tf.data.Dataset
    candidate_id : str
        The feature id for the Candidate Tower.

    Methods
    ---
    __call__(query: dict[str, Any])
        Returns a list of dictionaries with the recommendations for the provided query.
    """
    def __init__(
            self,
            index: tfrs.layers.factorized_top_k.BruteForce,
            model: RecommenderEngineModel,
            candidates: tf.data.Dataset,
            candidate_id: str
        ) -> None:
        self.index, self.model, self.candidates, self.candidate_id = index, model, candidates, candidate_id
        self.config = self.model.get_config()
        self.target = self.config["params"]["target"]

    def __call__(self, query: dict[str, Any]) -> list[dict]:
        """
        Recommendations for the provided query.

        Parameters
        ---
        query: dict[str, Any]

        Returns
        ---
        list[dict]
        """
        query = {
            k: tf.convert_to_tensor(v.isoformat() if isinstance(v, datetime) else v)
            for k, v in query.items()
        }
        # Get recommendations.
        # Note that I am expanding the dimension to match the batch size expected by the model
        query_text_features = [f.split("-")[1] for f in self.config["params"]["tower"]["query"] if f.startswith("text-")]
        query_index = {k: tf.expand_dims(v, axis=0) if k in query_text_features else v for k, v in query.items()}
        _, self.top_rec = self.index({k: [v] for k, v in query_index.items()})

        # Filter by id
        selected_candidates = self.candidates.filter(
            lambda x: tf.reduce_any(tf.equal(x[self.candidate_id], self.top_rec[0]))
        )

        # Get score
        recs = (
            selected_candidates
            .map(lambda x: {**query, **x}) # Concat with query input
            .batch(8).map(self._predict)
            .unbatch()
        )

        # Order by score
        sorted_recs = sorted(
            list(recs.as_numpy_iterator()),
            key=lambda x: x[self.target],
            reverse=True
        )

        # Decode and parse output
        decoded_recs = list(map(lambda x: {k: self._decode(v) for k, v in x.items()}, sorted_recs))
        return decoded_recs

    def _predict(self, input: dict) -> dict:
        # Get model features
        features = [
            f.split("-")[1] if "-" in f else f
            for f in (
                self.config["params"]["tower"]["candidate"] +
                self.config["params"]["tower"]["query"]
            )
        ]

        # Filter input to match model input
        input_data = {k: v for k, v in input.items() if k in features}
        _, _, score = self.model(input_data)
        # Discard query features from the output
        output = {
            k: v for k, v in input.items()
            if k not in
            [
                f.split("-")[1] if "-" in f else f
                for f in self.config["params"]["tower"]["query"]
            ]
        }
        output[self.target] = score
        return output

    def _decode(self, obj) -> dict:
        if isinstance(obj, bytes):
            return obj.decode('utf-8')
        elif isinstance(obj, np.ndarray):
            return obj.tolist()[0]
        elif isinstance(obj, np.generic):
            return obj.item()
        return obj
