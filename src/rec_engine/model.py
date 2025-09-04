import re
from pydantic import validate_call, ConfigDict

import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np

from .config import Params
from .preprocessing import Preprocessing


## Embedding Layer
@tf.keras.saving.register_keras_serializable(package="RecommendationEngine", name="EmbeddingsLayer")
class Embeddings(tf.keras.layers.Layer):
    """
    A custom Keras layer designed to handle embedding generation for various types of features.
    The layer manages multiple embeddings and concatenates them to create a feature representation.

    Attributes
    ---
    name : str
        Name of the layer
    params : dict
        Dictionary of model parameters. It is created by `rec_engine.config.create_params()` function.
    feature_dim : dict[str, int]
        Features with vocabulary size to be used as input dimension for the embedding layer.

    Methods
    ---
    call(input: dict[str, tf.Tensor])
        Generate embeddings for the given input features
    """
    def __init__(self, name: str, params: dict, feature_dim: dict):
        super().__init__(name= name)

        self.params = params
        self.feature_dim = feature_dim

        self.l_name = name.replace("Embeddings", "").lower()
        self.features = params["tower"][self.l_name]
        self.emb_weight = params["model"].get("emb_weight", 1)

        self.embeddings = {}
        self.output_dim = 0

        for feature in self.features:
            if "-" not in feature:
                continue

            prep, feat = feature.split("-")
            if prep in ["cat", "int"]:
                emb_dim, embedding = self._get_embedding(feature)
                self.embeddings[feature] = embedding

            elif prep == "text":
                emb_dim, embedding = self._get_embedding(feature)
                self.embeddings[feature] = tf.keras.Sequential([
                    embedding,
                    tf.keras.layers.GlobalAveragePooling1D(name= f"text_avg_{feature}"),
                ])

            elif prep == "disc":
                emb_dim, embedding = self._get_embedding(feature)
                self.embeddings[f"disc-clip_{feat}"] = embedding

            elif prep == "seq":
                emb_dim, embedding = self._get_embedding(feature)
                self.embeddings[feature] = tf.keras.Sequential([
                    embedding,
                    tf.keras.layers.GRU(emb_dim, name=f"seq_gru_{feature}"),
                ])

    def _get_embedding(self, feature: str) -> tuple[int, tf.keras.layers.Layer]:
        feature_dim = self.feature_dim[feature]
        emb_dim = int((np.log2(feature_dim) + 1) * self.emb_weight)
        self.output_dim += emb_dim
        return emb_dim, tf.keras.layers.Embedding(feature_dim + 1, emb_dim, name=f"emb_{feature}", mask_zero=feature.split("-")[0] == "text")

    def call(self, input: dict[str, tf.Tensor]):
        # Add normalized features as they are because they don't need embeddings. Only reshape is needed.
        normalized_features = [tf.reshape(input[f"norm-clip_{f.split('-')[1]}"], (-1, 1)) for f in self.features if f.split("-")[0] == "norm"]
        # Concat embeddings and normalized features
        return tf.concat([self.embeddings[feature](input[feature]) for feature in self.embeddings.keys()] + normalized_features, axis=1)

    def get_config(self) -> dict:
        # Method needed for serialization and saving
        config = super().get_config()
        config.update({
            "params": self.params,
            "feature_dim": self.feature_dim,
        })
        return config


## Deep Layers Model
@tf.keras.saving.register_keras_serializable(package="RecommendationEngine", name="DeepLayers")
class DeepLayers(tf.keras.Model):
    """
    A custom Keras model representing a deep neural network with optional cross-layer functionality.
    This layer is designed to build a stack of dense layers with ReLU activation and include dropout for regularization.
    Dropout layer is added after each deep layer except the last two layers.

    Attributes
    ---
    name : str
        Name of the layer
    model_params : dict[str, Any]
        Dictionary of model parameters
    input_dim : int
        Output dimension of the last layer which is the embedding layer in this case. Needed by Cross Layer projection.

    Methods
    ---
    call(input: tf.Tensor)
        Forward pass of the model
    """
    def __init__(self, name: str, model_params: dict, input_dim: int) -> None:
        super().__init__(name= name)

        self.model_params = model_params
        self.l_name = re.sub("Model", "", name).lower()
        self.deep_layers = model_params[f"{self.l_name}_layers"]
        self.model = tf.keras.Sequential(name=f"{self.l_name}_deep_layers")

        if model_params["cross_layer"]:
            # Note that projection_dim needs to be smaller than (input size)/2 to reduce the cost.
            # In practice, it've been observed using low-rank DCN with rank (input size)/4 consistently preserved the accuracy of a full-rank DCN.
            self.model.add(tfrs.layers.dcn.Cross(projection_dim=input_dim // 4, kernel_initializer="glorot_uniform", name=f"cross_{self.l_name}"))

        # Use the ReLU activation for all but the last layer.
        for i, layer_size in enumerate(self.deep_layers[:-1]):
            self.model.add(tf.keras.layers.Dense(layer_size, activation="relu", name=f"{self.l_name}_layer{i:02d}"))

            # Add dropout layer for regularization after each layer except between the last two layers
            if i != len(self.deep_layers) - 1:
                self.model.add(tf.keras.layers.Dropout(model_params["dropout"], name=f"{self.l_name}_dropout{i:02d}"))

        # No activation for the output layer.
        self.model.add(tf.keras.layers.Dense(self.deep_layers[-1], name=f"{self.l_name}_output_layer"))

    def call(self, input: tf.Tensor):
        return self.model(input)

    def get_config(self) -> dict:
        # Method needed for serialization and saving
        config = super().get_config()
        config.update({
            "model_params": self.model_params
        })
        return config


## Recommender Engine Model
class RecommenderEngineModel(tfrs.models.Model):
    """
    A custom recommendation model built on top of TensorFlow Recommenders (TFRS) framework.
    This model implements a dual-tower architecture for learning query and candidate embeddings.
    It supports both retrieval (matching querys to candidates) and ranking tasks (predicting query ratings for candidates).
    Optionally, it handles input feature preprocessing, which is disabled during the experimentation workflow and enabled during serving.

    Attributes
    ---
    params : dict
        Dictionary of model parameters. It is created by `rec_engine.config.create_params()` function.
    candidates : tf.data.Dataset
        Candidates dataset. Must be preprocessed if `preprocessing = False`.
    queries : tf.data.Dataset
        Query dataset. Only needed if `preprocessing = True`, but not preprocessed.
    preprocessing : bool
        Whether to apply preprocessing to the input datasets.
        This should be disabled in experimentation phase and enabled in serving phase.
    feature_dim : dict[str, int]
        Features with vocabulary size to be used as input dimension for the embedding layer.
    train_metrics : bool, optional
        Whether to enable training metrics.

    Methods
    ---
    call(input: dict[str, tf.Tensor])
        Forward pass of the model
    """
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
            self,
            params: Params,
            candidates: tf.data.Dataset,
            queries: tf.data.Dataset = None,
            preprocessing: bool = True,
            feature_dim: dict = None,
            train_metrics: bool = False
        ):
        super().__init__()

        self.params = params.model_dump()
        self.preprocessing = preprocessing
        self.feature_dim = feature_dim
        self.train_metrics = train_metrics

        # Input Deep Layers verification
        assert self.params["model"]["query_layers"][-1] == self.params["model"]["candidate_layers"][-1], "Query and Candidate output layers must have the same dimension"
        assert self.params["model"]["rating_layers"][-1] == 1, "Rating output layer must have 1 unit"

        # Preprocessing
        if preprocessing:
            self.query_prep_layer = Preprocessing(
                "query",
                self.params,
                queries.batch(1024).cache().prefetch(tf.data.AUTOTUNE)
            )
            self.candidate_prep_layer = Preprocessing(
                "candidate",
                self.params,
                candidates.batch(1024).cache().prefetch(tf.data.AUTOTUNE)
            )
            self.feature_dim = {**self.query_prep_layer.feature_dim, **self.candidate_prep_layer.feature_dim}
        else:
            self.query_prep_layer = self.candidate_prep_layer = None

            if self.feature_dim is None:
                raise ValueError("feature_dim must be provided when preprocessing is disabled")

        # QUERY
        self.query_embedding_layer: tf.Tensor = Embeddings("QueryEmbeddings", self.params, self.feature_dim)
        self.query_model_layer = DeepLayers("QueryModel", self.params["model"], self.query_embedding_layer.output_dim)

        # CANDIDATE
        self.candidate_embedding_layer: tf.Tensor = Embeddings("CandidateEmbeddings", self.params, self.feature_dim)
        self.candidate_model_layer = DeepLayers("CandidateModel", self.params["model"], self.candidate_embedding_layer.output_dim)

        # RATING
        self.rating_model: tf.Tensor = DeepLayers(
            "RatingModel",
            self.params["model"],
            self.query_embedding_layer.output_dim + self.candidate_embedding_layer.output_dim
        )

        # TASKS
        # Retrieval Task
        # Build the candidates model to be used as candidates.
        self.candidate_model = tf.keras.Sequential(name="candidates_model")
        if preprocessing:
            self.candidate_model.add(self.candidate_prep_layer)
        self.candidate_model.add(self.candidate_embedding_layer)
        self.candidate_model.add(self.candidate_model_layer)

        self.retrieval_task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                # candidates here is the preprocessed version of candidates
                (
                    candidates
                    .batch(1024)
                    .cache().prefetch(tf.data.AUTOTUNE)
                    .map(self.candidate_model, num_parallel_calls=tf.data.AUTOTUNE)
                )
            )
        )

        # Ranking Task
        self.rating_task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )

    def call(self, input: dict[str, tf.Tensor]):
        # Preprocess the input data
        if self.preprocessing:
            query_data = self.query_prep_layer(input)
            candidate_data = self.candidate_prep_layer(input)
            input = {**query_data, **candidate_data}

        # Get embeddings
        query_embedding: tf.Tensor = self.query_embedding_layer(input)
        candidate_embedding: tf.Tensor = self.candidate_embedding_layer(input)

        # Get model outputs
        return (
            self.query_model_layer(query_embedding),
            self.candidate_model_layer(candidate_embedding),
            self.rating_model(tf.concat([query_embedding, candidate_embedding], axis=1))
        )

    def compute_loss(self, input: dict[str, tf.Tensor], training=False):
        # Target feature for ranking is in params dict below `taarget` key
        score: tf.Tensor = input.pop(self.params["target"])

        query, candidate, preds = self(input)

        compute_metrics = True if self.train_metrics else not training
        # We compute the loss for each task.
        rating_loss = self.rating_task(
            labels=score,
            predictions=preds,
            compute_metrics=compute_metrics
        )
        retrieval_loss = self.retrieval_task(query, candidate, compute_metrics=compute_metrics)

        # And sum up the losses
        return (rating_loss + retrieval_loss)

    def get_config(self) -> dict:
        # Method needed for serialization and saving
        config = super().get_config()
        config.update({
            "params": self.params,
            "preprocessing": self.preprocessing,
            "feature_dim": self.feature_dim,
            "train_metrics": self.train_metrics
        })
        return config
