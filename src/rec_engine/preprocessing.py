import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

## Preprocessing Layer
@tf.function
def replace_empty_string(x: tf.Tensor) -> tf.Tensor:
    """
    Replace empty strings or whitespaces in a feature with "[NULL]"

    Parameters
    ----------
    x : tf.Tensor
        Feature to be processed

    Returns
    -------
    tf.Tensor
        Processed feature
    """
    return tf.where(
        tf.strings.regex_full_match(x, "\s*"),
        tf.constant("[NULL]"),
        x
    )

@tf.function
def extract_features(x: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    """
    Extracts time-based features such as the hour and day of the week from a timestamp in the input using Zellers Congruence.
    The timestamp is a feature called `time` in the input data.

    Parameters
    ----------
    x : dict[str, tf.Tensor]
        Dictionary of input data.

    Returns
    -------
    dict[str, tf.Tensor]
        Dictionary of processed data.
    """
    # Extract the "time" feature
    times = x["time"]

    # Extract date and time parts using substr
    date_str = tf.strings.substr(times, 0, 10)
    time_str = tf.strings.substr(times, 11, 12)

    # Extract date components
    years = tf.strings.to_number(tf.strings.substr(date_str, 0, 4), tf.int32)
    months = tf.strings.to_number(tf.strings.substr(date_str, 5, 2), tf.int32)
    days = tf.strings.to_number(tf.strings.substr(date_str, 8, 2), tf.int32)

    # Extract hour
    hours = tf.strings.to_number(tf.strings.substr(time_str, 0, 2), tf.int32)

    # Helper function to calculate day of week using Zeller"s Congruence
    def zellers_congruence(year, month, day) -> int:
        if month < 3:
            month += 12
            year -= 1
        K = year % 100
        J = year // 100
        f = day + ((13 * (month + 1)) // 5) + K + (K // 4) + (J // 4) + (5 * J)
        return (f % 7 + 5) % 7 # Shift the output to have Monday as 0, Tuesday as 1, ...

    # Calculate the day of the week
    day_of_week = tf.vectorized_map(
        lambda x: zellers_congruence(x[0], x[1], x[2]), (years, months, days)
    )

    # Add the parsed components back to the dictionary
    return {**x, "hour": hours, "day_of_week": day_of_week}

@tf.function
def clip_cont_features(x: dict[str, tf.Tensor], features: list[str]) -> dict:
    """
    Clip continuous features to their respective boundaries.
    The distribution of continuous features present skewness and outliers.

    To reduce the impact of outliers on the model, I decide to identify and clip outliers in continuous features by the following equation:

        Q1 - 3 * IQR <= x <= Q3 + 3 * IQR

    Where `Q1` and `Q3` are the first and third quartiles respectively, and `IQR` is the interquartile range given by:

        IQR = Q3 - Q1

    Parameters
    ----------
    x : dict[str, tf.Tensor]
        Dictionary of input data.
    features: list[str]
        List of continuous features to clip.

    Returns
    -------
    dict[str, tf.Tensor]
        Dictionary of processed data.
    """
    clipped_features = {}
    for feature in features:
        Q1 = tfp.stats.percentile(x[feature], 25.0, interpolation = "lower")
        Q3 = tfp.stats.percentile(x[feature], 75.0, interpolation = "lower")
        IQR = Q3 - Q1

        lower_bound = tf.maximum(Q1 - 3 * IQR, tf.reduce_min(x[feature]))
        upper_bound = Q3 + 3 * IQR
        clipped_features[f"clip_{feature}"] = tf.clip_by_value(
            x[feature], lower_bound, upper_bound
        )
    # Add the generated clipped features to the input
    return {**x, **clipped_features}


@tf.keras.saving.register_keras_serializable(
    package="RecommendationEngine",
    name="PreprocessingLayer"
)
class Preprocessing(tf.keras.layers.Layer):
    """
    A custom Keras preprocessing layer designed to handle various types of preprocessing
    tasks such as categorical encoding, integer mapping, text vectorization, discretization, and
    normalization.

    Attributes
    ---
    name : str
        Name of the preprocessing instance layer.
    features : list[str]
        List of features to be processed.
    ds : tf.data.Dataset
        Dataset to be used for preprocessing which contains the features specified in `features` argument.

    Methods
    ---
    call(input: dict[str, tf.Tensor])
    """
    def __init__(self, name: str, features: list[str], ds: tf.data.Dataset= None):
        super().__init__(name=name)

        self.features = features
        self.prep_layers = {}
        self.feature_dim = {}

        self.extract_time_features = None
        if "int-hour" in self.features or "int-day_of_week" in self.features:
            self.extract_time_features = tf.keras.layers.Lambda(
                extract_features, name= "extract_time_features"
            )

        self.cont_features = list(
            set(
                f.split("-")[1] for f in self.features
                if f.startswith("norm-") or f.startswith("disc-")
            )
        )
        self.clip_outliers = tf.keras.layers.Lambda(
            clip_cont_features,
            name= "clip_outliers",
            arguments = {"features": self.cont_features}
        )

        # Clip outliers to adapt the corresponding layers
        if self.cont_features:
            clip_ds = ds.map(lambda x: clip_cont_features(x, self.cont_features))

        for feature in self.features:
            if not "-" in feature:
                continue

            prep, feat = feature.split("-")
            if prep in ["cat", "seq"]:
                layer = tf.keras.layers.StringLookup(name=feature)
                layer.adapt(ds.map(lambda x: x[feat]))
                self.prep_layers[feature] = layer
                self.feature_dim[feature] = layer.vocabulary_size()

            elif prep == "int":
                if feat == "hour":
                    layer = tf.keras.layers.IntegerLookup(
                        vocabulary = np.arange(0, 24, dtype=np.int32), name=feature
                    )
                elif feat == "day_of_week":
                    layer = tf.keras.layers.IntegerLookup(
                        vocabulary = np.arange(0, 7, dtype=np.int32), name=feature
                    )
                else:
                    layer = tf.keras.layers.IntegerLookup(name=feature)
                    layer.adapt(ds.map(lambda x: x[feat]))
                self.prep_layers[feature] = layer
                self.feature_dim[feature] = layer.vocabulary_size()

            elif prep == "text":
                text_layer = tf.keras.layers.TextVectorization(
                        max_tokens = 10_000,
                        output_mode="int",
                        output_sequence_length=20,
                        name=f"tv_{feature}"
                    )
                text_layer.adapt(ds.map(lambda x: x[feat]))
                self.prep_layers[feature] = tf.keras.Sequential([
                    tf.keras.layers.Lambda(
                        replace_empty_string, name= f"text_null_{feature}"
                    ),
                    text_layer
                ], name=feature)
                self.feature_dim[feature] = text_layer.vocabulary_size()

            elif prep == "disc":
                # Need to add clip before var name in order to use the clipped features rather than the original ones
                layer = tf.keras.layers.Discretization(num_bins = 100, name=feature)
                layer.adapt(clip_ds.map(lambda x: x[f"clip_{feat}"]))
                self.prep_layers[f"disc-clip_{feat}"] = layer
                self.feature_dim[feature] = layer.num_bins

            elif prep == "norm":
                layer = tf.keras.layers.Normalization(axis = None, name=feature)
                self.prep_layers[f"norm-clip_{feat}"] = layer
                layer.adapt(clip_ds.map(lambda x: x[f"clip_{feat}"]))

            else:
                raise ValueError("Preprocessing type not supported.")

    def call(self, input: dict[str, tf.Tensor]):
        # Extract time features and add them if needed
        if self.extract_time_features:
            input = self.extract_time_features(input)
        # Clip outliers in continuous features if they are present
        if self.cont_features:
            input = self.clip_outliers(input)
        for feature, layer in self.prep_layers.items():
            feat = feature.split("-")[1]
            input[feature] = layer(input[feat])
        return input # Output features will replace input features

    def get_config(self) -> dict:
        # Method needed for serialization and saving
        config = super().get_config()
        config.update({
            "features": self.features,
            # Do not include "ds" as it is not needed for serialization
        })
        return config
