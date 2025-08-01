import json
from pydantic import validate_call, ConfigDict
from pathlib import Path

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

from .config import Params
from .utils import create_dataset
from .model import RecommenderEngineModel
from .callbacks import CustomEarlyStopping, CustomLRSchedule


# Train Function
# Training Function
@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def train_model(
    model: RecommenderEngineModel,
    train: tf.data.Dataset,
    val: tf.data.Dataset = None,
    params: Params = None,
    learning_rates: list[float] = None,
    logging: bool = True,
    profile: tuple | int = 0,
    verbose=True,
) -> RecommenderEngineModel:
    """
    Training Function for RecommenderEngineModel.
    Callbacks:
    - `rec_engine.callbacks.CustomEarlyStopping()`
    - `tf.keras.callbacks.TerminateOnNaN()`
    - `tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_total_loss",
            factor=0.1,
            patience=3,
            mode="min",
            verbose=1,
            min_lr=1e-6,
        )`

    Parameters
    ---
    model : RecommenderEngineModel
       Instance of RecommenderEngineModel to be trained.
    train : tf.data.Dataset
        The training dataset.
    val : tf.data.Dataset
        The validation dataset.
    params : dict
        Dictionary of model parameters. It is created by `rec_engine.config.create_params()` function.
    learning_rates : list[float]
        List of learning rates provided when validation data is not available.
        To be used for the full training when schedule is known.
    logging : bool
        If True, TensorBoard logging is enabled, and model parameters are saved to `params["logs_path"]`. Default is True.
    profile : tuple or int
        Batch to profile in TensorBoard for performance tracking. Default is 0 (no profiling).
    verbose : int
        Verbosity level for model training. Default is 1.

    Returns
    ---
    model : RecommenderEngineModel
        The trained recommendation model instance.
    """
    params = params.model_dump()

    # Setup learning rate scheduler
    if val:
        lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
            **params["callbacks"]["lr_schedule"],
            verbose=verbose
        )
    else:
        lr_schedule = CustomLRSchedule(lr_schedule = learning_rates)

    # Setup callbacks
    callbacks = [
        CustomEarlyStopping(**params["callbacks"]["early_stopping"], verbose=verbose),
        tf.keras.callbacks.TerminateOnNaN(),
        lr_schedule,
    ]

    # Optionally enable TensorBoard logging
    if logging:
        # Cast `logs_path` to string to avoid errors
        params["logs_path"] = params["logs_path"]
        # Create a file writer for the log directory
        file_writer = tf.summary.create_file_writer(params["logs_path"])

        # Write model parameters to log directory
        model_metadata = json.dumps(params, indent=4)
        with file_writer.as_default():
            tf.summary.text(
                f"Parameters for {params['logs_path']}:",
                f"```\n{model_metadata}\n```",
                step=0,
            )

        # Add TensorBoard callback for logging and profiling
        callbacks.append(
            tf.keras.callbacks.TensorBoard(params["logs_path"], profile_batch=profile)
        )

    # Choose and configure the optimizer
    initial_learning_rate = params["model"]["initial_learning_rate"]
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=initial_learning_rate)
    if params["model"]["optimizer"] == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

    # Compile the model
    model.compile(optimizer=optimizer)

    model.fit(
        train.cache().batch(params["batch_size"]).prefetch(tf.data.AUTOTUNE),
        epochs=params["model"]["max_epochs"],
        validation_data=(
            val.cache().batch(1024).prefetch(tf.data.AUTOTUNE) if val else None
        ),
        callbacks=callbacks,
        verbose=verbose,
    )

    return model


## KFold Cross Validation function
@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def train_CV(
    data_df: pd.DataFrame, candidates: tf.data.Dataset, params: Params
) -> list[dict]:
    """
    Cross Validation Function for RecommenderEngineModel.

    Parameters
    ---
    data_df : pd.DataFrame
        The dataframe of training data.
    candidates : tf.data.Dataset
        The dataset of candidate items.
    params : dict
        Dictionary of model parameters. It is created by `rec_engine.config.create_params()` function.

    Returns
    ---
    list[dict]
        Cross validation results.
    """
    params = params.model_dump()
    base_path = Path(params["logs_path"])
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=params["seed"])
    features = params["tower"]["query"] + params["tower"]["candidate"]

    results = []
    for fold, (train_index, val_index) in tqdm(
        enumerate(kf.split(data_df)),
        ascii=True,
        total=n_splits,
        desc="Cross Validation",
        unit="fold",
    ):
        params["logs_path"] = str(base_path / f"fold_{fold}")
        train_df = data_df.iloc[train_index]
        val_df = data_df.iloc[val_index]

        # Convert to TF Dataset
        train = create_dataset(train_df, features)
        val = create_dataset(val_df, features)

        # Create model instance
        model = RecommenderEngineModel(
            params=params,
            queries=train,
            candidates=candidates,
            preprocessing=True,
        )

        # Train the model
        model = train_model(model=model, train=train, val=val, params=params)

        # Get metrics
        result = {
            m: v[-1] for m, v in model.history.history.items() if m.startswith("val")
        }
        result["fold"] = fold
        result["n_epochs"] = len(model.history.history["loss"])
        results.append(result)

    return results
