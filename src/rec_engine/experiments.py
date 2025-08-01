from pydantic import validate_call, ConfigDict
from pathlib import Path

import tensorflow as tf
import pandas as pd
import ImbalancedLearningRegression as iblr
import matplotlib.pyplot as plt

from . import log
from .config import Params
from .utils import create_dataset
from .train_functions import train_model, train_CV
from .model import RecommenderEngineModel

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def deep_layers_exp(
        candidates: tf.data.Dataset,
        data_df: pd.DataFrame,
        params: Params,
        deep_layers: list,
    ) -> pd.DataFrame:
    """
    Deep layers experimentation. It will use crossvalidation to evaluate different deep layers architectures.

    Parameters
    ---
    candidates : tf.data.Dataset
        The candidates dataset.
    data_df : pd.DataFrame
        The interactions dataset.
    params : dict
        Dictionary of model parameters. It is created by `rec_engine.config.create_params()` function.
    deep_layers : list
        The list of deep layers architectures to try.


    Returns
    ---
    pd.DataFrame
        The results for deep layers experiment.
    """
    params = params.model_dump()
    path = Path(params["logs_path"])

    # Try different deep layers architectures
    log.info("Running deep layers experiment")
    results = []
    for layers in deep_layers:
        # Use the number of layers as ID of each model
        n_layers = len(layers) - 1
        log.info(f"Training model with the following layers: \n- Query Layers: {layers} \n- Candidate Layers: {layers}")
        params["logs_path"] = str(path / f"{n_layers}_deep_layers")
        # Update layers architecture for each tower
        params["model"]["query_layers"] = layers
        params["model"]["candidate_layers"] = layers

        result = pd.DataFrame(train_CV(data_df, candidates, params))
        result["n_layers"] = n_layers
        results.append(result)

    return pd.concat(results, ignore_index= True)


def resample(data: pd.DataFrame) -> tuple:
    """
    Apply different resampling methods (*Random Oversample*, *Random Undersampling*, *Gaussian Noise*).

    Parameters
    ---
    data: pd.DataFrame
        The dataset to be resampled.

    Returns
    ---
    tuple
        A tuple of DataFrames containing the results for each resampling method.
    """

    # The rel_ctrl_pts_rg argument takes a 2d array (matrix).
    # It is used to manually specify the regions of interest or rare "minority" values in y.
    # The first column indicates the y values of interest, the second column indicates a mapped value of relevance, either 0 or 1,
    # where 0 is the least relevant and 1 is the most relevant, and the third column is indicative.
    # It will be adjusted afterwards, use 0 in most cases.
    rg_matrix = [
        [0.5, 1, 0], # minority class, high relevance
        [1.0, 1, 0], # minority class, high relevance
        [0, 0, 0] # majority class, low relevance
    ]

    # Random Oversample
    log.info("Random Oversample")
    ro_clicks_train_df = iblr.ro(
        data = data,
        y = "score",
        rel_method="manual", # Set manual to use manual relevance control
        rel_ctrl_pts_rg= rg_matrix # Set relevance control points
    )

    # Random Undersampling
    log.info("Random Undersampling")
    ru_clicks_train_df = iblr.random_under(
        data = data,
        y = "score",
        rel_method="manual",
        rel_ctrl_pts_rg= rg_matrix
    )

    # Gaussian Noise
    log.info("Gaussian Noise")
    gn_clicks_train_df = iblr.gn(
        data = data,
        y = "score",
        rel_method="manual",
        rel_ctrl_pts_rg= rg_matrix
    )

    return data, ro_clicks_train_df, ru_clicks_train_df, gn_clicks_train_df

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def resample_exp(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        candidates: tf.data.Dataset,
        params: Params,
    ) -> list[dict]:
    """
    Run resampling experiment. Resample train dataset with different techniques
    (*Random Oversample*, *Random Undersampling*, *Gaussian Noise*) and evaluate the models.

    Parameters
    ---
    train_df : pd.DataFrame
        The train dataset.
    val_df : pd.DataFrame
        The validation dataset.
    candidates : tf.data.Dataset
        The candidates dataset.
    params : dict
        Dictionary of model parameters. It is created by `rec_engine.config.create_params()` function.


    Returns
    ---
    list[dict]
        A list of history results for each resampling method.
    """
    params = params.model_dump()
    path = Path(params["logs_path"])
    path.mkdir(parents=True, exist_ok=True)
    log.info("Running resampling experiment")
    features = params["tower"]["query"] + params["tower"]["candidate"]

    # list of tf datasets
    train_sets = [*resample(train_df.reset_index(drop=True))]
    train_names = ["original", "RO", "RU", "GN"]

    # Plot densities
    plt.figure(figsize=(15, 5))
    for name, df in zip(train_names, train_sets):
        df["score"].plot(kind="kde", label=f"{name} {df.shape}")

    plt.title("Resampling Comparison")
    plt.xlabel("Score")
    plt.xticks([0, 0.5, 1])
    plt.legend(title="Resampling Method | Output shape")
    fig_path = path / "resampling_comparison.png"
    plt.savefig(fig_path)
    log.info(f"Resampling comparison plot saved at {fig_path}")

    results = []
    for i, (train_ds, name) in enumerate(zip(train_sets, train_names)):
        params["logs_path"] = str(path / name)
        log.info(f"[{i+1}/{len(train_sets)}] Training model for {name} dataset.")

        # Convert dataframe to dataset
        train = create_dataset(train_ds, features)
        val = create_dataset(val_df, features)

        # Create model instance
        model = RecommenderEngineModel(
            params=params,
            queries= train,
            candidates= candidates,
            preprocessing= True
        )

        fitted_model = train_model(
            model = model,
            train= train,
            val= val,
            params= params,
        )
        result = {k: v[-1] for k, v in fitted_model.history.history.items() if k.startswith("val")}
        results.append({"name": name, **result})

    return results
