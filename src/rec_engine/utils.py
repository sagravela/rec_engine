from pydantic import validate_call, ConfigDict
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Params
from .preprocessing import Preprocessing


# Function to create sequences
def create_feature_sequence(
        data_df: pd.DataFrame,
        feature: str,
        agent_id: str,
        time_feature: str,
        fix_len: int = 5
    ) -> pd.DataFrame:
    """
    Create sequential features.

    Parameters
    ---
    data_df : pd.DataFrame
        The dataframe of training data.
    feature : str
        The feature to create sequences from.
    agent_id : str
        The column name of the agent or user id.
    time_feature : str
        The column name of the time feature.
    fix_len : int
        The length of the sequences. Default is 5.

    Returns
    ---
    pd.DataFrame
        The dataframe with the added sequential features.
    """
    # Function to create cumulative lists
    def create_cumulative_list(candidates: pd.Series) -> list:
        cumulative_list = []
        result = []
        for candidate in candidates:
            result.append(cumulative_list.copy())
            cumulative_list.append(candidate)

        return result

    name = f"seq_{feature}"
    # Sort values by user_id and time and create list of value for each user
    data_df[name] = data_df.sort_values(by=[agent_id, time_feature]).groupby(agent_id)[feature].transform(create_cumulative_list)
    # Pad sequences with zeros
    data_df[name] = data_df[name].apply(lambda x: (x + [0] * fix_len)[:fix_len])
    # Cast to string
    data_df[name] = data_df[name].apply(lambda x: [str(p) for p in x])

    return data_df

def create_dataset(data_df: pd.DataFrame, features: list) -> tf.data.Dataset:
    """
    Parse Pandas DataFrame to TensorFlow Dataset.

    Parameters
    ---
    data_df : pd.DataFrame
        The dataframe of training data.
    features : list
        The list of features names to be included.

    Returns
    ---
    tf.data.Dataset
        The dataset of training data.
    """
    # Convert pandas dataframe to tensorflow dataset
    data_dict = {}
    # Only keep the features we want to use
    for f in features:
        f = f.split("-")[1] if "-" in f else f
        # Ignore if feature is not in the dataframe or if it is already in the dictionary
        if f not in data_df.columns or f in data_dict.keys():
            continue
        # Sequential features has to be handled differently
        data_dict[f] = tf.constant(data_df[f].to_list(), dtype=tf.string) if f.startswith("seq_") else data_df[f]
    return tf.data.Dataset.from_tensor_slices(data_dict)

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def split_and_preprocess(
        data_df: pd.DataFrame,
        candidate_df: pd.DataFrame,
        params: Params,
        test_size: float = 0.2,
        shuffle: bool = True,
        seed: int = 42
    ) -> tuple:
    """
    Split the input data into train and validation sets and preprocess them.

    Parameters
    ---
    data_df : pd.DataFrame
        The dataframe of training data.
    candidate_df : tf.data.Dataset
        The Candidate dataframe.
    params : dict
        Dictionary of model parameters. It is created by `rec_engine.config.create_params()` function.

    Returns
    ---
    tuple
        Tuple containing:
        - Train dataset
        - Validation dataset
        - Candidate dataset
        - Feature dimension dictionary
    """
    params = params.model_dump()
    bs = params["batch_size"]
    train_df, val_df = train_test_split(
        data_df,
        test_size=test_size,
        shuffle=shuffle,
        stratify= data_df[params['target']] if shuffle else None,
        random_state=seed
    )
    # Create dataset
    features = params["tower"]["query"] + params["tower"]["candidate"]
    train_ds = create_dataset(train_df, features)
    val_ds = create_dataset(val_df, features)
    candidate_ds = create_dataset(candidate_df, features)

    # Create preprocessing layers instances
    query_prep_layer = Preprocessing(
        "query",
        params,
        train_ds.batch(512)
    )
    candidate_prep_layer = Preprocessing(
        "candidate",
        params,
        candidate_ds.batch(512)
    )
    feature_dim = {**query_prep_layer.feature_dim, **candidate_prep_layer.feature_dim}

    # Preprocess datasets
    # For products, use the whole dataset
    prep_candidate = candidate_ds.batch(bs).map(candidate_prep_layer).unbatch()
    # For clicks, use the train dataset
    prep_query_train = train_ds.batch(bs).map(query_prep_layer).map(candidate_prep_layer).unbatch()
    prep_query_val = val_ds.batch(bs).map(query_prep_layer).map(candidate_prep_layer).unbatch()

    return prep_query_train, prep_query_val, prep_candidate, feature_dim
