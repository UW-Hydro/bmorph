import numpy as np
import pandas as pd


def mbe(observe: pd.DataFrame, predict: pd.DataFrame) -> pd.DataFrame:
    """
    Mean Bias Error
    ----
    observe: pandas DataFrame
            the observations
    predict: pandas DataFrame
            the predictions

    Returns: mean bias error of predictions
    """
    diff = predict - observe
    mbe = diff.sum()/len(observe.index)
    mbe = mbe.to_frame().T
    return mbe


def rmse(observe: pd.DataFrame, predict: pd.DataFrame) -> pd.DataFrame:
    """
    Root Mean Square Error
    ----
    observe: pandas DataFrame
            the observations
    predict: pandas DataFrame
            the predictions

    Returns: root mean square error of predcitions
    """
    n = len(observe.index)
    diff = predict - observe
    diff = diff.pow(2)
    rmse = diff.sum()/len(observe.index)
    rmse = rmse.to_frame().T
    rmse = rmse.pow(0.5)
    return rmse


def pbias(observe: pd.DataFrame, predict: pd.DataFrame) -> pd.DataFrame:
    """
    Percent Bias
    ----
    observe: pandas DataFrame
            the observations
    predict: pandas DataFrame
            the predictions

    Returns: precent bias of predictions
    """
    pbdf = predict-observe
    pbdf = 100*(pbdf.sum()/observe.sum())
    pbdf = pbdf.to_frame().T
    return pbdf


def normalize_flow(data: pd.DataFrame) -> pd.DataFrame:
    """
    normalize_flow
        normalizes only the flow values of a given
        DataFrame
    ----
    data: pd.DataFrame

    returns: pd.DataFrame of 'data' normalized without altering
        the index
    """
    normal_df = pd.DataFrame(index = data.index, columns = data.columns)

    for column in data.columns:
        column_series = data[column]
        min_x = column_series.min()
        max_x = column_series.max()
        column_series = (column_series - min_x)/(max_x-min_x)
        normal_df[column] = column_series

    return normal_df


def mean_standardize_flow(data: pd.DataFrame) -> pd.DataFrame:
    """
    mean_standardize_flow
        standardize only the flow values of a given DataFrame
        about their mean
    ----
    data: pd.DataFrame

    returns: pd.DataFrame of 'data' standardized about the mean
        without alterting the index
    """
    mean_standard_df = pd.DataFrame(index=data.index,columns = data.columns)

    for column in data.columns:
        column_series = data[column]
        mean = column_series.mean()
        std = column_series.std()
        column_series = (column_series - mean)/std
        mean_standard_df[column] = column_series

    return mean_standard_df


def median_standardize_flow(data:pd.DataFrame) -> pd.DataFrame:
    """
    mean_standardize_flow
        standardize only the flow values of a given DataFrame
        about their median
    ----
    data: pd.DataFrame

    returns: pd.DataFrame of 'data' standardized about the medain
        without alterting the index
    """

    median_standard_df = pd.DataFrame(index = data.index, columns = data.columns)

    for column in data.columns:
        column_series = data[column]
        median = column_series.median()
        mad = column_series.mad()
        column_series = (column_series - median)/mad
        median_standard_df[column] = column_series

    return median_standard_df
