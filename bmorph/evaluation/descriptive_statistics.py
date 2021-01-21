import numpy as np
import pandas as pd


def mbe(observe:pd.DataFrame, predict:pd.DataFrame) -> pd.DataFrame:
    """Mean bias error of predicted data.
    
    Calculates mean bias error as sum(predict - observe)/(total entries).
    
    Parameters
    ----------
    observe : pandas.DataFrame
            Observed values.
    predict : pandas.DataFrame
            Predicted values.

    Returns
    -------
    pandas.DataFrame
        Mean bias error of predicted values.
    """
    diff = predict - observe
    mbe = diff.sum()/len(observe.index)
    mbe = mbe.to_frame().T
    return mbe


def rmse(observe:pd.DataFrame, predict:pd.DataFrame) -> pd.DataFrame:
    """Root mean square error of predicted data.
    
    Calculates root mean square error between observe and predict.
    
    Parameters
    ----------
    observe : pandas.DataFrame
            Oserved values.
    predict : pandas.DataFrame
            Predicited values.

    Returns
    -------
    pandas.DataFrame
        Root mean square error of predcited values.
    """
    n = len(observe.index)
    diff = predict - observe
    diff = diff.pow(2)
    rmse = diff.sum()/len(observe.index)
    rmse = rmse.to_frame().T
    rmse = rmse.pow(0.5)
    return rmse


def pbias(observe:pd.DataFrame, predict:pd.DataFrame) -> pd.DataFrame:
    """Percent bias of predicted data.
    
    Calculates percent bias as 100%*(sum(predict-observe)/sum(observe)).
    
    Parameters
    ----------
    observe : pandas.DataFrame
            Observed values.
    predict : pandas.DataFrame
            Predicted values.

    Returns
    -------
    pandas.DataFrame
        Precent bias of predicted values.
    """
    pbdf = predict-observe
    pbdf = 100*(pbdf.sum()/observe.sum())
    pbdf = pbdf.to_frame().T
    return pbdf

def pbias_by_index(observe:pd.DataFrame, predict:pd.DataFrame):
    """Percent bias of predicted data for each entry.
    
    Computes percent bias at the same regularity as the index 
    using predict-obsererve assuming aggregation has already 
    been performed on both DataFrames. This is useful for 
    looking at percent bias by month or year for example.
    
    Parameters
    ----------
    observe : pandas.DataFrame
        Observed values.
    predict : pandas.DataFrame
        Predicted values.
    
    Returns
    -------
    pandas.Dataframe
        Percent bias of predictions according to index provided.
    """
    pbdf = predict-observe
    pbdf = 100*(pbdf/observe)
    return pbdf


def normalize_flow(data:pd.DataFrame) -> pd.DataFrame:
    """Normalizes data.
    
    Normalizes only the flow values of a given DataFrame. A normalized
    value is computed as: (value - column_min)/(column_max -column_min).
    
    Parameters
    ----------
    data : pandas.DataFrame

    Returns
    -------
    pd.DataFrame 
        normalized 'data' without altering the index
    """
    normal_df = pd.DataFrame(index=data.index, columns=data.columns)

    for column in data.columns:
        column_series = data[column]
        min_x = column_series.min()
        max_x = column_series.max()
        column_series = (column_series - min_x)/(max_x-min_x)
        normal_df[column] = column_series

    return normal_df

def normalize_flow_pair(data:pd.DataFrame, norming_data:pd.DataFrame):
    """Normalize flow by other data.
    
    Normalizes two DataFrames by one of them, (norming_data), and 
    returns both normalized by the same DataFrame.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Contains flows to be normalized by the norming_data.
    norming_data : pandas.DataFrame
        Contains flows to be normalized and to normalize data by.
    
    Returns
    -------
    data_normed : pandas.DataFrame
        The given Dataset 'data' normalized by 'norming_data'.
    norming_data_normed : pandas.DataFrame
        The given Dataset 'norming_data' normalized by itself.
    """
    data_normed = pd.DataFrame(index=data.index, columns=data.columns)
    norming_normed = pd.DataFrame(index=norming_data.index, columns=norming_data.columns)
    
    for data_column, norming_column in zip(data.columns, norming_data.columns):
        data_column_series = data[data_column]
        norming_column_series = norming_data[data_column]
        
        min_x = norming_column_series.min()
        max_x = norming_column_series.max()
        
        data_column_series = (data_column_series-min_x)/(max_x-min_x)
        norming_column_series = (norming_column_series-min_x)/(max_x-min_x)
        
        data_normed[data_column] = data_column_series
        norming_normed[norming_column] = norming_column_series
    
    return data_normed, norming_normed


def mean_standardize_flow(data:pd.DataFrame) -> pd.DataFrame:
    """Standardizes data by the mean.
    
    Standardize only the flow values of a given DataFrame about 
    their mean flow value.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Contains values to be standardized.

    Returns
    -------
    pandas.DataFrame 
        'data' standardized about the mean without alterting the index.
    """
    mean_standard_df = pd.DataFrame(index=data.index, columns=data.columns)

    for column in data.columns:
        column_series = data[column]
        mean = column_series.mean()
        std = column_series.std()
        column_series = (column_series - mean)/std
        mean_standard_df[column] = column_series

    return mean_standard_df


def median_standardize_flow(data:pd.DataFrame) -> pd.DataFrame:
    """Standardizes data by the median.
    
    Standardize only the flow values of a given DataFrame about 
    their median flow value.
    
    Parameters
    ----------
    data : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame 
        'data' standardized about the medain without alterting the index.
    """

    median_standard_df = pd.DataFrame(index = data.index, columns = data.columns)

    for column in data.columns:
        column_series = data[column]
        median = column_series.median()
        mad = column_series.mad()
        column_series = (column_series - median)/mad
        median_standard_df[column] = column_series

    return median_standard_df