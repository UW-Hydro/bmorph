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