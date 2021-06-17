import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import distance
from typing import Dict, Any


def find_local_segment(lats, lons, target_latlon, n_return=10,
        metric='euclidean', gridsearch=False) -> Dict[str, np.ndarray]:
    """
    Finds the closest coordinates to a given target. Can return multiple coordinates,
    in ascending order from closest to furthest.

    Parameters
    ----------
    lats:
        Latitudes to search through
    lons:
        Longitudes to search through
    target_latlon:
        Tuple of (lat, lon) that is being searched for
    n_return:
        Number of closest coordinates to return
    metric:
        Distance metric. Can be any valid metric from ``scipy.spatial.distance.cdist``
    gridsearch:
        Whether to create a meshgrid from the given ``lats`` and ``lons``

    Returns
    -------
    dictionary containing:
        ``coords``: Coordinates of ``n_return`` nearest coordinates as (lat, lon)
        ``distances``: Distances according to ``metric`` from ``target_latlon``
        ``indices``: Indices from ``lats`` and ``lons`` to the ``n_return`` nearest
    """
    if gridsearch:
        lats, lons = np.meshgrid(lats, lons)
        source_latlons = np.vstack([lats.flatten(), lons.flatten()]).T
    else:
        source_latlons = np.vstack([lats, lons]).T
    all_distances = distance.cdist(source_latlons, [target_latlon], metric=metric)
    min_indices = np.argsort(all_distances.flatten())[:n_return]
    sort_distances = all_distances[min_indices]
    min_latlon = source_latlons[min_indices]
    return {'coords': min_latlon, 'distances': sort_distances, 'indices': min_indices}


def flow_fraction_multiplier(total_flow: pd.Series, local_flow: pd.Series,
        nsmooth: int=30) -> pd.Series:
    """
    Calculate a the ratio of local flow to total flow from timeseries
    after applying a rolling mean.

    Parameters
    ----------
    total_flow:
        Total accumulated flow at the given location
    local_flow:
        Portion of flow that is directly from the sub-basin
        (excludes upstream contributions)
    nsmooth:
        Number of timesteps to use
    """
    smoothed_total_flow = total_flow.rolling(time=nsmooth, min_periods=1, center=True).mean()
    smoothed_local_flow = local_flow.rolling(time=nsmooth, min_periods=1, center=True).mean()
    return smoothed_local_flow / smoothed_total_flow


def quantile_regression(total_flow: pd.Series, local_flow: pd.Series,
        nsmooth: int=30):
    """TODO: Implement"""
    raise NotImplementedError()


def estimate_local_flow(
        ref_total_flow: pd.Series, sim_total_flow: pd.Series, sim_local_flow: pd.Series,
        how: str='flow_fraction', method_kwargs: Dict[str, Any]={}) -> pd.Series:
    """
    Estimate the local flow for a single site.

    Parameters
    ----------
    ref_total_flow:
        Reference total flow to calculate the reference_local_flow from
    sim_total_flow:
        Simulated total flow
    sim_local_flow:
        Simulated reference flow
    how:
        How to estimate the local reference flow. Available options are
        ``flow_fraction``
    method_kwargs:
        Additional arguments to pass to the estimator.

    Returns
    -------
    the estimated local reference flow
    """
    valid_methods = ['flow_fraction']
    if how == 'flow_fraction':
        ffm = flow_fraction_multiplier(sim_total_flow, sim_local_flow, **method_kwargs)
        est_local_flow = ref_total_flow * ffm
    elif how == 'quantile_regression':
        regressor = quantile_regression(sim_total_flow, sim_local_flow, **method_kwargs)
        est_local_flow = regressor.predict(ref_total_flow)
    else:
        raise NotImplementedError(f'Must set how to one of the available methods: {valid_methods}')
    return est_local_flow
