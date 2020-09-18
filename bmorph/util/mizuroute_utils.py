from glob import glob
import xarray as xr
import pandas as pd
import geopandas as gpd
import bmorph
import numpy as np


def find_up(ds, seg):
    if ds.sel(seg=seg)['is_headwaters']:
        return None
    up_idx = np.argwhere(ds['down_seg'].values == seg).flatten()[0]
    up_seg = ds['seg'].isel(seg=up_idx).values[()]
    return up_seg


def walk_down(ds, start_seg):
    tot_length = 0.0
    cur_seg = start_seg
    if ds['is_gauge'].sel(seg=cur_seg):
        return 0.0, cur_seg
    else:
        while (ds['down_seg'].sel(seg=cur_seg).values[()] in ds['seg'].values
              and not ds['is_gauge'].sel(seg=ds['down_seg'].sel(seg=cur_seg).values[()]).values[()]):
            cur_seg = ds['down_seg'].sel(seg=cur_seg).values[()]
            tot_length += ds.sel(seg=cur_seg)['length'].values[()]
        cur_seg = ds['down_seg'].sel(seg=cur_seg).values[()]
        return tot_length, cur_seg


def walk_up(ds, start_seg):
    tot_length = 0.0
    cur_seg = start_seg
    if ds['is_gauge'].sel(seg=cur_seg):
        return 0.0, cur_seg
    else:
        while (ds['up_seg'].sel(seg=cur_seg).values[()] in ds['seg'].values
              and not ds['is_gauge'].sel(seg=ds['up_seg'].sel(seg=cur_seg).values[()]).values[()]):
            cur_seg = ds['up_seg'].sel(seg=cur_seg).values[()]
            tot_length += ds.sel(seg=cur_seg)['length'].values[()]
        cur_seg = ds['up_seg'].sel(seg=cur_seg).values[()]
        return tot_length, cur_seg


def trim_time(dataset_list: list):
    """
    Trims all times of the xarray.Datasets in the list to the shortest timeseries.
    ----
    dataset_list: list
        contains a list of xarray.Datasets
    ----
    Return: list
        contains a list in the same order as dataset_list except with all items
        in the list having the same start and end time
    """
    t_starts = list()
    t_finishes = list()

    for ds in dataset_list:
        assert isinstance(ds, xr.Dataset) #quick type check
        t_ds = ds.time.values[[0, -1]]
        t_starts.append(t_ds[0])
        t_finishes.append(t_ds[1])

    t_trim_start = np.max(t_starts)
    t_trim_finish = np.min(t_finishes)
    t_slice = slice(t_trim_start, t_trim_finish)

    dataset_list_trimmed = list()

    for ds in dataset_list:
        dataset_list_trimmed.append(ds.sel(time=t_slice))

    return dataset_list_trimmed


def map_segs_topology(routed: xr.Dataset, topology: xr.Dataset):
    """
    adds contributing_area, average elevation, length, and down_seg to
    routed from topology
    """
    routed = routed.sel(seg=topology['seg'])
    routed['contributing_area'] = topology['Contrib_Area']
    routed['elevation'] = 0.5 * (topology['TopElev'] + topology['BotElev'])
    routed['length'] = topology['Length']
    routed['down_seg'] = topology['Tosegment']

    return routed


def map_ref_sites(routed: xr.Dataset, gauge_reference: xr.Dataset,
                    gauge_sites=None):
    """
    boolean identifies whether a seg is a gauge with 'is_gauge'
    """
    if isinstance(gauge_sites, type(None)):
        gauge_sites = gauge_reference['site'].values
    else:
        # need to typecheck since we do a for loop later and don't
        # want to end up iterating through a string by accident
        assert isinstance(gauge_sites, list)

    gauge_segs = gauge_reference.sel(site=gauge_sites)['seg'].values

    routed['is_gauge'] = False * routed['seg']
    routed['down_ref_seg'] = np.nan * routed['seg']
    routed['up_ref_seg'] = np.nan * routed['seg']
    routed['up_seg'] = 0 * routed['is_headwaters']
    routed['up_seg'].values = [find_up(routed, s) for s in routed['seg'].values]
    for s in routed['seg']:
        if s in list(gauge_segs):
            routed['is_gauge'].loc[{'seg':s}] = True

    for s in routed['seg']:
        if routed.sel(seg=s)['is_gauge']:
            down_seg = routed.sel(seg=s)['down_seg'].values[()]
            down_ref_seg =  walk_down(routed, down_seg)[1]
            if down_ref_seg in routed['seg']:
            	routed['down_ref_seg'].loc[{'seg':s}] = down_ref_seg

    for s in routed['seg']:
        if routed.sel(seg=s)['is_gauge']:
            routed['up_ref_seg'].loc[{'seg': s}] = s

    for seg in routed['seg']:
        cur_seg = seg.values[()]
        while cur_seg in routed['seg'].values and np.isnan(routed['down_ref_seg'].sel(seg=cur_seg)):
            cur_seg = routed['down_seg'].sel(seg=cur_seg).values[()]
        if cur_seg in routed['seg'].values:
            routed['down_ref_seg'].loc[{'seg':seg}] = routed['down_ref_seg'].sel(seg=cur_seg).values[()]

    for seg in routed['seg']:
        cur_seg = seg.values[()]
        while cur_seg in routed['seg'].values and np.isnan(routed['up_ref_seg'].sel(seg=cur_seg)):
            cur_seg = routed['up_seg'].sel(seg=cur_seg).values[()]
        if cur_seg in routed['seg'].values:
            routed['up_ref_seg'].loc[{'seg':seg}] = routed['up_ref_seg'].sel(seg=cur_seg).values[()]

    # Fill in any remaining nulls (head/tailwaters)
    routed['up_ref_seg'] = (routed['up_ref_seg'].where(
        ~np.isnan(routed['up_ref_seg']), other=routed['down_ref_seg'])).ffill('seg')
    routed['down_ref_seg'] = (routed['down_ref_seg'].where(
        ~np.isnan(routed['down_ref_seg']), other=routed['up_ref_seg'])).ffill('seg')

    return routed


def map_headwater_sites(routed: xr.Dataset):
    """
    boolean identifies whether a seg is a headwater with 'is_headwater'
    """
    if not 'down_seg' in list(routed.var()):
        raise Exception("Please denote down segs with 'down_seg'")

    routed['is_headwaters'] = False * routed['seg']
    headwaters = [s not in routed['down_seg'].values for s in routed['seg'].values]
    routed['is_headwaters'].values = headwaters

    return routed


def calculate_cdf_blend_factor(routed: xr.Dataset):
    """
    calcultes the cumulative distirbtuion function blend factor based on distance
    to a seg's nearest up gauge site with respect to the total distance between
    the two closest guage sites to the seg
    """
    if not 'is_gauge' in list(routed.var()):
        # needed for walk_up and walk_down
        raise Exception("Please denote headwater segs with 'is_headwaters'")

    routed['distance_to_up_gauge'] = 0 * routed['is_gauge']
    routed['distance_to_down_gauge'] = 0 * routed['is_gauge']
    routed['cdf_blend_factor'] = 0 * routed['is_gauge']

    routed['distance_to_up_gauge'].values = [walk_up(routed, s)[0] for s in routed['seg']]
    routed['distance_to_down_gauge'].values = [walk_down(routed, s)[0] for s in routed['seg']]
    routed['cdf_blend_factor'].values = (routed['distance_to_up_gauge']
                                        / (routed['distance_to_up_gauge']
                                          + routed['distance_to_down_gauge'])).values
    routed['cdf_blend_factor'] = routed['cdf_blend_factor'].where(~np.isnan(routed['cdf_blend_factor']), other=0.0)

    return routed


def map_ref_seg(routed: xr.Dataset):
    """
    maps the nearest gauge site upsteam and down of a seg
    """
    if 'down_seg' not in list(routed.var()):
        #note 'up_seg' does not show up in var()
        raise Exception("Please run 'map_segs_topology' and 'map_up_segs' first")

    for seg in routed['seg']:
        cur_seg = seg.values[()]
        while cur_seg in routed['seg'].values and np.isnan(routed['down_ref_seg'].sel(seg=cur_seg)):
            cur_seg = routed['down_seg'].sel(seg=cur_seg).values[()]
        if cur_seg in routed['seg'].values:
            routed['down_ref_seg'].loc[{'seg':seg}] = routed['down_ref_seg'].sel(seg=cur_seg).values[()]

        cur_seg = seg.values[()]
        while cur_seg in routed['seg'].values and np.isnan(routed['up_ref_seg'].sel(seg=cur_seg)):
            cur_seg = routed['up_seg'].sel(seg=cur_seg).values[()]
        if cur_seg in routed['seg'].values:
            routed['up_ref_seg'].loc[{'seg':seg}] = routed['up_ref_seg'].sel(seg=cur_seg).values[()]

    # Fill in any remaining nulls (head/tailwaters)
    routed['down_ref_seg'] = (routed['down_ref_seg'].where(
        ~np.isnan(routed['down_ref_seg']), other=routed['up_ref_seg']))
    routed['up_ref_seg'] = (routed['up_ref_seg'].where(
        ~np.isnan(routed['up_ref_seg']), other=routed['down_ref_seg']))

    return routed


def calculate_blend_vars(routed: xr.Dataset, topology: xr.Dataset, reference: xr.Dataset,
                     gauge_sites = None):
    """
    calculates a number of variables used in blendmorph and map_var_to_seg
    ----
    routed: xr.Dataset
        the dataset that will be modified and returned ready for map_var_to_seg
    topology: xr.Dataset
        contains the network topology with a "seg" dimension that identifies reaches,
        matching the routed dataset
    reference: xr.Dataset
        contains reaches used for reference with dimension "site" and coordinate "seg"
    gauge_sites: None
        contains the gauge site names from the reference dataset to be used that are
        automatically pulled from reference if None are given
    ----
    Return: routed (xr.Dataset)
        with the following added:
        'is_headwaters'
        'is_gauge'
        'down_seg'
        'distance_to_up_gauge'
        'distance_to_down_gauge'
        'cdf_blend_factor'
        'up_seg'
        'up_ref_seg'
        'down_ref_seg'
    """
    routed = map_segs_topology(routed=routed, topology=topology)

    # check if trim_time should be suggested
    t_reference = reference.time.values[[0, -1]]
    t_routed = routed.time.values[[0, -1]]
    if t_reference[0] != t_routed[0] or t_reference[1] != t_routed[1]:
        raise Exception("Please ensure reference and routed have the same starting and ending times, (this can be done with trim_time)")

    routed = map_headwater_sites(routed=routed)
    routed = map_ref_sites(routed=routed, gauge_reference=reference,
                             gauge_sites = gauge_sites)
    routed = calculate_cdf_blend_factor(routed=routed)
    return routed


def map_var_to_segs(routed: xr.Dataset, map_var: xr.DataArray, var_label: str,
                    gauge_segs = None):
    """
    splits the variable into its up and down components to be used in blendmorph
    ----
    routed: xr.Dataset
        the dataset that will be modified and returned having been prepared by calculate_blend_vars
        with the dimension 'seg'
    map_var: xr.DataArray
        contains the variable to be split into up and down components and can be
        the same as routed, (must also contain the dimension 'seg')
    var_label: str
        suffix of the up and down parts of the variable
    gauge_segs: None
        list of the gauge segs that identify the reaches that are gauge sites, pulled from routed
        if None
    ----
    Return: routed (xr.Dataset)
        with the following added:
        f'down_{var_label}'
        f'up_{var_label}'
    """

    if not 'down_seg' in list(routed.var()):
        raise Exception("Please run calculate_blend_vars before running this function")

    # check if trim_time should be suggested
    t_map_var = map_var.time.values[[0, -1]]
    t_routed = routed.time.values[[0, -1]]
    if t_map_var[0] != t_routed[0] or t_map_var[1] != t_routed[1]:
        raise Exception("Please ensure reference and routed have the same starting and ending times, (this can be done with trim_time)")

    down_var = f'down_{var_label}'
    up_var = f'up_{var_label}'

    # need to override dask array data protections
    map_var.load()

    routed[down_var] = np.nan * map_var
    routed[up_var] = np.nan * map_var

    for seg in routed['seg'].values:
        up_seg = routed['up_ref_seg'].sel(seg=seg)
        down_seg = routed['down_ref_seg'].sel(seg=seg)
        routed[up_var].loc[{'seg': seg}] = map_var.sel(seg=up_seg).values[:]
        routed[down_var].loc[{'seg': seg}] = map_var.sel(seg=down_seg).values[:]

    return routed


def map_met_hru_to_seg(met_hru, topo):

    hru_2_seg = topo['seg_hru_id'].values
    met_vars = set(met_hru.variables.keys()) - set(met_hru.coords)
    # Prep met data structures
    met_seg = xr.Dataset({'time': met_hru['time']})
    for v  in met_vars:
        met_seg[v] = xr.DataArray(data=np.nan, dims=('time', 'seg', ),
                            coords={'time': met_hru['time'], 'seg': topo['seg']})

    # Map from hru -> segment for met data
    # In case a mapping doesn't exist to all segments,
    # we define a neighborhood search to spatially average
    null_neighborhood = [-3, -2, -1, 0, 1, 2, 3]
    for var in met_vars:
        for seg in met_seg['seg'].values:
            subset = np.argwhere(hru_2_seg == seg).flatten()
            # First fallback, search in the null_neighborhood
            if not len(subset):
                subset = np.hstack([np.argwhere(hru_2_seg == seg-offset).flatten()
                                    for offset in null_neighborhood])
            # Second fallback, use domain average
            if not len(subset):
                subset = met_hru['hru'].values
            met_seg[var].loc[{'seg': seg}] = met_hru[var].isel(hru=subset).mean(dim='hru')

    return met_seg


def mizuroute_to_blendmorph(topo: xr.Dataset, routed: xr.Dataset, reference: xr.Dataset,
                            met_hru: xr.Dataset=None, route_var: str='IRFroutedRunoff'):
    '''
    Prepare mizuroute output for bias correction via the blendmorph algorithm. This
    allows an optional dataset of hru meteorological data to be given for conditional
    bias correction.

    Parameters
    ----------
    topo:
        Topology dataset for running mizuRoute.
        We expect this to have ``seg`` and ``hru`` dimensions
    routed:
        The initially routed dataset from mizuRoute
    reference:
        A dataset containing reference flows for bias correction.
        We expect this to have ``site`` and ``time`` dimensions.
    met_hru:
        (Optional) A dataset of meteorological data to be mapped
        onto the stream segments to facilitate conditioning.
        All variables in this dataset will automatically be mapped
        onto the stream segments and returned
    route_var:
        Name of the variable of the routed runoff in the ``routed``
        dataset. Defaults to ``IRFroutedRunoff``

    Returns
    -------
    met_seg:
        A dataset with the required data for applying the ``blendmorph``
        routines. See the ``blendmorph`` documentation for further information.
    '''
    if met_hru is None:
        met_hru = xr.Dataset(coords={'time': routed['time']})
    # Provide some convenience data for mapping/loops
    ref_sites = list(reference['site'].values)
    ref_segs = list(reference['seg'].values)
    hru_2_seg = topo['seg_hru_id'].values
    met_vars = set(met_hru.variables.keys()) - set(met_hru.coords)

    # Remap any meteorological data from hru to stream segment
    met_seg = map_met_hru_to_seg(met_hru, topo)

    # Get the longest overlapping time period between all datasets
    [routed, reference, met_seg] = trim_time([routed, reference, met_seg])
    routed = calculate_blend_vars(routed, topo, reference)

    # Put all data on segments
    seg_ref =  xr.Dataset({'reference_flow':(('time','seg'), reference['reference_flow'].values)},
                            coords = {'time': reference['time'].values, 'seg': ref_segs},)
    routed = map_var_to_segs(routed, routed[route_var], 'raw_flow')
    routed = map_var_to_segs(routed, seg_ref['reference_flow'], 'ref_flow')
    for v in met_vars:
        routed = map_var_to_segs(routed, met_seg[v], v)

    # Merge it all together
    met_seg = xr.merge([met_seg, routed])
    return met_seg
