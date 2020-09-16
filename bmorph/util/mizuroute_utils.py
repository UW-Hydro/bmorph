#import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.simplefilter(action='ignore', category=UserWarning)

from glob import glob
import xarray as xr
import pandas as pd
import geopandas as gpd
import bmorph
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KernelDensity

def find_upstream(ds, seg):
    if ds.sel(seg=seg)['is_headwaters']:
        return None
    up_idx = np.argwhere(ds['down_seg'].values == seg).flatten()[0]
    up_seg = ds['seg'].isel(seg=up_idx).values[()]
    return up_seg

def walk_downstream(ds, start_seg):
    tot_length = 0.0
    current_seg = start_seg
    if ds['is_gauge'].sel(seg=current_seg):
        return 0.0
    else:
        while (ds['down_seg'].sel(seg=current_seg).values[()] in ds['seg'].values 
              and not ds['is_gauge'].sel(seg=ds['down_seg'].sel(seg=current_seg).values[()]).values[()]):
            current_seg = ds['down_seg'].sel(seg=current_seg).values[()]
            tot_length += ds.sel(seg=current_seg)['length'].values[()]
        return tot_length

def walk_upstream(ds, start_seg):
    tot_length = 0.0
    current_seg = start_seg
    if ds['is_gauge'].sel(seg=current_seg):
        return 1.0
    else:
        while (ds['up_seg'].sel(seg=current_seg).values[()] in ds['seg'].values 
              and not ds['is_gauge'].sel(seg=ds['up_seg'].sel(seg=current_seg).values[()]).values[()]):
            current_seg = ds['up_seg'].sel(seg=current_seg).values[()]
            tot_length += ds.sel(seg=current_seg)['length'].values[()]
        return tot_length

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

def map_gauge_sites(routed: xr.Dataset, gauge_reference: xr.Dataset, 
                    gauge_sites=None):
    """
    boolean identifies whether a seg is a gauge with 'is_gauge'
    """
    if isinstance(gauge_sites, type(None)):
        gauge_sites = gauge_reference['outlet'].values
    else:
        # need to typecheck since we do a for loop later and don't
        # want to end up iterating through a string by accident
        assert isinstance(gauge_sites, list)
        
    gauge_segs = gauge_reference.sel(outlet=gauge_sites)['seg'].values
    
    routed['is_gauge'] = False * routed['seg']
    is_gauge = []
    for s in routed['seg']:
        if s in list(gauge_segs):
            is_gauge.append(True)
        else:
            is_gauge.append(False)
            
    routed['is_gauge'].values[:] = is_gauge
    
    return routed

def map_headwater_sites(routed: xr.Dataset):
    """
    boolean identifies whether a seg is a headwater with 'is_headwater'
    """
    if not 'down_seg' in list(routed.var()):
        raise Exception("Please denote downstream segs with 'down_seg'")
        
    routed['is_headwaters'] = False * routed['seg']
    headwaters = [s not in routed['down_seg'].values for s in routed['seg'].values]
    routed['is_headwaters'].values = headwaters
    
    return routed

def map_up_segs(routed: xr.Dataset):
    """
    maps what segs are upstream from each other, opposite to `down_seg`
    """
    if not 'is_headwaters' in list(routed.var()):
        # needed for find_upstream, so checking before we get too far
        raise Exception("Please denote headwater segs with 'is_headwaters'")
    
    routed['up_seg'] = 0 * routed['is_headwaters']
    routed['up_seg'].values = [find_upstream(routed, s) for s in routed['seg'].values]
    
    return routed

def calculate_cdf_blend_factor(routed: xr.Dataset):    
    """
    calcultes the cumulative distirbtuion function blend factor based on distance
    to a seg's nearest upstream gauge site with respect to the total distance between
    the two closest guage sites to the seg
    """
    if not 'is_gauge' in list(routed.var()):
        # needed for walk_upstream and walk_downstream
        raise Exception("Please denote headwater segs with 'is_headwaters'")
        
    routed['distance_to_up_gauge'] = 0 * routed['is_gauge']
    routed['distance_to_down_gauge'] = 0 * routed['is_gauge']
    routed['cdf_blend_factor'] = 0 * routed['is_gauge']
    
    routed['distance_to_up_gauge'].values = [walk_upstream(routed, s) for s in routed['seg']]
    routed['distance_to_down_gauge'].values = [walk_downstream(routed, s) for s in routed['seg']]
    routed['cdf_blend_factor'].values = (routed['distance_to_up_gauge']
                                        / (routed['distance_to_up_gauge']
                                          + routed['distance_to_down_gauge'])).values
    routed['cdf_blend_factor'] = routed['cdf_blend_factor'].where(~np.isnan(routed['cdf_blend_factor']), other=0.0)
    
    return routed

def map_ref_seg(routed: xr.Dataset):
    """
    maps the nearest gauge site upsteam and downstream of a seg
    """
    if 'down_seg' not in list(routed.var()):
        #note 'up_seg' does not show up in var()
        raise Exception("Please run 'map_segs_topology' and 'map_up_segs' first")
    
    routed['downstream_ref_seg'] = np.nan * routed['seg']
    routed['upstream_ref_seg'] = np.nan * routed['seg']
    
    gauge_segs = list()
    for seg in routed['seg'].values:
        if routed.sel(seg=seg)['is_gauge']:
            gauge_segs.append(seg)
            routed['upstream_ref_seg'].loc[{'seg':seg}] = seg
            routed['downstream_ref_seg'].loc[{'seg':seg}] = seg
            
    for seg in routed['seg'].values:
        if seg in gauge_segs or routed.sel(seg=seg)['down_seg'].values[()] not in routed['seg'].values:
            continue
        current_seg = seg
        while routed.sel(seg=current_seg)['down_seg'].values[()] not in gauge_segs:
            current_seg = routed.sel(seg=current_seg)['down_seg'].values[()]
        downstream_gauge_seg = routed.sel(seg=current_seg)['down_seg'].values[()]
        routed['downstream_ref_seg'].loc[{'seg': seg}] = downstream_gauge_seg
        
    for seg in routed['seg'].values:
        if seg in gauge_segs or routed.sel(seg=seg)['up_seg'].values[()] not in routed['seg'].values:
            continue
        current_seg = seg
        while routed.sel(seg=current_seg)['up_seg'].values[()] not in gauge_segs:
            current_seg = routed.sel(seg=current_seg)['up_seg'].values[()]
            if current_seg is None:
                break
        if current_seg is None:
            routed['upstream_ref_seg'].loc[{'seg':seg}] = routed['downstream_ref_seg'].loc[{'seg': seg}]
        else:
            upstream_gauge_seg = routed.sel(seg=current_seg)['up_seg'].values[()]
            routed['upstream_ref_seg'].loc[{'seg': seg}] = upstream_gauge_seg
        
    routed['downstream_ref_seg'] = (routed['downstream_ref_seg'].where(
        ~np.isnan(routed['downstream_ref_seg']), other=routed['upstream_ref_seg']))
    routed['upstream_ref_seg'] = (routed['upstream_ref_seg'].where(
        ~np.isnan(routed['upstream_ref_seg']), other=routed['downstream_ref_seg']))
    
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
        contains reaches used for reference with dimension "outlet" and coordinate "seg"
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
        'upstream_ref_seg'
        'downstream_ref_seg'
    """
    
    # map_segs_topology
    routed = map_segs_topology(routed=routed, topology=topology)
    
    # check if trim_time should be suggested
    t_reference = reference.time.values[[0, -1]]
    t_routed = routed.time.values[[0, -1]]
    if t_reference[0] != t_routed[0] or t_reference[1] != t_routed[1]:
        raise Exception("Please ensure reference and routed have the same starting and ending times, (this can be done with trim_time)")
    
    # map_headwater_sites
    routed = map_headwater_sites(routed=routed)

    # map_gauge_sites
    routed = map_gauge_sites(routed=routed, gauge_reference=reference,
                             gauge_sites = gauge_sites)
    
    # map_up_segs
    routed = map_up_segs(routed=routed)
    
    # calculate_cdf_blend_factor
    routed = calculate_cdf_blend_factor(routed=routed)
    
    # map_ref_seg
    routed = map_ref_seg(routed=routed)
    
    return routed
    
def map_var_to_segs(routed: xr.Dataset, map_var: xr.DataArray, var_label: str, 
                      gauge_segs = None):
    """
    splits the variable into its upstream and downstream components to be used in blendmorph
    ----
    routed: xr.Dataset
        the dataset that will be modified and returned having been prepared by calculate_blend_vars
        with the dimension 'seg'
    map_var: xr.DataArray
        contains the variable to be split into upstream and downstream components and can be 
        the same as routed, (must also contain the dimension 'seg')
    var_label: str
        suffix of the upstream and downstream parts of the variable
    gauge_segs: None
        list of the gauge segs that identify the reaches that are gauge sites, pulled from routed 
        if None
    ----
    Return: routed (xr.Dataset)
        with the following added:
        f'downstream_{var_label}'
        f'upstream_{var_label}'
    """
    
    if not 'down_seg' in list(routed.var()):
        raise Exception("Please run calculate_blend_vars before running this function")
        
    # check if trim_time should be suggested
    t_map_var = map_var.time.values[[0, -1]]
    t_routed = routed.time.values[[0, -1]]
    if t_map_var[0] != t_routed[0] or t_map_var[1] != t_routed[1]:
        raise Exception("Please ensure reference and routed have the same starting and ending times, (this can be done with trim_time)")
    
    downstream_var = f'downstream_{var_label}'
    upstream_var = f'upstream_{var_label}'
    
    # need to override dask array data protections
    map_var.load()
    
    routed[downstream_var] = np.nan * map_var
    routed[upstream_var] = np.nan * map_var
    
    if isinstance(gauge_segs, type(None)):
        gauge_segs = list()
        for seg in routed['seg'].values:
            if routed.sel(seg=seg)['is_gauge']:
                gauge_segs.append(seg)
    
    for seg in gauge_segs:
        routed[downstream_var].loc[{'seg':seg}] = map_var.sel(seg=seg).values[:]
        
    for seg in routed['seg'].values:
        if seg in gauge_segs or routed.sel(seg=seg)['down_seg'].values[()] not in routed['seg'].values:
            continue
        current_seg = seg
        while routed.sel(seg=current_seg)['down_seg'].values[()] not in gauge_segs:
            current_seg = routed.sel(seg=current_seg)['down_seg'].values[()]
        downstream_gauge_seg = routed.sel(seg=current_seg)['down_seg'].values[()]
        routed[downstream_var].loc[{'seg':seg}] = map_var.sel(seg=downstream_gauge_seg).values[:]
    
    for seg in routed['seg'].values:
        if seg in gauge_segs or routed.sel(seg=seg)['up_seg'].values[()] not in routed['seg'].values:
            continue
        current_seg = seg
        while routed.sel(seg=current_seg)['up_seg'].values[()] not in gauge_segs:
            current_seg = routed.sel(seg=current_seg)['up_seg'].values[()]
            if current_seg is None:
                break
        if current_seg is None:
            routed[upstream_var].loc[{'seg':seg}] = routed[downstream_var].loc[{'seg':seg}]
        else:
            upstream_gauge_seg = routed.sel(seg=current_seg)['up_seg'].values[()]
            routed[upstream_var].loc[{'seg':seg}] = map_var.sel(seg=upstream_gauge_seg).values[:]
            
    routed[downstream_var] = (routed[downstream_var].where(
        ~np.isnan(routed[downstream_var]).any(dim='time'), other=routed[upstream_var]))
    routed[upstream_var] = (routed[upstream_var].where(
        ~np.isnan(routed[upstream_var]).any(dim='time'), other=routed[downstream_var]))
    
    return routed

