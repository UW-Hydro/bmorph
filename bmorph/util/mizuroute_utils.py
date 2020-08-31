import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from glob import glob
import xarray as xr
import pandas as pd
import geopandas as gpd
import bmorph
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KernelDensity

def find_upstream(ds, seg):
    """
    """
    if ds.sel(seg=seg)['is_headwaters']:
        return None
    up_idx = np.argwhere(ds['down_seg'].values == seg).flatten()[0]
    up_seg = ds['seg'].isel(seg=up_idx).values[()]
    return up_seg

def walk_downstream(ds, start_seg):
    """
    """
    tot_length = 0.0
    current_seg = start_seg
    while (ds['down_seg'].sel(seg=current_seg).values[()] in ds['seg'].values 
          and not ds['is_gauge'].sel(seg=ds['down_seg'].sel(seg=current_seg).values[()]).values[()]):
        current_seg = ds['down_seg'].sel(seg=current_seg).values[()]
        tot_length += ds.sel(seg=current_seg)['length'].values[()]
    return tot_length

def walk_upstream(ds, start_seg):
    """
    """
    tot_length = 0.0
    current_seg = start_seg
    while (ds['up_seg'].sel(seg=current_seg).values[()] in ds['seg'].values 
          and not ds['is_gauge'].sel(seg=ds['up_seg'].sel(seg=current_seg).values[()]).values[()]):
        current_seg = ds['up_seg'].sel(seg=current_seg).values[()]
        tot_length += ds.sel(seg=current_seg)['length'].values[()]
    return tot_length

def annotate_reaches(routed: xr.Dataset, topology: xr.Dataset, reference: xr.Dataset, 
                     gauge_sites: list):
    """
    sets up the dataset to have upstream/downstream distances and the blend factors prepared
    """
    
    routed['length'] = topology['Length']
    routed['down_seg'] = topology['Tosegment']
    routed.load()
    
    reference['time'] = reference['time']
    reference = reference.sel(outlet=gauge_sites)
    site_2_seg = {site: reference.sel(outlet=site)['seg'].values[()] for site in gauge_sites}
    seg_2_site = {reference.sel(outlet=site)['seg'].values[()]: site for site in gauge_sites}
    
    t_orig = reference.time.values[[0, -1]]
    t_corrected = routed.time.values[[0, -1]]
    t_start    = np.max([t_orig[0], t_corrected[0]])
    t_finish   = np.min([t_orig[1], t_corrected[1]])
    t_slice = slice(t_start, t_finish)

    routed = routed.sel(time=t_slice)
    reference = reference.sel(time=t_slice)
    
    routed['is_headwaters'] = False * routed['seg']
    headwaters = [s not in routed['down_seg'].values for s in routed['seg'].values]
    routed['is_headwaters'].values = headwaters

    routed['is_gauge'] = False * routed['seg']
    is_gauge = []
    for s in routed['seg']:
        if s in list(site_2_seg.values()):
            is_gauge.append(True)
        else:
            is_gauge.append(False)

    routed['is_gauge'].values[:]  = is_gauge
    
    routed['distance_to_up_gauge'] = 0 * routed['is_headwaters']
    routed['distance_to_down_gauge'] = 0 * routed['is_headwaters']
    routed['cdf_blend_factor'] = 0 * routed['is_headwaters']
    routed['up_seg'] = 0 * routed['is_headwaters']
    routed['upstream_ref_seg'] = np.nan * routed['is_headwaters']
    routed['downstream_ref_seg'] = np.nan * routed['is_headwaters']
    
    routed['up_seg'].values = [find_upstream(routed, s) for s in routed['seg'].values]
    routed['distance_to_up_gauge'].values =[walk_upstream(routed, s) for s in routed['seg']]
    routed['distance_to_down_gauge'].values = [walk_downstream(routed, s) for s in routed['seg']]
    routed['cdf_blend_factor'].values = (routed['distance_to_up_gauge'] 
                                         / (routed['distance_to_up_gauge'] + routed['distance_to_down_gauge'])).values
    routed['cdf_blend_factor'] = routed['cdf_blend_factor'].where(~np.isnan(routed['cdf_blend_factor']), other=0.0)
    
    for site, seg in site_2_seg.items():
        routed['downstream_ref_seg'].loc[{'seg':seg}] = seg
        routed['upstream_ref_seg'].loc[{'seg':seg}] = seg
    
    for seg in routed['seg'].values:
        if seg in seg_2_site.keys() or routed.sel(seg=seg)['down_seg'].values[()] not in routed['seg'].values:
            continue
        current_seg = seg
        while routed.sel(seg=current_seg)['down_seg'].values[()] not in seg_2_site.keys():
            current_seg = routed.sel(seg=current_seg)['down_seg'].values[()]
        downstream_gauge_seg = routed.sel(seg=current_seg)['down_seg'].values[()]
        routed['downstream_ref_seg'].loc[{'seg': seg}] = downstream_gauge_seg
        
    for seg in tqdm(routed['seg'].values):
        if seg in seg_2_site.keys() or routed.sel(seg=seg)['up_seg'].values[()] not in routed['seg'].values:
            continue
        current_seg = seg
        while routed.sel(seg=current_seg)['up_seg'].values[()] not in seg_2_site.keys():
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
    
def annotate_variable(routed: xr.Dataset, condition: xr.Dataset, var_label: str, condition_var: str, 
                      gauge_segs = list):
    """
    splits the variable into its upstream and downstream components
    """
    downstream_var = f'downstream_{var_label}'
    upstream_var = f'upstream_{var_label}'
    
    routed[downstream_var] = np.nan * condition[condition_var]
    routed[upstream_var] = np.nan * condition[condition_var]
    
    routed[downstream_var].load()
    routed[upstream_var].load()
    
    for seg in gauge_segs:
        routed[downstream_var].loc[{'seg':seg}] = condition[condition_var].sel(seg=seg).values[:]
        
    for seg in routed['seg'].values:
        if seg in gauge_segs or routed.sel(seg=seg)['down_seg'].values[()] not in routed['seg'].values:
            continue
        current_seg = seg
        while routed.sel(seg=current_seg)['up_seg'].values[()] not in gauge_segs:
            current_seg = routed.sel(seg=current_seg)['down_seg'].values[()]
        downstream_gauge_seg = routed.sel(seg=current_seg)['down_seg'].values[()]
        routed[downstream_var].loc[{'seg':seg}] = condition[condition_var].sel(seg=downstream_gauge_seg).values[:]
    
    for seg in routed[seg].values:
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
            routed[upstream_var].loc[{'seg':seg}] = condition[condition_var].sel(seg=upstream_gauge_seg).values[:]
            
    routed[downstream_var] = (routed[downstream_var].where(
        ~np.isnan(routed[downstream_var]).any(dim='time'), other=routed[upstream_var]))
    routed[upstream_var] = (routed[upstream_var].where(
        ~np.isnan(routed[upstream_var]).any(dim='time'), other=routed[downstream_var]))
    
    return routed

