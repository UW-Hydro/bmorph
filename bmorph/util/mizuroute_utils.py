import os
from glob import glob
import xarray as xr
import pandas as pd
import geopandas as gpd
import bmorph
import numpy as np
from scipy.stats import entropy
from string import Template
import subprocess

CONTROL_TEMPLATE = Template(
"""<ancil_dir>            $ancil_dir !
<input_dir>            $input_dir !
<output_dir>           $output_dir !
<sim_start>            $sim_start !
<sim_end>              $sim_end !
<fname_ntopOld>        $topo_file !
<dname_nhru>           seg !
<dname_sseg>           seg !
<seg_outlet>           -9999 !
<fname_qsim>           $flow_file !
<vname_qsim>           scbc_flow !
<vname_time>           time !
<dname_time>           time !
<dname_hruid>          seg !
<vname_hruid>          seg !
<units_qsim>           mm/d !
<dt_qsim>              86400 !
<is_remap>              F !
<route_opt>             1 !
<case_name>             $out_name !
<param_nml>             param.nml.default !
<doesBasinRoute>        0 !
<varname_area>          Contrib_Area !
<varname_length>        Length !
<varname_slope>         Slope !
<varname_HRUid>         seg_id !
<varname_segId>         seg_id !
<varname_downSegId>     Tosegment !
<varname_hruSegId>      seg_id !
""")


def write_mizuroute_config(region, scbc_type, time_window,
                           config_dir='../mizuroute_configs/',
                           topo_dir='../topologies/',
                           input_dir='../input/',
                           output_dir='../output/'):
    mizuroute_config = {
        'ancil_dir': os.path.abspath(topo_dir)+'/',
        'input_dir': os.path.abspath(input_dir)+'/',
        'output_dir': os.path.abspath(output_dir)+'/',
        'sim_start': time_window[0].strftime("%Y-%m-%d"),
        'sim_end': time_window[1].strftime("%Y-%m-%d"),
        'topo_file': f'{region}_huc12_topology_scaled_area.nc',
        'flow_file': f'{region}_local_{scbc_type}_scbc.nc',
        'out_name': f'{region}_{scbc_type}_scbc'
    }

    config_path = os.path.abspath(f'{config_dir}reroute_{region}_{scbc_type}.control')
    with open(config_path, 'w') as f:
        f.write(CONTROL_TEMPLATE.substitute(mizuroute_config))
    return config_path, mizuroute_config


def run_mizuroute(mizuroute_exe, mizuroute_config):
    cmd = f'{mizuroute_exe} {mizuroute_config}'
    p = subprocess.run([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p


def find_up(ds, seg):
    """
    Finds the segment directly upstream of seg given seg is not
    a headwater segment, (in which case np.nan is returned).

    Parameters
    ---------
    ds: xr.Dataset
        Dataset containing river segments as 'seg', headwater
        segments by 'is_headwaters', and what is downstream of
        each seg in 'down_seg'.
    seg: int
        River segment designation to search from.

    Returns
    -------
    up_seg: int
        Upstream segment designation found, or np.nan if seg
        is a headwater segement.
    """
    if ds.sel(seg=seg)['is_headwaters']:
        return np.nan
    up_idx = np.argwhere(ds['down_seg'].values == seg).flatten()[0]
    up_seg = ds['seg'].isel(seg=up_idx).values[()]
    return up_seg


def walk_down(ds, start_seg):
    """
    Finds the nearest downstream gauge site and returns the distance
    traveled to reach it from start_seg.

    Parameters
    ---------
    ds: xr.Dataset
        Dataset containing river segments, downstream segs, the length
        of the river segments, and which segs are gauge sites as
        'seg', 'down_seg', 'lenght', and 'is_gauge', respectively.
    start_seg: int
        River segment designation to start walking from to a
        downstream gauge site.

    Returns
    -------
    tot_length: float
        Total length traveled during walk, (e.g. cumulative river
        distance from start_seg to the downstream gauge site).
    cur_seg: int
        River segment designation of the gauge site reached.
    """
    tot_length = 0.0
    cur_seg = start_seg
    if ds['is_gauge'].sel(seg=cur_seg):
        return 0.0, cur_seg
    else:
        while (ds['down_seg'].sel(seg=cur_seg).values[()] in ds['seg'].values
              and not ds['is_gauge'].sel(seg=ds['seg'].sel(seg=cur_seg).values[()]).values[()]):
            cur_seg = ds['down_seg'].sel(seg=cur_seg).values[()]
            tot_length += ds.sel(seg=cur_seg)['length'].values[()]
        cur_seg = ds['down_seg'].sel(seg=cur_seg).values[()]
        return tot_length, cur_seg


def walk_up(ds, start_seg):
    """
    Finds the nearest upstream gauge site and returns the distance
    traveled to reach it from start_seg.

    Parameters
    ---------
    ds: xr.Dataset
        Dataset containing river segments, upstream segs, the length
        of the river segments, and which segs are gauge sites as
        'seg', 'up_seg', 'lenght', and 'is_gauge', respectively.
    start_seg: int
        River segment designation to start walking from to an
        upstream gauge site.

    Returns
    -------
    tot_length: float
        Total length traveled during walk, (e.g. cumulative river
        distance from start_seg to the downstream gauge site).
    cur_seg: int
        River segment designation of the gauge site reached.
    """
    tot_length = 0.0
    cur_seg = start_seg
    if ds['is_gauge'].sel(seg=cur_seg):
        return 0.0, cur_seg
    else:
        # assume flows are at the end of the reach, so if we are
        # walking upstream we will be walking through start_seg
        # and need to account for that
        tot_length += ds.sel(seg=cur_seg)['length'].values[()]
        while (ds['up_seg'].sel(seg=cur_seg).values[()] in ds['seg'].values
              and not ds['is_gauge'].sel(seg=ds['up_seg'].sel(seg=cur_seg).values[()]).values[()]):
            cur_seg = ds['up_seg'].sel(seg=cur_seg).values[()]
            tot_length += ds.sel(seg=cur_seg)['length'].values[()]
        cur_seg = ds['up_seg'].sel(seg=cur_seg).values[()]
        return tot_length, cur_seg


def find_max_r2(ds, curr_seg_flow):
    """
    Searches through ds to find which seg has the greatest
    r2 value with respect to curr_seg_flow. If no seg is found,
    max_r2 = 0 and max_r2_ref_seg = -1.

    Parameters
    ----------
    ds: xr.Dataset
        Contains the variable 'reference_flow' to compare
        curr_seg_flow against and the coordinate 'seg'.
    curr_seg_flow: np.array
        A numpy array containing flow values that r2 is to
        be maximized with respect to.

    Returns
    -------
    max_r2: float
        Magnitude of the maximum R squared value found.
    max_r2_ref_seg: int
        River segement designation corresponding to the max_r2.
    """
    max_r2 = 0.0
    max_r2_ref_seg = -1
    for ref_seg in ds['seg'].values:
        ref_flow = ds.sel(seg=ref_seg).values
        curr_ref_r2 = np.corrcoef(curr_seg_flow, ref_flow)[0, 1]**2
        if curr_ref_r2 > max_r2:
            max_r2 = curr_ref_r2
            max_r2_ref_seg = ref_seg
    return max_r2, max_r2_ref_seg


def find_min_kldiv(ds, curr_seg_flow):
    """
    Searches through ds to find which seg has the smallest
    Kullback-Leibler Divergence value with respect to curr_seg_flow.
    If no seg is found, min_kldiv = -1 and min_kldiv_ref_seg = -1.
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

    Parameters
    ----------
    ds: xr.Dataset
        contains the variable 'reference_flow' to compare
        curr_seg_flow against and the coordinate 'seg'.
    curr_seg_flow: np.array
        a numpy array containing flow values that KL Divergence
        is to be maximized with respect to.

    Returns
    -------
    min_kldiv: float
        Magnitude of the minimum KL Divergence found.
    min_kldiv_ref_seg: int
        River segment designation corresponding to min_kldiv.
    """
    TINY_VAL = 1e-6
    min_kldiv = np.inf
    min_kldiv_ref_seg = -1

    total_bins = int(np.sqrt(len(curr_seg_flow)))
    curr_seg_flow_pdf, curr_seg_flow_edges = np.histogram(
        curr_seg_flow, bins=total_bins, density=True)
    curr_seg_flow_pdf[curr_seg_flow_pdf == 0] = TINY_VAL

    for ref_seg in ds['seg'].values:
        ref_flow = ds.sel(seg=ref_seg).values
        ref_flow_pdf = np.histogram(ref_flow, bins=curr_seg_flow_edges, density=True)[0]
        ref_flow_pdf[ref_flow_pdf == 0] = TINY_VAL
        curr_ref_kldiv = entropy(pk=ref_flow_pdf, qk=curr_seg_flow_pdf)
        if curr_ref_kldiv < min_kldiv:
            min_kldiv = curr_ref_kldiv
            min_kldiv_ref_seg = ref_seg
    if min_kldiv == np.inf:
        # meaning something went wrong and kldiv cannot be used
        # to select refernce sites
        min_kldiv = -1
        # kl divergence can never be less than zero, so we can
        # trust that if a -1 pops up down the line, it is because
        # we set it here and something went wrong ... but we dont
        # want the method to break and stop running for other sites
    return min_kldiv, min_kldiv_ref_seg

def kling_gupta_efficiency(sim, obs):
    """
    Calculates the Kling-Gupta Efficiency (KGE) between two flow arrays.
    https://agrimetsoft.com/calculators/Kling-Gupta%20efficiency

    Parameters
    ---------
    sim: array-like
        Simulated flow array.
    obs: array-like
        Observed flow array.

    Returns
    -------
    kge: float
        Kling-Gupta Efficiency calculated between the two arrays.
    """
    obs = np.asarray(obs)
    sim = np.asarray(sim)
    obs_filtered = obs[np.logical_and(~np.isnan(obs), ~np.isnan(sim))]
    sim_filtered = sim[np.logical_and(~np.isnan(obs), ~np.isnan(sim))]
    sim_std = np.std(sim_filtered, ddof=1)
    obs_std = np.std(obs_filtered, ddof=1)
    sim_mu = np.mean(sim_filtered)
    obs_mu = np.mean(obs_filtered)
    r = np.corrcoef(sim_filtered, obs_filtered)[0, 1]
    var = sim_std / obs_std
    bias = sim_mu / obs_mu
    kge = 1 - np.sqrt((bias-1)**2 + (var-1)**2 + (r-1)**2)
    return kge

def find_max_kge(ds, curr_seg_flow):
    """
    Searches through ds to find which seg has the larges
    Kling-Gupta Efficiency (KGE) value with respect to curr_seg_flow.
    If no seg is found, max_kge = -np.inf and max_kge_ref_seg = -1.

    Parameters
    ----------
    ds: xr.Dataset
        Contains the variable 'reference_flow' to compare
        curr_seg_flow against and the coordinate 'seg'.
    curr_seg_flow: int
        A numpy array containing flow values that KGE
        is to be maximized with respect to.

    Returns
    -------
    max_kge: float
        Maximum KGE value found.
    max_kge_ref_seg
        River segment designation corresponding to max_kge.
    """
    max_kge = -np.inf
    max_kge_ref_seg = -1
    for ref_seg in ds['seg'].values:
        ref_flow = ds.sel(seg=ref_seg).values
        curr_ref_kge = kling_gupta_efficiency(curr_seg_flow, ref_flow)
        if curr_ref_kge > max_kge:
            max_kge = curr_ref_kge
            max_kge_ref_seg = ref_seg
    return max_kge, max_kge_ref_seg


def trim_time(dataset_list: list):
    """
    Trims all times of the xr.Datasets in the list to the shortest timeseries.

    Parameters
    ----------
    dataset_list: List[xr.Dataset]
        Contains a list of xr.Datasets

    Returns
    -------
    list
        Contains a list in the same order as dataset_list except with all items
        in the list having the same start and end time.
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
    Adds contributing_area, average elevation, length, and down_seg to
    routed from topology.

    Parameters
    ---------
    routed: xr.Dataset
        Contains streamflow timeseries mapped to river segments denoted
        as 'seg'.
    topology: xr.Dataset
        Contains topological data of the watershed that routed's streamflow
        timeseries describe. River segment designations, lengths, and
        immeditate downstream segments are expected as 'seg', 'Length',
        and 'Tosegment'.
    Returns
    -------
    routed: xr.Dataset
        The input dataset routed updated with the topological data.
    """
    routed = routed.sel(seg=topology['seg'])
    #routed['contributing_area'] = topology['Contrib_Area']
    #routed['elevation'] = 0.5 * (topology['TopElev'] + topology['BotElev'])
    routed['length'] = topology['Length']
    routed['down_seg'] = topology['Tosegment']

    return routed


def map_ref_sites(routed: xr.Dataset, gauge_reference: xr.Dataset,
                    gauge_sites=None, route_var = 'IRFroutedRunoff',
                    fill_method='kldiv'):
    """
    Assigns segs within routed boolean 'is_gauge' "identifiers" and
    what each seg's upstream and downstream reference seg designations are.

    Parameters
    ----------
    routed: xr.Dataset
        Contains the input flow timeseries data.
    gauge_reference: xr.Dataset
        Contains reference flow timeseries data for the same watershed
        as the routed dataset.
    gauge_sites: list, optional
        If None, gauge_sites will be taken as all those listed in
        gauge_reference.
    route_var: str
        Variable name of flows used for fill_method purposes within routed.
        This is defaulted as 'IRFroutedRunoff'.
    fill_method: str
        While finding some upstream/downstream reference segs may be simple,
        (segs with 'is_gauge' = True are their own reference segs, others
        may be easy to find looking directly up or downstream), some river
        networks may have multiple options to select gauge sites and may fail
        to have upstream/downstream reference segs designated. 'fill_method'
        specifies how segs should be assigned upstream/downstream reference
        segs for bias correction if they are missed walking upstream or downstream.

        Currently supported methods:
            'leave_null'
                nothing is done to fill missing reference segs, np.nan values are
                replaced with a -1 seg designation and that's it
            'forward_fill'
                xarray's ffill method is used to fill in any np.nan values
            'r2'
                reference segs are selected based on which reference site that
                seg's flows has the greatest r2 value with
            'kldiv'
                reference segs are selected based on which reference site that
                seg's flows has the smallest KL Divergence value with
            'kge'
                reference segs are selected based on which reference site that
                seg's flows has the greatest KGE value with
    Returns
    -------
    routed: xr.Dataset
        Routed timeseries with reference gauge site river segments assigned to
        each river segement in the original routed.
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
            routed['down_ref_seg'].loc[{'seg': s}] = s
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
    if fill_method == 'leave_null':
        # since there should be no -1 segs from mizuroute, we can set nan's to -1 to acknowledge
        # that they have been addressed and still set them apart from the rest of the data
        routed['up_ref_seg'] = (routed['up_ref_seg'].where(~np.isnan(routed['up_ref_seg']), other=-1))
        routed['down_ref_seg'] = (routed['down_ref_seg'].where(~np.isnan(routed['down_ref_seg']), other=-1))
    elif fill_method == 'forward_fill':
        routed['up_ref_seg'] = (routed['up_ref_seg'].where(
            ~np.isnan(routed['up_ref_seg']), other=routed['down_ref_seg'])).ffill('seg')
        routed['down_ref_seg'] = (routed['down_ref_seg'].where(
            ~np.isnan(routed['down_ref_seg']), other=routed['up_ref_seg'])).ffill('seg')
    elif fill_method == 'r2':

        fill_up_isegs = np.where(np.isnan(routed['up_ref_seg'].values))[0]
        fill_down_isegs = np.where(np.isnan(routed['down_ref_seg'].values))[0]

        routed['r2_up_gauge'] = 0 * routed['is_gauge']
        routed['r2_down_gauge'] = 0 * routed['is_gauge']

        for curr_seg in routed['seg'].values:
            up_ref_seg = np.nan
            curr_seg_flow = routed[route_var].sel(seg=curr_seg).values
            if np.isnan(routed['up_ref_seg'].sel(seg=curr_seg).values):
                up_ref_r2, up_ref_seg = find_max_r2(routed[route_var].sel(seg=gauge_segs), curr_seg_flow)
                routed['r2_up_gauge'].loc[{'seg':curr_seg}] = up_ref_r2
                routed['up_ref_seg'].loc[{'seg':curr_seg}] = up_ref_seg
            else:
                # this seg has already been filled in, but r2 still needs to be calculated
                ref_flow = routed[route_var].sel(seg=routed['up_ref_seg'].sel(seg=curr_seg)).values
                up_ref_r2 = np.corrcoef(curr_seg_flow, ref_flow)[0, 1]**2
                routed['r2_up_gauge'].loc[{'seg':curr_seg}] = up_ref_r2

        for curr_seg in routed['seg'].values:
            down_ref_seg = np.nan
            curr_seg_flow = routed[route_var].sel(seg=curr_seg).values
            if np.isnan(routed['down_ref_seg'].sel(seg=curr_seg).values):
                down_ref_r2, down_ref_seg = find_max_r2(routed[route_var].sel(seg=gauge_segs), curr_seg_flow)
                routed['r2_down_gauge'].loc[{'seg':curr_seg}] = down_ref_r2
                routed['down_ref_seg'].loc[{'seg':curr_seg}] = down_ref_seg
            else:
                # this seg has already been filled in, but r2 still needs to be calculated
                ref_flow = routed[route_var].sel(seg=routed['down_ref_seg'].sel(seg=curr_seg)).values
                down_ref_r2 = np.corrcoef(curr_seg_flow, ref_flow)[0, 1]**2
                routed['r2_down_gauge'].loc[{'seg':curr_seg}] = down_ref_r2


    elif fill_method == 'kldiv':
        fill_up_isegs = np.where(np.isnan(routed['up_ref_seg'].values))[0]
        fill_down_isegs = np.where(np.isnan(routed['down_ref_seg'].values))[0]

        routed['kldiv_up_gauge'] = 0 * routed['is_gauge']
        routed['kldiv_down_gauge'] = 0 * routed['is_gauge']
        for curr_seg in routed['seg'].values:
            curr_seg_flow = routed[route_var].sel(seg=curr_seg).values
            if np.isnan(routed['up_ref_seg'].sel(seg=curr_seg).values):
                up_ref_kldiv, up_ref_seg = find_min_kldiv(routed[route_var].sel(seg=gauge_segs), curr_seg_flow)
                routed['kldiv_up_gauge'].loc[{'seg':curr_seg}] = up_ref_kldiv
                routed['up_ref_seg'].loc[{'seg':curr_seg}] = up_ref_seg
            else:
                # this seg has already been filled in, but kldiv still needs to be calculated
                # kldiv computation could probably be gutted in the furture ...
                TINY_VAL = 1e-6
                total_bins = int(np.sqrt(len(curr_seg_flow)))
                curr_seg_flow_pdf, curr_seg_flow_edges = np.histogram(
                    curr_seg_flow, bins=total_bins, density=True)
                curr_seg_flow_pdf[curr_seg_flow_pdf == 0] = TINY_VAL

                ref_flow = routed[route_var].sel(seg=routed['up_ref_seg'].sel(seg=curr_seg).values).values
                ref_flow_pdf = np.histogram(ref_flow, bins=curr_seg_flow_edges, density=True)[0]
                ref_flow_pdf[ref_flow_pdf == 0] = TINY_VAL

                up_ref_kldiv = entropy(pk=ref_flow_pdf, qk=curr_seg_flow_pdf)
                routed['kldiv_up_gauge'].loc[{'seg':curr_seg}] = up_ref_kldiv

        for curr_seg in routed['seg'].values:
            curr_seg_flow = routed[route_var].sel(seg=curr_seg).values
            if np.isnan(routed['down_ref_seg'].sel(seg=curr_seg).values):
                down_ref_kldiv, down_ref_seg = find_min_kldiv(routed[route_var].sel(seg=gauge_segs), curr_seg_flow)
                routed['kldiv_down_gauge'].loc[{'seg':curr_seg}] = down_ref_kldiv
                routed['down_ref_seg'].loc[{'seg':curr_seg}] = down_ref_seg
            else:
                # this seg has already been filled in, but kldiv still needs to be calculated
                # kldiv computation could probably be gutted in the furture ...
                TINY_VAL = 1e-6
                total_bins = int(np.sqrt(len(curr_seg_flow)))
                curr_seg_flow_pdf, curr_seg_flow_edges = np.histogram(
                    curr_seg_flow, bins=total_bins, density=True)
                curr_seg_flow_pdf[curr_seg_flow_pdf == 0] = TINY_VAL

                ref_flow = routed[route_var].sel(seg=routed['down_ref_seg'].sel(seg=curr_seg).values).values
                ref_flow_pdf = np.histogram(ref_flow, bins=curr_seg_flow_edges, density=True)[0]
                ref_flow_pdf[ref_flow_pdf == 0] = TINY_VAL

                down_ref_kldiv = entropy(pk=ref_flow_pdf, qk=curr_seg_flow_pdf)
                routed['kldiv_down_gauge'].loc[{'seg':curr_seg}] = down_ref_kldiv

    elif fill_method == 'kge':

        fill_up_isegs = np.where(np.isnan(routed['up_ref_seg'].values))[0]
        fill_down_isegs = np.where(np.isnan(routed['down_ref_seg'].values))[0]

        routed['kge_up_gauge'] = 0 * routed['is_gauge']
        routed['kge_down_gauge'] = 0 * routed['is_gauge']

        for curr_seg in routed['seg'].values:
            up_ref_seg = np.nan
            curr_seg_flow = routed[route_var].sel(seg=curr_seg).values
            if np.isnan(routed['up_ref_seg'].sel(seg=curr_seg).values):
                up_ref_kge, up_ref_seg = find_max_kge(routed[route_var].sel(seg=gauge_segs), curr_seg_flow)
                routed['kge_up_gauge'].loc[{'seg':curr_seg}] = up_ref_kge
                routed['up_ref_seg'].loc[{'seg':curr_seg}] = up_ref_seg
            else:
                # this seg has already been filled in, but kge still needs to be calculated
                ref_flow = routed[route_var].sel(seg=routed['up_ref_seg'].sel(seg=curr_seg)).values
                up_ref_kge = kling_gupta_efficiency(curr_seg_flow, ref_flow)
                routed['kge_up_gauge'].loc[{'seg':curr_seg}] = up_ref_kge

        for curr_seg in routed['seg'].values:
            down_ref_seg = np.nan
            curr_seg_flow = routed[route_var].sel(seg=curr_seg).values
            if np.isnan(routed['down_ref_seg'].sel(seg=curr_seg).values):
                down_ref_kge, down_ref_seg = find_max_kge(routed[route_var].sel(seg=gauge_segs), curr_seg_flow)
                routed['kge_down_gauge'].loc[{'seg':curr_seg}] = down_ref_kge
                routed['down_ref_seg'].loc[{'seg':curr_seg}] = down_ref_seg
            else:
                # this seg has already been filled in, but kge still needs to be calculated
                ref_flow = routed[route_var].sel(seg=routed['down_ref_seg'].sel(seg=curr_seg)).values
                down_ref_kge = kling_gupta_efficiency(curr_seg_flow, ref_flow)
                routed['kge_down_gauge'].loc[{'seg':curr_seg}] = down_ref_kge
    else:
        raise ValueError('Invalid method provided for "fill_method"')

    return routed


def map_headwater_sites(routed: xr.Dataset):
    """
    Boolean identifies whether a river segement is a headwater with 'is_headwater'.

    Parameters
    ----------
    routed: xr.Dataset
        Contains watershed river segments designated as the dimension 'seg'.
        River segments are connected by referencing immediate downstream segments
        as 'down_seg' for each 'seg'.

    Returns
    -------
    routed: xr.Dataset
        The original routed dataset updated with which sites are headwaters.
    """
    if not 'down_seg' in list(routed.var()):
        raise Exception("Please denote down segs with 'down_seg'")

    routed['is_headwaters'] = False * routed['seg']
    headwaters = [s not in routed['down_seg'].values for s in routed['seg'].values]
    routed['is_headwaters'].values = headwaters

    return routed


def calculate_cdf_blend_factor(routed: xr.Dataset, gauge_reference: xr.Dataset,
                               gauge_sites=None, fill_method='kldiv'):
    """
    Calculates the cumulative distribution function blend factor based on distance
    to a seg's nearest up gauge site with respect to the total distance between
    the two closest guage sites to the seg.

    Parameters
    ----------
    routed: xr.Dataset
        Contains flow timeseries data.
    gauge_reference: xr.Dataset
        Contains reference flow timeseries data for the same watershed
        as the routed dataset.
    gauge_sites: list, optional
        If None, gauge_sites will be taken as all those listed in
        gauge_reference.
    fill_method: str
        See map_ref_sites for full description of how fill_method works.

        Because each fill_method selects reference segs differently, calculate_blend_vars
        needs to know how they were selected to create blend factors. Note that 'leave_null'
        is not supported for this method because there is no filling for this method.
        Currently supported:
            'forward_fill'
                cdf_blend_factor = distance_to_upstream /
                        (distance_to_upstream + distance_to_downstream)
            'kldiv'
                cdf_blend_factor = kldiv_upstream / (kldiv_upstream + kldiv_downstream)
            'r2'
                cdf_blend_factor = r2_upstream / (r2_upstream + r2_downstream)

    Returns
    -------
    routed: xr.Dataset
        The original routed dataset updated with 'cdf_blend_factors' used to combine
        upstream and downstream relative bias corrections. Each fill_method will also
        add or use upstream and downstream statistical measures calculated in map_ref_sites.
    """
    if not 'is_gauge' in list(routed.var()):
        # needed for walk_up and walk_down
        raise Exception("Please denote headwater segs with 'is_headwaters'")

    routed['cdf_blend_factor'] = 0 * routed['is_gauge']

    if fill_method == 'forward_fill':
        routed['distance_to_up_gauge'] = 0 * routed['is_gauge']
        routed['distance_to_down_gauge'] = 0 * routed['is_gauge']

        routed['distance_to_up_gauge'].values = [walk_up(routed, s)[0] for s in routed['seg']]
        routed['distance_to_down_gauge'].values = [walk_down(routed, s)[0] for s in routed['seg']]
        routed['cdf_blend_factor'].values = (routed['distance_to_up_gauge']
                                            / (routed['distance_to_up_gauge']
                                              + routed['distance_to_down_gauge'])).values
    else:
        if isinstance(gauge_sites, type(None)):
            gauge_sites = gauge_reference['site'].values
        else:
            # need to typecheck since we do a for loop later and don't
            # want to end up iterating through a string by accident
            assert isinstance(gauge_sites, list)

        if fill_method == 'kldiv':
            routed['cdf_blend_factor'].values = (routed['kldiv_down_gauge']
                                                 / (routed['kldiv_up_gauge']
                                                   + routed['kldiv_down_gauge'])).values
        elif fill_method == 'r2':
            routed['cdf_blend_factor'].values = (routed['r2_up_gauge']
                                                 / (routed['r2_up_gauge']
                                                   + routed['r2_down_gauge'])).values
        elif fill_method == 'kge':
            # since kge can be negative, the blend factor ratios
            # will use kge squared to ensure they don't cancel out
            raise Exception('kge is not currently supported, please select a different method')
            routed['cdf_blend_factor'].values = ((routed['kge_up_gauge']**2)
                                                 / ((routed['kge_up_gauge']**2)
                                                   + (routed['kge_down_gauge']**2))).values

    routed['cdf_blend_factor'] = routed['cdf_blend_factor'].where(~np.isnan(routed['cdf_blend_factor']), other=0.0)
    return routed

def calculate_blend_vars(routed: xr.Dataset, topology: xr.Dataset, reference: xr.Dataset,
                         gauge_sites = None, route_var = 'IRFroutedRunoff',
                         fill_method='kldiv'):
    """
    Calculates a number of variables used in blendmorph and map_var_to_seg.

    Parameters
    ----------
    routed: xr.Dataset
        The dataset that will be modified and returned ready for map_var_to_seg.
    topology: xr.Dataset
        Contains the network topology with a "seg" dimension that identifies reaches,
        matching the routed dataset.
    reference: xr.Dataset
        Contains reaches used for reference with dimension "site" and coordinate "seg".
    gauge_sites: list, optional
        Contains the gauge site names from the reference dataset to be used that are
        automatically pulled from reference if None are given.
    route_var: str
        Variable name of flows used for fill_method purposes within routed.
        This is defaulted as 'IRFroutedRunoff'.
    fill_method: str
        While finding some upstream/downstream reference segs may be simple,
        (segs with 'is_gauge' = True are their own reference segs, others
        may be easy to find looking directly up or downstream), some river
        networks may have multiple options to select gauge sites and may fail
        to have upstream/downstream reference segs designated. 'fill_method'
        specifies how segs should be assigned upstream/downstream reference
        segs for bias correction if they are missed walking upstream or downstream.

        Currently supported methods:
            'leave_null'
                nothing is done to fill missing reference segs, np.nan values are
                replaced with a -1 seg designation and that's it
            'forward_fill'
                xarray's ffill method is used to fill in any np.nan values
            'r2'
                reference segs are selected based on which reference site that
                seg's flows has the greatest r2 value with
            'kldiv' (default)
                reference segs are selected based on which reference site that
                seg's flows has the smallest KL Divergence value with
            'kge'
                reference segs are selected based on which reference site that
                seg's flows has the greatest KGE value with
    min_kge: float
        If not None, all upstream/downstream reference seg selections will be filtered
        according to the min_kge criteria, where seg selections that have a kge with
        the current seg that is less that min_kge will be set to -1 and determined
        unsuitable for bias correction. This is defaulted as -0.41.

    Returns
    -------
    routed: xr.Dataset
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
    routed = map_headwater_sites(routed=routed)
    routed = map_ref_sites(routed=routed, gauge_reference=reference,
                             gauge_sites = gauge_sites, route_var = route_var,
                             fill_method = fill_method)
    routed = calculate_cdf_blend_factor(routed=routed, gauge_reference=reference,
                             gauge_sites = gauge_sites, fill_method = fill_method)

    for seg in routed['seg']:
        # if one of the refernece sites has been left null or determined
        # non bias correcteable according to the fill methods, then both
        # reference sites should be considered so to prevent any weird
        # partial bias correction attemps
        up_ref_seg = routed['up_ref_seg'].sel(seg=seg)
        down_ref_seg = routed['down_ref_seg'].sel(seg=seg)

        if up_ref_seg == -1 or down_ref_seg == -1:
            routed['up_ref_seg'].loc[{'seg':seg}] = -1
            routed['down_ref_seg'].loc[{'seg':seg}] = -1

    return routed


def map_var_to_segs(routed: xr.Dataset, map_var: xr.DataArray, var_label: str,
                    gauge_segs = None):
    """
    Splits the variable into its up and down components to be used in blendmorph.

    Parameters
    ----------
    routed: xr.Dataset
        the dataset that will be modified and returned having been prepared by calculate_blend_vars
        with the dimension 'seg'
    map_var: xr.DataArray
        contains the variable to be split into up and down components and can be
        the same as routed, (must also contain the dimension 'seg')
    var_label: str
        suffix of the up and down parts of the variable
    gauge_segs: list, optional
        List of the gauge segs that identify the reaches that are gauge sites, pulled from routed
        if None.
    ----
    Returns
    -------
    routed: xr.Dataset
        with the following added:
        f'down_{var_label}'
        f'up_{var_label}'
    """
    down_var = f'down_{var_label}'
    up_var = f'up_{var_label}'
    # need to override dask array data protections
    map_var.load()
    routed[down_var] = np.nan * map_var
    routed[up_var] = np.nan * map_var

    for seg in routed['seg'].values:
        up_seg = routed['up_ref_seg'].sel(seg=seg)
        down_seg = routed['down_ref_seg'].sel(seg=seg)
        if up_seg != -1:
            routed[up_var].loc[{'seg': seg}] = map_var.sel(seg=up_seg).values[:]
        if down_seg != -1:
            routed[down_var].loc[{'seg': seg}] = map_var.sel(seg=down_seg).values[:]

    return routed


def map_met_hru_to_seg(met_hru, topo):
    """
    Maps meterological data from hru to seg.

    Parameters
    ----------
    met_hru: xr.Dataset
        A dataset of meteorological data to be mapped
        onto the stream segments to facilitate conditioning.
        All variables in this dataset will automatically be mapped
        onto the stream segments and returned.
    topo: xr.Dataset
        Topology dataset for running mizuRoute.
        We expect this to have ``seg`` and ``hru`` dimensions.

    Returns
    -------
    met_seg: xr.Dataset
        A dataset of meterological data mapped onto the stream
        segments to facilitate conditioning.
    """
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


def to_bmorph(topo: xr.Dataset, routed: xr.Dataset, reference: xr.Dataset,
                            met_hru: xr.Dataset=None, route_var: str='IRFroutedRunoff',
                            fill_method = 'kldiv'):

    '''
    Prepare mizuroute output for bias correction via the blendmorph algorithm. This
    allows an optional dataset of hru meteorological data to be given for conditional
    bias correction.

    Parameters
    ----------
    topo: xr.Dataset
        Topology dataset for running mizuRoute.
        We expect this to have ``seg`` and ``hru`` dimensions.
    routed: xr.Dataset
        The initially routed dataset from mizuRoute
    reference: xr.Dataset
        A dataset containing reference flows for bias correction.
        We expect this to have ``site`` and ``time`` dimensions.
    met_hru: xr.Dataset, optional
        A dataset of meteorological data to be mapped
        onto the stream segments to facilitate conditioning.
        All variables in this dataset will automatically be mapped
        onto the stream segments and returned.
    route_var: str
        Name of the variable of the routed runoff in the ``routed``
        dataset. Defaults to ``IRFroutedRunoff``.
    fill_method: str
        While finding some upstream/downstream reference segs may be simple,
        (segs with 'is_gauge' = True are their own reference segs, others
        may be easy to find looking directly up or downstream), some river
        networks may have multiple options to select gauge sites and may fail
        to have upstream/downstream reference segs designated. 'fill_method'
        specifies how segs should be assigned upstream/downstream reference
        segs for bias correction if they are missed walking upstream or downstream.

        Currently supported methods:
            'leave_null'
                nothing is done to fill missing reference segs, np.nan values are
                replaced with a -1 seg designation and that's it
            'forward_fill'
                xarray's ffill method is used to fill in any np.nan values
            'r2'
                reference segs are selected based on which reference site that
                seg's flows has the greatest r2 value with
            'kldiv' (default)
                reference segs are selected based on which reference site that
                seg's flows has the smallest KL Divergence value with
            'kge'
                reference segs are selected based on which reference site that
                seg's flows has the greatest KGE value with
    min_kge: float, optional
        See calculate_blend_vars for more information
        defaults None unless fill_method = 'kge'.

    Returns
    -------
    met_seg:
        A dataset with the required data for applying the ``blendmorph``
        routines. See the ``blendmorph`` documentation for further information.
    '''
    if fill_method == 'kge' and min_kge is None:
        min_kge = -0.41

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
    routed, reference, met_seg = trim_time([routed, reference, met_seg])
    routed = calculate_blend_vars(routed, topo, reference, route_var = route_var,
                                  fill_method = fill_method)

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

