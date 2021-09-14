import bmorph
import pandas as pd
import numpy as np
import xarray as xr
from functools import partial
from tqdm import tqdm
from bmorph.util import mizuroute_utils as mizutil
import os

def apply_bmorph(raw_ts, train_ts, ref_ts,
        apply_window, raw_train_window, ref_train_window,
        condition_ts=None,
        raw_y=None, train_y=None, ref_y=None,
        interval=pd.DateOffset(years=1),
        overlap=60, n_smooth_long=None, n_smooth_short=5,
        bw=3, xbins=200, ybins=10,
        rtol=1e-6, atol=1e-8, method='hist', train_cdf_min=1e-6, **kwargs):
    """Bias correction is performed by bmorph on user-defined intervals.

    Parameters
    ----------
    raw_ts : pandas.Series
        Raw flow timeseries.
    train_ts : pandas.Series
        Flow timeseries to train the bias correction model with.
    ref_ts : pandas.Series
        Observed/reference flow timeseries.
    condition_ts: pandas.Series
        A timeseries with a variable to condition on. This will be
        used in place of `raw_y`, `train_y`, and `ref_y`. It is
        mainly added as a convenience over specifying the same
        timeseries for each of those variables.
    apply_window : pandas.date_range
        Date range to apply bmorph onto flow timeseries.
    raw_train_window : pandas.date_range
        Date range to train the bias correction model.
    ref_train_window : pandas.date_range
        Date range to smooth elements in 'raw_ts' and 'bmorph_ts'.
    raw_y : pandas.Series, optional
        Raw time series of the second time series variable for conditioning.
    train_y : pandas.Series, optional
        Training second time series.
    ref_y : pandas.Series, optional
        Target second time series.
    interval : pandas.DateOffset
        Difference between bmorph application intervals.
    overlap : int
        Total number of days overlap CDF windows have with each other,
        distributed evenly before and after the application window.
    n_smooth_long : int, optional
        Number of elements that will be smoothed in `raw_ts` and `bmorph_ts`.
        The nsmooth value in this case is typically much larger than the one
        used for the bmorph function itself. For example, 365 days.
    n_smooth_short : int, optional
        Number of elements that will be smoothed when determining CDFs used
        for the bmorph function itself.
    bw : int, optional
        Bandwidth for KernelDensity. This should only be used if `method='kde'`.
    xbins : int, optional
        Bins for the first time series. This should only be used if `method='hist'`.
    ybins : int, optional
        Bins for the second time series. This should only be used if `method='hist'`.
    rtol : float, optional
        The desired relatie tolerance of the result for KernelDensity.
        This should only be used if `method='kde'`.
    atol : float, optional
        The desired absolute tolerance of the result for KernelDensity.
        This should only be used if `method='kde'`.
    method : str
        Method to use for conditioning. Currently 'hist' using hist2D and 'kde'
        using kde2D are the only supported methods.
    **kwargs:
        Additional keyword arguments. Mainly implemented for cross-compatibility with
        other methods so that a unified configuration can be used

    Returns
    -------
    bmorph_corr_ts : pandas.Series
        Returns a time series of length of an interval in the bmoprh window
        with bmorphed values.
    bmorph_multipliers : pandas.Series
        Returns a time series of equal length to bc_totals used to scale the
        raw flow values into the bmorphed values returned in bc_totals.
    """
    assert isinstance(interval, pd.DateOffset)

    if condition_ts is not None:
        raw_y = condition_ts
        ref_y = condition_ts
        train_y = condition_ts

    if interval == pd.DateOffset(days=1):
        raise Exception("Please enter an interval greater than 1 day(s)")

    raw_train_window = slice(*raw_train_window)
    apply_window = slice(*apply_window)
    ref_train_window = slice(*ref_train_window)
    raw_ts_window = slice(pd.to_datetime(raw_ts.index.values[0]),
                          pd.to_datetime(raw_ts.index.values[-1]))

    overlap_period = int(overlap / 2)
    bmorph_ts = pd.Series([])
    bmorph_multipliers = pd.Series([])
    bmorph_range = pd.date_range(apply_window.start,
                                 apply_window.stop+interval,
                                 freq=interval)
    for i in range(0,len(bmorph_range)-1):
        bmorph_start = bmorph_range[i]
        bmorph_end = bmorph_range[i+1]

        raw_apply_window =  slice(pd.to_datetime(str(bmorph_start)),
                                   pd.to_datetime(str(bmorph_end)))
        raw_cdf_window = slice(
                pd.to_datetime(str(bmorph_start - pd.DateOffset(days=overlap_period))),
                pd.to_datetime(str(bmorph_end + pd.DateOffset(days=overlap_period))))
        # No overlap for the first/last periods
        if ((raw_cdf_window.start < raw_ts_window.start) or (raw_cdf_window.stop > raw_ts_window.stop)):
            raw_cdf_window = slice(raw_ts_window.start, raw_ts_window.stop)

        # Perform the bias correction
        bc_total, bc_mult = bmorph.bmorph(
                raw_ts, train_ts, ref_ts,
                raw_apply_window, raw_train_window, ref_train_window, raw_cdf_window,
                raw_y, ref_y, train_y,
                n_smooth_short, bw=bw, xbins=xbins, ybins=ybins, rtol=rtol, atol=atol,
                method=method, train_cdf_min=train_cdf_min)
        bmorph_ts = bmorph_ts.append(bc_total)
        bmorph_multipliers = bmorph_multipliers.append(bc_mult)

    # Eliminate duplicate timestamps because of overlapping period
    bmorph_ts = bmorph_ts.groupby(bmorph_ts.index).mean()
    bmorph_multipliers = bmorph_multipliers.groupby(bmorph_multipliers.index).mean()

    # Apply the correction to the change in mean
    if n_smooth_long:
        nrni_mean = ref_ts[ref_train_window].mean()
        train_mean = train_ts[raw_train_window].mean()
        bmorph_corr_ts, corr_ts = bmorph.bmorph_correct(raw_ts, bmorph_ts, raw_ts_window,
                                                        nrni_mean, train_mean, n_smooth_long)
    else:
        bmorph_corr_ts = bmorph_ts
    return bmorph_corr_ts[apply_window], bmorph_multipliers[apply_window]


def apply_blendmorph(raw_upstream_ts, raw_downstream_ts,
                     train_upstream_ts, train_downstream_ts,
                     ref_upstream_ts, ref_downstream_ts,
                     apply_window, raw_train_window, ref_train_window,
                     interval, overlap, blend_factor,
                     raw_upstream_y=None, raw_downstream_y=None,
                     train_upstream_y=None, train_downstream_y=None,
                     ref_upstream_y=None, ref_downstream_y=None,
                     n_smooth_long=None, n_smooth_short=5,
                     bw=3, xbins=200, ybins=10, rtol=1e-6, atol=1e-8, 
                     method='hist', train_cdf_min=1e-6, **kwargs):
    """Bias correction performed by blending bmorphed flows on user defined intervals.

    Blendmorph is used to perform spatially consistent bias correction, this function
    does so on a user-defined interval. This is done by performing bmorph bias correction
    for each site's timeseries according to upstream and downstream gauge sites
    (or proxies) where true flows are known. The upstream and downstream corrected
    timeseries are then multiplied by fractional weights, `blend_factor`, that sum
    to 1 between them so the corrected flows can be combined, or "blended," into one,
    representative corrected flow series for the site. It is thereby important to specify
    upstream and downstream values so bias corrections are performed with values that
    most closely represent each site being corrected.

    Parameters
    ---------
    raw_upstream_ts : pandas.Series
        Raw flow timeseries corresponding to the upstream flows.
    raw_downstream_ts : pandas.Series
        Raw flow timerseries corresponding to the downstream flows.
    train_upstream_ts : pandas.Series
        Flow timeseries to train the bias correction model with for the upstream flows.
    train_downstream_ts : pandas.Series
        Flow timeseries to train the bias correction model with for the downstream flows.
    ref_upstream_ts : pandas.Series
        Observed/reference flow timeseries corresponding to the upstream flows.
    ref_downstream_ts : pandas.Series
        Observed/reference flow timeseries corresponding to the downstream flows.
    raw_train_window : pandas.date_range
        Date range to train the bias correction model.
    apply_window : pandas.date_range
        Date range to apply bmorph onto flow timeseries.
    ref_train_window : pandas.date_range
        Date range to smooth elements in 'raw_ts' and 'bmorph_ts'.
    interval : pandas.DateOffset
        Difference between bmorph application intervals.
    overlap: int
        Total overlap in number of days the CDF windows have with each other,
        distributed evenly before and after the application window.
    blend_factor : numpy.array
        An array determining how upstream and downstream bmorphing is proportioned.
        This is determined by the fill_method used in mizuroute_utils. The blend_factor
        entries are the proportion of upstream multiplers and totals added with
        1-blend_factor of downstream multipliers and totals.
    n_smooth_long : int, optional
        Number of elements that will be smoothed in `raw_ts` and `bmorph_ts`.
        The nsmooth value in this case is typically much larger than the one
        used for the bmorph function itself. For example, 365 days.
    n_smooth_short : int
        Number of elements that will be smoothed when determining CDFs used
        for the bmorph function itself.
    raw_upstream_y : pandas.Series, optional
        Raw time series of the second time series variable for conditioning corresponding
        to upstream flows.
    raw_downstream_y : pandas.Series, optional
        Raw time series of the second time series variable for conditioning corresponding
        to downstream flows.
    train_upstream_y : pandas.Series, optional
        Training second time series variable for conditioning correpsonding to downstream flows.
    train_downstream_y : pandas.Series, optional
        Training second time series variable for conditioning correpsonding to upstream flows.
    ref_upstream_y : pandas.Series, optional
        Target second time series variable for conditioning corresponding to upstream flows.
    ref_downstream_y : pandas.Series, optional
        Target second time series variable for conditioning corresponding to downtream flows.
    bw : int, optional
        Bandwidth for KernelDensity. This should only be used if `method='kde'`.
    xbins : int, optional
        Bins for the first time series. This should only be used if `method='hist'`.
    ybins : int, optional
        Bins for the second time series. This should only be used if `method='hist'`.
    rtol : float, optional
        The desired relatie tolerance of the result for KernelDensity.
        This should only be used if `method='kde'`.
    atol : float, optional
        The desired absolute tolerance of the result for KernelDensity.
        This should only be used if `method='kde'`.
    method : str, optional
        Method to use for conditioning. Currently 'hist' using hist2D and 'kde'
        using kde2D are the only supported methods.
    **kwargs:
        Additional keyword arguments. Mainly implemented for cross-compatibility with
        other methods so that a unified configuration can be used

    Returns
    -------
    bc_totals : pandas.Series
        Returns a time series of length of an interval in the bmoprh window
        with bmorphed values.
    bc_multipliers : pandas.Series
        Returns a time series of equal length to bc_totals used to scale the
        raw flow values into the bmorphed values returned in bc_totals.
    """
    assert isinstance(interval, pd.DateOffset)

    if interval == pd.DateOffset(days=1):
        raise Exception("Please enter a interval greater than 1 day(s)")

    bc_multipliers = pd.Series([])
    bc_totals = pd.Series([])

    apply_window = slice(*apply_window)
    raw_train_window = slice(*raw_train_window)
    ref_train_window = slice(*ref_train_window)
    raw_ts_window = slice(pd.to_datetime(raw_upstream_ts.index.values[0]),
                          pd.to_datetime(raw_upstream_ts.index.values[-1]))

    # Check if there is enough data input to run conditioning for both upstream
    # and downstream bmorphs. Boolean used here instead of later to make certain
    # both upstream and downstream use the same method and to minimze checks within
    # the for-loop
    run_mdcd = False
    y_varlist = [raw_upstream_y, train_upstream_y, ref_upstream_y,
                 raw_downstream_y, train_downstream_y, ref_downstream_y]
    if np.any(list(map(lambda x: x is not None, y_varlist))):
        run_mdcd = True

    # bmorph the series
    overlap_period = int(overlap / 2)
    bmorph_ts = pd.Series([])
    bmorph_multipliers = pd.Series([])
    bmorph_range = pd.date_range(apply_window.start, apply_window.stop+interval,
                                 freq=interval)
    for i in range(0,len(bmorph_range)-1):
        bmorph_start = bmorph_range[i]
        bmorph_end = bmorph_range[i+1]

        # we don't need ot adjust for overlap if it is the last entry at i+1
        if i < len(bmorph_range)-2:
            bmorph_end -= pd.DateOffset(days=1)

        raw_apply_window =  slice(pd.to_datetime(str(bmorph_start)),
                                   pd.to_datetime(str(bmorph_end)))
        raw_cdf_window = slice(pd.to_datetime(str(bmorph_start - pd.DateOffset(days=overlap_period))),
                               pd.to_datetime(str(bmorph_end + pd.DateOffset(days=overlap_period))))
        if (raw_cdf_window.start < raw_ts_window.start):
            offset = raw_ts_window.start - raw_cdf_window.start
            raw_cdf_window = slice(raw_ts_window.start, raw_cdf_window.stop + offset)
        if(raw_cdf_window.stop > raw_ts_window.stop):
            offset = raw_ts_window.stop - raw_cdf_window.stop
            raw_cdf_window = slice(raw_cdf_window.start + offset, raw_ts_window.stop)

        # Upstream & Downstream bias correction
        if run_mdcd:
            bc_up_total, bc_up_mult = bmorph.bmorph(
                    raw_upstream_ts, train_upstream_ts, ref_upstream_ts,
                    raw_apply_window, raw_train_window, ref_train_window, raw_cdf_window,
                    raw_upstream_y, ref_upstream_y, train_upstream_y,
                    n_smooth_short, bw=bw, xbins=xbins, ybins=ybins, rtol=rtol, atol=atol,
                    method=method, train_cdf_min=train_cdf_min)
            bc_down_total, bc_down_mult = bmorph.bmorph(
                    raw_downstream_ts, train_downstream_ts, ref_downstream_ts,
                    raw_apply_window, raw_train_window, ref_train_window, raw_cdf_window,
                    raw_downstream_y, ref_downstream_y, train_downstream_y,
                    n_smooth_short, bw=bw, xbins=xbins, ybins=ybins, rtol=rtol, atol=atol,
                    method=method, train_cdf_min=train_cdf_min)
        else:
            bc_up_total, bc_up_mult = bmorph.bmorph(
                    raw_upstream_ts, train_upstream_ts, ref_upstream_ts,
                    raw_apply_window, raw_train_window, ref_train_window, raw_cdf_window,
                    n_smooth_short, train_cdf_min=train_cdf_min)
            bc_down_total, bc_down_mult = bmorph.bmorph(
                    raw_downstream_ts, train_downstream_ts, ref_downstream_ts,
                    raw_apply_window, raw_train_window, ref_train_window, raw_cdf_window,
                    n_smooth_short, train_cdf_min=train_cdf_min)

        bc_multiplier = (blend_factor * bc_up_mult) + ((1 - blend_factor) * bc_down_mult)
        bc_total = (blend_factor * bc_up_total) + ((1 - blend_factor) * bc_down_total)

        bc_multipliers = bc_multipliers.append(bc_multiplier)
        bc_totals = bc_totals.append(bc_total)

    bc_totals = bc_totals.groupby(bc_totals.index).mean()
    bc_multipliers = bc_multipliers.groupby(bc_multipliers.index).mean()

    # Apply the correction to preserve the mean change
    if n_smooth_long:
        raw_ts = (blend_factor * raw_upstream_ts) + ((1 - blend_factor) * raw_downstream_ts)
        ref_ts = (blend_factor * ref_upstream_ts) + ((1 - blend_factor) * ref_downstream_ts)
        train_ts = (blend_factor * train_upstream_ts) + ((1 - blend_factor) * train_downstream_ts)
        nrni_mean = ref_ts[ref_train_window].mean()
        train_mean = train_ts[raw_train_window].mean()
        bc_totals, corr_ts = bmorph.bmorph_correct(raw_ts, bc_totals, raw_ts_window,
                                                   nrni_mean, train_mean,
                                                   n_smooth_long)
        bc_multipliers *= corr_ts

    return bc_totals[apply_window], bc_multipliers[apply_window]


def _scbc_c_seg(ds, apply_window, raw_train_window, ref_train_window,
                interval, overlap, condition_var, **kwargs):
    up_raw_ts =    ds['IRFroutedRunoff'].to_series()
    up_train_ts =  ds['up_raw_flow'].to_series()
    up_ref_ts =    ds['up_ref_flow'].to_series()
    up_seg =  int( ds['up_ref_seg'].values[()])
    up_cond =      ds[f'up_{condition_var}'].to_series()
    dn_raw_ts =    ds['IRFroutedRunoff'].to_series()
    dn_train_ts =  ds['down_raw_flow'].to_series()
    dn_ref_ts =    ds['down_ref_flow'].to_series()
    dn_seg =  int( ds['down_ref_seg'].values[()])
    dn_cond =      ds[f'down_{condition_var}'].to_series()
    blend_factor = ds['cdf_blend_factor'].values[()]
    local_flow =   ds['dlayRunoff']

    scbc_c_flows, scbc_c_mults = apply_blendmorph(
        up_raw_ts, dn_raw_ts,
        up_train_ts, dn_train_ts,
        up_ref_ts, dn_ref_ts,
        apply_window, raw_train_window, ref_train_window,
        interval, overlap, blend_factor,
        raw_upstream_y=up_cond, raw_downstream_y=dn_cond,
        train_upstream_y=up_cond, train_downstream_y=dn_cond,
        ref_upstream_y=up_cond, ref_downstream_y=dn_cond, **kwargs)

    scbc_c_locals = scbc_c_mults * local_flow.sel(time=scbc_c_mults.index)
    return scbc_c_flows, scbc_c_mults, scbc_c_locals


def _scbc_u_seg(ds, apply_window, raw_train_window, ref_train_window,
               interval, overlap, condition_var=None, **kwargs):
    up_raw_ts =    ds['IRFroutedRunoff'].to_series()
    up_train_ts =  ds['up_raw_flow'].to_series()
    up_ref_ts =    ds['up_ref_flow'].to_series()
    up_seg =  int( ds['up_ref_seg'].values[()])
    dn_raw_ts =    ds['IRFroutedRunoff'].to_series()
    dn_train_ts =  ds['down_raw_flow'].to_series()
    dn_ref_ts =    ds['down_ref_flow'].to_series()
    dn_seg =  int( ds['down_ref_seg'].values[()])
    blend_factor = ds['cdf_blend_factor'].values[()]
    local_flow =   ds['dlayRunoff'].copy(deep=True)

    scbc_u_flows, scbc_u_mults = apply_blendmorph(
            up_raw_ts, dn_raw_ts,
            up_train_ts, dn_train_ts,
            up_ref_ts, dn_ref_ts,
            apply_window, raw_train_window, ref_train_window,
            interval, overlap, blend_factor, **kwargs)

    scbc_u_locals = scbc_u_mults * local_flow.sel(time=scbc_u_mults.index)
    return scbc_u_flows, scbc_u_mults, scbc_u_locals

def _scbc_pass(ds, apply_window, **kwargs):
    """
    Formats flows that are not to be bias corrected because they are
    from segements without hrus into a format compatible with flows
    that are bias corrected. This is to ensure all the data is still
    returned to the user, but that the flows that are not bias
    corrected stay the same.    
    """
    raw_ts =    ds['IRFroutedRunoff'].to_series()
    local_flow =   ds['dlayRunoff']

    # taken from apply_blendmorph
    apply_window = slice(*apply_window)
    # multipliers of 1 will represent no change
    pass_mults = 0*local_flow.sel(time=apply_window).to_pandas() + 1
    # properly format the lcoals
    pass_locals = pass_mults * local_flow.sel(time=pass_mults.index)
    return raw_ts, pass_mults, pass_locals


def apply_scbc(ds, mizuroute_exe, bmorph_config, client=None, save_mults=False):
    """
    Applies Spatially Consistent Bias Correction (SCBC) by
    bias correcting local flows and re-routing them through
    mizuroute. This method can be run in parallel by providing
    a `dask client`.

    Parameters
    ----------
    ds: xr.Dataset
        An xarray dataset containing `time` and `seg` dimensions and
        variables to be bias corrected. This will mostly likely come from
        the provided preprocessing utility, `mizuroute_utils.to_bmorph`
    mizuroute_exe: str
        The path to the mizuroute executable
    bmorph_config: dict
        The configuration for the bias correction. See the documentation
        on input specifications and selecting bias correction techniques
        for descriptions of the options and their choices.
    client: dask.Client (optional)
        A `client` object to manage parallel computation.
    save_mults: boolean (optional)
        Whether to save multipliers from bmorph for diagnosis. If True,
        multipliers are saved in the same directory as local flows. Defaults
        as False to not save multipliers.

    Returns
    -------
    region_totals: xr.Dataset
        The rerouted, total, bias corrected flows for the region
    """
    def unpack_and_write_netcdf(results, segs, file_path, out_varname='scbc_flow', mult_path=None):
        flows = [r[0] for r in results]
        mults = [r[1] for r in results]
        local = [r[2] for r in results]
        local_ds = xr.DataArray(np.vstack(local), dims=('seg','time'))
        local_ds['seg'] = segs
        local_ds['time'] = flows[0].index
        local_ds.name = out_varname
        local_ds = local_ds.where(local_ds >= 1e-4, other=1e-4)
        try:
            os.remove(file_path)
        except OSError:
            pass
        local_ds.transpose().to_netcdf(file_path)

        if mult_path:
            try:
                os.remove(mult_path)
            except OSError:
                pass
            mult_ds = xr.DataArray(np.vstack(mults), dims=('seg','time'))
            mult_ds['seg'] = segs
            mult_ds['time'] = flows[0].index
            mult_ds.name = 'mults'
            mult_ds.transpose().to_netcdf(mult_path)

    if 'condition_var' in bmorph_config.keys() and bmorph_config['condition_var']:
        scbc_type = 'conditional'
        scbc_fun = partial(_scbc_c_seg, **bmorph_config)
    else:
        scbc_type = 'univariate'
        scbc_fun = partial(_scbc_u_seg, **bmorph_config)

    sel_vars = ['IRFroutedRunoff',
                'up_raw_flow',
                'up_ref_flow',
                'up_ref_seg',
                'IRFroutedRunoff',
                'down_raw_flow',
                'down_ref_flow',
                'down_ref_seg',
                'cdf_blend_factor',
                'dlayRunoff']
    if scbc_type == 'conditional':
        sel_vars.append(f'down_{bmorph_config["condition_var"]}')
        sel_vars.append(f'up_{bmorph_config["condition_var"]}')

    # select out segs that have an hru
    # and prep the pass function if there are
    # segs without an hru
    bc_segs_idx = np.where(ds['has_hru'])[0]
    if len(bc_segs_idx) != len(ds['seg'].values):
        scbc_pass_fun = partial(_scbc_pass, **bmorph_config)

    if client:
        big_futures =  [client.scatter(ds[sel_vars].sel(seg=seg)) for seg in tqdm(ds['seg'].values)]
        futures = []
        for bf, seg_idx in zip(big_futures, np.arange(len(ds['seg'].values))):
            if seg_idx in bc_segs_idx:
                futures.append(client.submit(scbc_fun, bf))
            else:
                futures.append(client.submit(scbc_pass_fun, bf))
        results = client.gather(futures)
    else:
        results = []
        for seg_idx in tqdm(np.arange(len(ds['seg'].values))):
            if seg_idx in bc_segs_idx:
                results.append(scbc_fun(ds.isel(seg=seg_idx)))
            else:
                results.append(scbc_pass_fun(ds.isel(seg=seg_idx)))

    out_file = (f'{bmorph_config["data_path"]}/input/'
                f'{bmorph_config["output_prefix"]}_local_{scbc_type}_scbc.nc')
    if save_mults: 
        mult_file = (f'{bmorph_config["data_path"]}/input/'
                f'{bmorph_config["output_prefix"]}_local_mult_{scbc_type}_scbc.nc')
    else:
        mult_file = None
    unpack_and_write_netcdf(results, ds['seg'], out_file, mult_path=mult_file)
    config_path, mizuroute_config = mizutil.write_mizuroute_config(
            bmorph_config["output_prefix"],
            scbc_type, bmorph_config['apply_window'],
            config_dir=bmorph_config['data_path']+'/mizuroute_configs/',
            topo_dir=bmorph_config['data_path']+'/topologies/',
            input_dir=bmorph_config['data_path']+'/input/',
            output_dir=bmorph_config['data_path']+bmorph_config['output_sub'],
            )

    mizutil.run_mizuroute(mizuroute_exe, config_path)
    region_totals = xr.open_mfdataset(f'{mizuroute_config["output_dir"]}{bmorph_config["output_prefix"]}_{scbc_type}_scbc*', engine='netcdf4')
    region_totals = region_totals.sel(time=slice(*bmorph_config['apply_window']))
    region_totals['seg'] = region_totals['reachID'].isel(time=0)
    return region_totals.load()

def bmorph_to_dataarray(dict_flows, name):
    da = xr.DataArray(np.vstack(dict_flows.values()), dims=('site', 'time'))
    da['site'] = list(dict_flows.keys())
    da['time'] = list(dict_flows.values())[0].index
    da.name = name
    return da.transpose()
