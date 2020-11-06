import bmorph
import pandas as pd
import numpy as np
import xarray as xr
from functools import partial
from bmorph.util import mizuroute_utils as mizutil

def apply_annual_bmorph(raw_ts, train_ts, obs_ts,
        training_window, bmorph_window, reference_window,
        window_size, n_smooth_long=None, n_smooth_short=5, train_on_year=False,
        raw_y=None, train_y=None, obs_y=None, bw=3, xbins=200, ybins=10,
        rtol=1e-7, atol=0, method='hist'):

    training_window = slice(*training_window)
    bmorph_window = slice(*bmorph_window)
    reference_window = slice(*reference_window)
    raw_ts_window = slice(pd.to_datetime(raw_ts.index.values[0]),
                          pd.to_datetime(raw_ts.index.values[-1]))

    # bmorph the series
    overlap_period = int(window_size / 2)
    bmorph_ts = pd.Series([])
    bmorph_multipliers = pd.Series([])
    for year in range(bmorph_window.start.year, bmorph_window.stop.year+1):
        if train_on_year:
            training_window = slice(pd.to_datetime('{}-10-01 00:00:00'.format(year-1)),
                                   pd.to_datetime('{}-09-30 00:00:00'.format(year)))
        raw_bmorph_window =  slice(pd.to_datetime('{}-01-01 00:00:00'.format(year)),
                                   pd.to_datetime('{}-12-31 00:00:00'.format(year)))
        raw_cdf_window = slice(pd.to_datetime('{}-01-01 00:00:00'.format(year - overlap_period)),
                               pd.to_datetime('{}-12-31 00:00:00'.format(year + overlap_period)))
        if (raw_cdf_window.start < raw_ts_window.start):
            offset = raw_ts_window.start - raw_cdf_window.start
            raw_cdf_window = slice(raw_ts_window.start, raw_cdf_window.stop + offset)
        if(raw_cdf_window.stop > raw_ts_window.stop):
            offset = raw_ts_window.stop - raw_cdf_window.stop
            raw_cdf_window = slice(raw_cdf_window.start + offset, raw_ts_window.stop)

        bc_total, bc_mult = bmorph.bmorph(raw_ts, raw_cdf_window,raw_bmorph_window, obs_ts, train_ts,
                                          training_window, n_smooth_short, raw_y, obs_y, train_y,
                                          bw=bw, xbins=xbins, ybins=ybins, rtol=rtol, atol=atol,
                                          method=method)
        bmorph_ts = bmorph_ts.append(bc_total)
        bmorph_multipliers = bmorph_multipliers.append(bc_mult)


    # Apply the correction
    if n_smooth_long:
        nrni_mean = obs_ts[reference_window].mean()
        train_mean = train_ts[reference_window].mean()
        bmorph_corr_ts = bmorph.bmorph_correct(raw_ts, bmorph_ts, raw_ts_window,
                                               nrni_mean, train_mean,
                                               n_smooth_long)
    else:
        bmorph_corr_ts = bmorph_ts
    return bmorph_corr_ts, bmorph_multipliers

def apply_interval_bmorph(raw_ts, train_ts, obs_ts,
        training_window, bmorph_window, reference_window, bmorph_step,
        window_size, n_smooth_long=None, n_smooth_short=5,
        raw_y=None, train_y=None, obs_y=None, bw=3, xbins=200, ybins=10,
        rtol=1e-6, atol=1e-8, method='hist'):

    assert isinstance(bmorph_step, pd.DateOffset)

    if bmorph_step == pd.DateOffset(days=1):
        raise Exception("Please enter a bmorph_interval greater than 1 day(s)")

    training_window = slice(*training_window)
    bmorph_window = slice(*bmorph_window)
    reference_window = slice(*reference_window)
    raw_ts_window = slice(pd.to_datetime(raw_ts.index.values[0]),
                          pd.to_datetime(raw_ts.index.values[-1]))

    # bmorph the series
    overlap_period = int(window_size / 2)
    bmorph_ts = pd.Series([])
    bmorph_multipliers = pd.Series([])
    bmorph_range = pd.date_range(bmorph_window.start, bmorph_window.stop+bmorph_step,
                                 freq=bmorph_step)
    for i in range(0,len(bmorph_range)-1):
        bmorph_start = bmorph_range[i]
        bmorph_end = bmorph_range[i+1]

        # we don't need ot adjust for overlap if it is the last entry at i+1
        if i < len(bmorph_range)-2:
            bmorph_end -= pd.DateOffset(days=1)

        raw_bmorph_window =  slice(pd.to_datetime(str(bmorph_start)),
                                   pd.to_datetime(str(bmorph_end)))
        raw_cdf_window = slice(pd.to_datetime(str(bmorph_start - pd.DateOffset(years=overlap_period))),
                               pd.to_datetime(str(bmorph_end + pd.DateOffset(years=overlap_period))))
        if (raw_cdf_window.start < raw_ts_window.start):
            offset = raw_ts_window.start - raw_cdf_window.start
            raw_cdf_window = slice(raw_ts_window.start, raw_cdf_window.stop + offset)
        if(raw_cdf_window.stop > raw_ts_window.stop):
            offset = raw_ts_window.stop - raw_cdf_window.stop
            raw_cdf_window = slice(raw_cdf_window.start + offset, raw_ts_window.stop)

        bc_total, bc_mult = bmorph.bmorph(raw_ts, raw_cdf_window, raw_bmorph_window, obs_ts, train_ts,
                                          training_window, n_smooth_short, raw_y, obs_y, train_y,
                                          bw=bw, xbins=xbins, ybins=ybins, rtol=rtol, atol=atol,
                                          method=method)
        bmorph_ts = bmorph_ts.append(bc_total)
        bmorph_multipliers = bmorph_multipliers.append(bc_mult)


    # Apply the correction
    if n_smooth_long:
        nrni_mean = obs_ts[reference_window].mean()
        train_mean = train_ts[reference_window].mean()
        bmorph_corr_ts = bmorph.bmorph_correct(raw_ts, bmorph_ts, raw_ts_window,
                                               nrni_mean, train_mean,
                                               n_smooth_long)
    else:
        bmorph_corr_ts = bmorph_ts
    return bmorph_corr_ts, bmorph_multipliers

def apply_annual_blendmorph(raw_upstream_ts, raw_downstream_ts,
                            train_upstream_ts, train_downstream_ts,
                            truth_upstream_ts, truth_downstream_ts,
                            training_window, bmorph_window, reference_window, window_size,
                            blend_factor, n_smooth_long=None, n_smooth_short=5, train_on_year=False,
                            raw_upstream_y = None, raw_downstream_y = None,
                            train_upstream_y = None, train_downstream_y = None,
                            truth_upstream_y = None, truth_downstream_y = None,
                            bw=3, xbins=200, ybins=10, atol=0, rtol=1e-7, method='hist'):
    """
    Apply Annual bmorph Blending
        applies the bmorph bias correction and blends the multipliers
        computed by bmorph to produce a statistically bias corrected
        streamflow data set
    ----
    raw_upstream_ts: pd.Series
    raw_downstream_ts: pd.Series
    train_upstream_ts: pd.Series
    train_downstream_ts: pd.Series
    truth_upstream_ts: pd.Series
    truth_downstream_ts: pd.Series
    training_window: pd.date_range
    bmorph_window: pd.date_range
    reference_window: pd.date_range
    window_size: int
    blend_factor:
    n_smooth_long: int
    n_smooth_short: int
    train_on_year: boolean
    """
    bc_multipliers = pd.Series([])
    bc_totals = pd.Series([])

    training_window = slice(*training_window)
    bmorph_window = slice(*bmorph_window)
    reference_window = slice(*reference_window)
    raw_ts_window = slice(pd.to_datetime(raw_upstream_ts.index.values[0]),
                          pd.to_datetime(raw_upstream_ts.index.values[-1]))
    # Check if there is enough data input to run mdcdedcdfm for both upstream
    # and downstream bmorphs. Boolean used here instead of later to make certain
    # both upstream and downstream use the same method and to minimze checks within
    # the for-loop
    run_mdcd = False
    y_varlist = [raw_upstream_y, train_upstream_y, truth_upstream_y,
                 raw_downstream_y, train_downstream_y, truth_downstream_y]
    if np.any(list(map(lambda x: x is not None, y_varlist))):
        run_mdcd = True

    # bmorph the series
    overlap_period = int(window_size / 2)

    for year in range(bmorph_window.start.year, bmorph_window.stop.year+1):
        # set up annual windows
        if train_on_year:
            training_window = slice(pd.to_datetime('{}-10-01 00:00:00'.format(year-1)),
                                   pd.to_datetime('{}-09-30 00:00:00'.format(year)))
        raw_bmorph_window =  slice(pd.to_datetime('{}-01-01 00:00:00'.format(year)),
                                   pd.to_datetime('{}-12-31 00:00:00'.format(year)))
        raw_cdf_window = slice(pd.to_datetime('{}-01-01 00:00:00'.format(year - overlap_period)),
                               pd.to_datetime('{}-12-31 00:00:00'.format(year + overlap_period)))

        # account for offset in windows
        if (raw_cdf_window.start < raw_ts_window.start):
            offset = raw_ts_window.start - raw_cdf_window.start
            raw_cdf_window = slice(raw_ts_window.start, raw_cdf_window.stop + offset)
        if(raw_cdf_window.stop > raw_ts_window.stop):
            offset = raw_ts_window.stop - raw_cdf_window.stop
            raw_cdf_window = slice(raw_cdf_window.start + offset, raw_ts_window.stop)
        # Upstream & Downstream bias correction
        if run_mdcd:
            bc_up_total, bc_up_mult = bmorph.bmorph(raw_upstream_ts, raw_cdf_window,
                                                    raw_bmorph_window,
                                                    truth_upstream_ts, train_upstream_ts,
                                                    training_window, n_smooth_short,
                                                    raw_upstream_y, truth_upstream_y,
                                                    train_upstream_y, bw=bw, xbins=xbins,
                                                    ybins=ybins, rtol=rtol, atol=atol,
                                                    method=method)

            bc_down_total, bc_down_mult = bmorph.bmorph(raw_downstream_ts, raw_cdf_window,
                                                        raw_bmorph_window,
                                                        truth_downstream_ts, train_downstream_ts,
                                                        training_window, n_smooth_short,
                                                        raw_downstream_y, truth_downstream_y,
                                                        train_downstream_y, bw=bw, xbins=xbins,
                                                        ybins=ybins, rtol=rtol, atol=atol,
                                                        method=method)
        else:
            bc_up_total, bc_up_mult = bmorph.bmorph(raw_upstream_ts, raw_cdf_window,
                                                    raw_bmorph_window,
                                                    truth_upstream_ts, train_upstream_ts,
                                                    training_window, n_smooth_short)

            bc_down_total, bc_down_mult = bmorph.bmorph(raw_downstream_ts, raw_cdf_window,
                                                        raw_bmorph_window,
                                                        truth_downstream_ts, train_downstream_ts,
                                                       training_window, n_smooth_short)

        bc_multiplier = (blend_factor * bc_up_mult) + ((1 - blend_factor) * bc_down_mult)
        bc_total = (blend_factor * bc_up_total) + ((1 - blend_factor) * bc_down_total)

        bc_multipliers = bc_multipliers.append(bc_multiplier)
        bc_totals = bc_totals.append(bc_total)

    return bc_totals, bc_multipliers

def apply_interval_blendmorph(raw_upstream_ts, raw_downstream_ts,
                            train_upstream_ts, train_downstream_ts,
                            truth_upstream_ts, truth_downstream_ts,
                            training_window, bmorph_window, reference_window, bmorph_step, window_size,
                            blend_factor, n_smooth_long=None, n_smooth_short=5,
                            raw_upstream_y = None, raw_downstream_y = None,
                            train_upstream_y = None, train_downstream_y = None,
                            truth_upstream_y = None, truth_downstream_y = None,
                            bw=3, xbins=200, ybins=10, rtol=1e-6, atol=1e-8, method='hist'):

    assert isinstance(bmorph_step, pd.DateOffset)

    if bmorph_step == pd.DateOffset(days=1):
        raise Exception("Please enter a bmorph_interval greater than 1 day(s)")

    bc_multipliers = pd.Series([])
    bc_totals = pd.Series([])

    training_window = slice(*training_window)
    bmorph_window = slice(*bmorph_window)
    reference_window = slice(*reference_window)
    raw_ts_window = slice(pd.to_datetime(raw_upstream_ts.index.values[0]),
                          pd.to_datetime(raw_upstream_ts.index.values[-1]))

    # Check if there is enough data input to run mdcdedcdfm for both upstream
    # and downstream bmorphs. Boolean used here instead of later to make certain
    # both upstream and downstream use the same method and to minimze checks within
    # the for-loop
    run_mdcd = False
    y_varlist = [raw_upstream_y, train_upstream_y, truth_upstream_y,
                 raw_downstream_y, train_downstream_y, truth_downstream_y]
    if np.any(list(map(lambda x: x is not None, y_varlist))):
        run_mdcd = True

    # bmorph the series
    overlap_period = int(window_size / 2)
    bmorph_ts = pd.Series([])
    bmorph_multipliers = pd.Series([])
    bmorph_range = pd.date_range(bmorph_window.start, bmorph_window.stop+bmorph_step,
                                 freq=bmorph_step)
    for i in range(0,len(bmorph_range)-1):
        bmorph_start = bmorph_range[i]
        bmorph_end = bmorph_range[i+1]

        # we don't need ot adjust for overlap if it is the last entry at i+1
        if i < len(bmorph_range)-2:
            bmorph_end -= pd.DateOffset(days=1)

        raw_bmorph_window =  slice(pd.to_datetime(str(bmorph_start)),
                                   pd.to_datetime(str(bmorph_end)))
        raw_cdf_window = slice(pd.to_datetime(str(bmorph_start - pd.DateOffset(years=overlap_period))),
                               pd.to_datetime(str(bmorph_end + pd.DateOffset(years=overlap_period))))
        if (raw_cdf_window.start < raw_ts_window.start):
            offset = raw_ts_window.start - raw_cdf_window.start
            raw_cdf_window = slice(raw_ts_window.start, raw_cdf_window.stop + offset)
        if(raw_cdf_window.stop > raw_ts_window.stop):
            offset = raw_ts_window.stop - raw_cdf_window.stop
            raw_cdf_window = slice(raw_cdf_window.start + offset, raw_ts_window.stop)

        # Upstream & Downstream bias correction
        if run_mdcd:
            bc_up_total, bc_up_mult = bmorph.bmorph(raw_upstream_ts, raw_cdf_window,
                                                    raw_bmorph_window,
                                                    truth_upstream_ts, train_upstream_ts,
                                                    training_window, n_smooth_short,
                                                    raw_upstream_y, truth_upstream_y,
                                                    train_upstream_y, bw=bw, xbins=xbins,
                                                    ybins=ybins, rtol=rtol, atol=atol,
                                                    method=method)

            bc_down_total, bc_down_mult = bmorph.bmorph(raw_downstream_ts, raw_cdf_window,
                                                        raw_bmorph_window,
                                                        truth_downstream_ts, train_downstream_ts,
                                                        training_window, n_smooth_short,
                                                        raw_downstream_y, truth_downstream_y,
                                                        train_downstream_y, bw=bw, xbins=xbins,
                                                        ybins=ybins, rtol=rtol, atol=atol,
                                                        method=method)
        else:
            bc_up_total, bc_up_mult = bmorph.bmorph(raw_upstream_ts, raw_cdf_window,
                                                    raw_bmorph_window,
                                                    truth_upstream_ts, train_upstream_ts,
                                                    training_window, n_smooth_short)

            bc_down_total, bc_down_mult = bmorph.bmorph(raw_downstream_ts, raw_cdf_window,
                                                        raw_bmorph_window,
                                                        truth_downstream_ts, train_downstream_ts,
                                                       training_window, n_smooth_short)


        bc_multiplier = (blend_factor * bc_up_mult) + ((1 - blend_factor) * bc_down_mult)
        bc_total = (blend_factor * bc_up_total) + ((1 - blend_factor) * bc_down_total)

        bc_multipliers = bc_multipliers.append(bc_multiplier)
        bc_totals = bc_totals.append(bc_total)

    return bc_totals, bc_multipliers


def _scbc_c_seg(ds, train_window, bmorph_window, reference_window,
               bmorph_interval, bmorph_overlap, condition_var):
    up_raw_ts =    ds['IRFroutedRunoff'].to_series()
    up_train_ts =  ds['up_raw_flow'].to_series()
    up_obs_ts =    ds['up_ref_flow'].to_series()
    up_seg =  int( ds['up_ref_seg'].values[()])
    up_cond =      ds[f'up_{condition_var}'].to_series()
    dn_raw_ts =    ds['IRFroutedRunoff'].to_series()
    dn_train_ts =  ds['down_raw_flow'].to_series()
    dn_obs_ts =    ds['down_ref_flow'].to_series()
    dn_seg =  int( ds['down_ref_seg'].values[()])
    dn_cond =      ds[f'down_{condition_var}'].to_series()
    blend_factor = ds['cdf_blend_factor'].values[()]
    local_flow =   ds['dlayRunoff']

    scbc_c_flows, scbc_c_mults = apply_interval_blendmorph(
        up_raw_ts, dn_raw_ts,
        up_train_ts, dn_train_ts,
        up_obs_ts, dn_obs_ts,
        train_window, bmorph_window, reference_window,
        bmorph_interval, bmorph_overlap, blend_factor,
        raw_upstream_y=up_cond, raw_downstream_y=dn_cond,
        train_upstream_y=up_cond, train_downstream_y=dn_cond,
        truth_upstream_y=up_cond, truth_downstream_y=dn_cond)

    scbc_c_locals = scbc_c_mults * local_flow.sel(time=scbc_c_mults.index)
    return scbc_c_flows, scbc_c_mults, scbc_c_locals


def _scbc_u_seg(ds, train_window, bmorph_window, reference_window,
               bmorph_interval, bmorph_overlap, condition_var=None):
    up_raw_ts =    ds['IRFroutedRunoff'].to_series()
    up_train_ts =  ds['up_raw_flow'].to_series()
    up_obs_ts =    ds['up_ref_flow'].to_series()
    up_seg =  int( ds['up_ref_seg'].values[()])
    dn_raw_ts =    ds['IRFroutedRunoff'].to_series()
    dn_train_ts =  ds['down_raw_flow'].to_series()
    dn_obs_ts =    ds['down_ref_flow'].to_series()
    dn_seg =  int( ds['down_ref_seg'].values[()])
    blend_factor = ds['cdf_blend_factor'].values[()]
    local_flow =   ds['dlayRunoff']

    scbc_u_flows, scbc_u_mults = apply_interval_blendmorph(
            up_raw_ts, dn_raw_ts,
            up_train_ts, dn_train_ts,
            up_obs_ts, dn_obs_ts,
            train_window, bmorph_window, reference_window,
            bmorph_interval, bmorph_overlap, blend_factor)

    scbc_u_locals = scbc_u_mults * local_flow.sel(time=scbc_u_mults.index)
    return scbc_u_flows, scbc_u_mults, scbc_u_locals


def run_parallel_scbc(ds, client, region, mizuroute_exe, bmorph_config):
    def unpack_and_write_netcdf(results, segs, file_path, out_varname='scbc_flow'):
        flows = [r[0] for r in results]
        mults = [r[1] for r in results]
        local = [r[2] for r in results]
        local_ds = xr.DataArray(np.vstack(local), dims=('seg','time'))
        local_ds['seg'] = segs
        local_ds['time'] = flows[0].index
        local_ds.name = out_varname
        local_ds = local_ds.where(local_ds >= 1e-4, other=1e-4)
        local_ds.transpose().to_netcdf(file_path)

    if 'condition_var' in bmorph_config.keys() and bmorph_config['condition_var']:
        scbc_type = 'conditional'
        scbc_fun = partial(_scbc_c_seg, **bmorph_config)
    else:
        scbc_type = 'univariate'
        scbc_fun = partial(_scbc_u_seg, **bmorph_config)

    futures = [client.submit(scbc_fun, ds.sel(seg=seg)) for seg in ds['seg'].values]
    results = client.gather(futures)
    unpack_and_write_netcdf(results, ds['seg'], f'../input/{region.lower()}_local_{scbc_type}_scbc.nc')
    mizuroute_config = mizutil.write_mizuroute_config(region, scbc_type, bmorph_config['bmorph_window'])
    mizutil.run_mizuroute(mizuroute_exe, mizuroute_config)
    region_totals = xr.open_mfdataset(f'../reroute/{region.lower()}_{scbc_type}_scbc*').load()
    region_totals['seg'] = region_totals['reachID'].isel(time=0)
    return region_totals
