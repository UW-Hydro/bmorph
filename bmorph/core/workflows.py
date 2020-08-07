import bmorph
import pandas as pd

def apply_annual_bmorph(raw_ts, train_ts, obs_ts,
        training_window, bmorph_window, reference_window,
        window_size, n_smooth_long=None, n_smooth_short=5, train_on_year=False):
    training_window = slice(*training_window)
    bmorph_window = slice(*bmorph_window)
    reference_window = slice(*reference_window)
    raw_ts_window = slice(pd.to_datetime(raw_ts.index.values[0]),
                          pd.to_datetime(raw_ts.index.values[-1]))

    # bmorph the series
    overlap_period = int(window_size / 2)
    bmorph_ts = pd.Series([])
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
        bmorph_ts = bmorph_ts.append(bmorph.bmorph(raw_ts, raw_cdf_window,
                                                   raw_bmorph_window,
                                                   obs_ts, train_ts, training_window,
                                                   n_smooth_short))
    # Apply the correction
    if n_smooth_long:
        nrni_mean = obs_ts[reference_window].mean()
        train_mean = train_ts[reference_window].mean()
        bmorph_corr_ts = bmorph.bmorph_correct(raw_ts, bmorph_ts, raw_ts_window,
                                               nrni_mean, train_mean,
                                               n_smooth_long)
    else:
        bmorph_corr_ts = bmorph_ts
    return bmorph_corr_ts

def apply_annual_blendmorph(raw_upstream_ts, raw_downstream_ts, 
                            train_upstream_ts, train_downstream_ts,
                            truth_upstream_ts, truth_dowsntream_ts,
                            training_window, bmorph_window, reference_window, window_size,
                            blend_factor, n_smooth_long=None, n_smooth_short=5, train_on_year=False):
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

    bc_down_multipliers = pd.Series([])
    bc_down_totals = pd.Series([])

    bc_up_multipliers = pd.Series([])
    bc_up_totals = pd.Series([])
    
    training_window = slice(*training_window)
    bmorph_window = slice(*bmorph_window)
    reference_window = slice(*reference_window)
    raw_ts_window = slice(pd.to_datetime(raw_upstream_ts.index.values[0]),
                          pd.to_datetime(raw_upstream_ts.index.values[-1]))
    
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
        bc_up_total, bc_up_mult = bmorph.bmorph(raw_upstream_ts, raw_cdf_window,
                                                raw_bmorph_window,
                                                truth_upstream_ts, train_upstream_ts,
                                                training_window, n_smooth_short
                                               )
        bc_up_multipliers = bc_up_multipliers.append(bc_up_mult.to_xarray())
        bc_up_totals = bc_up_totals.append(bc_up_total.to_xarray())
        
        bc_down_total, bc_down_mult = bmorph.bmorph(raw_downstream_ts, raw_cdf_window,
                                                    raw_bmorph_window,
                                                    truth_dowsntream_ts, train_downstream_ts,
                                                   training_window, n_smooth_short)
        bc_down_multipliers = bc_down_multipliers.append(bc_down_mult.to_xarray())
        bc_down_totals = bc_down_totals.append(bc_down_total.to_xarray())
        
        bc_multiplier = (blend_factor * bc_up_mult) + ((1 - blend_factor) * bc_down_mult)
        bc_total = (blend_factor * bc_up_total) + ((1 - blend_factor) * bc_down_total)
        
        bc_multipliers = bc_multipliers.append(bc_multiplier.to_xarray())
        bc_totals = bc_totals.append(bc_total.to_xarray())
        
    """
    # Apply the correction
    if n_smooth_long:
        nrni_mean = obs_ts[reference_window].mean()
        train_mean = train_ts[reference_window].mean()
        bmorph_corr_ts = bmorph.bmorph_correct(raw_ts, bmorph_ts, raw_ts_window,
                                               nrni_mean, train_mean,
                                               n_smooth_long)
    else:
        bmorph_corr_ts = bmorph_ts
    """
    
    return bc_totals, bc_multipliers
