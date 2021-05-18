import bmorph
import pandas as pd
import numpy as np
import xarray as xr
from functools import partial
from tqdm.autonotebook import tqdm
from bmorph.util import mizuroute_utils as mizutil


def apply_annual_bmorph(raw_ts, train_ts, obs_ts,
        train_window, bmorph_window, reference_window,
        bmorph_overlap, n_smooth_long=None, n_smooth_short=5, train_on_year=False,
        raw_y=None, train_y=None, obs_y=None, bw=3, xbins=200, ybins=10,
        rtol=1e-7, atol=0, method='hist', **kwargs):
    """Bias correction is performed by bmorph on yearly intervals.

    Parameters
    ----------
    raw_ts : pandas.Series
        Raw flow timeseries.
    train_ts : pandas.Series
        Flow timeseries to train the bias correction model with.
    obs_ts : pandas.Series
        Observed/reference flow timeseries.
    train_window : pandas.date_range
        Date range to train the bias correction model.
    bmorph_window : pandas.date_range
        Date range to apply bmorph onto flow timeseries.
    reference_window : pandas.date_range
        Date range to smooth elements in 'raw_ts' and 'bmorph_ts'.
    bmorph_overlap : int
        Total overlap CDF windows have with each other, distributed evenly
        before and after the application window.
    n_smooth_long : int, optional
        Number of elements that will be smoothed in `raw_ts` and `bmorph_ts`.
        The nsmooth value in this case is typically much larger than the one
        used for the bmorph function itself. For example, 365 days.
    n_smooth_short : int, optional
        Number of elements that will be smoothed when determining CDFs used
        for the bmorph function itself.
    train_on_year : boolean, optional
        Fits a new target CDF for each year in the training period, rather than a
        single CDF for the entire period. This should only be used for testing
        purposes.
    raw_y : pandas.Series, optional
        Raw time series of the second time series variable for conditioning.
    train_y : pandas.Series, optional
        Training second time series.
    obs_y : pandas.Series, optional
        Target second time series.
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

    Returns
    -------
    bmorph_corr_ts : pd.Series
        Returns a time series of length of an interval in the bmoprh window
        with bmorphed values.
    bmorph_mulitpliers : pd.Series
        Returns a time series of equal length to bc_totals used to scale the
        raw flow values into the bmorphed values returned in bc_totals.
    """
    train_window = slice(*train_window)
    bmorph_window = slice(*bmorph_window)
    reference_window = slice(*reference_window)
    raw_ts_window = slice(pd.to_datetime(raw_ts.index.values[0]),
                          pd.to_datetime(raw_ts.index.values[-1]))

    # bmorph the series
    overlap_period = int(bmorph_overlap / 2)
    bmorph_ts = pd.Series([])
    bmorph_multipliers = pd.Series([])
    for year in range(bmorph_window.start.year, bmorph_window.stop.year+1):
        if train_on_year:
            train_window = slice(pd.to_datetime('{}-10-01 00:00:00'.format(year-1)),
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

        bc_total, bc_mult = bmorph.bmorph(raw_ts, raw_cdf_window, raw_bmorph_window, obs_ts, train_ts,
                                          train_window, n_smooth_short, raw_y, obs_y, train_y,
                                          bw=bw, xbins=xbins, ybins=ybins, rtol=rtol, atol=atol,
                                          method=method)
        bmorph_ts = bmorph_ts.append(bc_total)
        bmorph_multipliers = bmorph_multipliers.append(bc_mult)


    # Apply the correction
    if n_smooth_long:
        nrni_mean = obs_ts[reference_window].mean()
        train_mean = train_ts[reference_window].mean()
        bmorph_corr_ts, corr_ts = bmorph.bmorph_correct(raw_ts, bmorph_ts, raw_ts_window,
                                                        nrni_mean, train_mean,
                                                        n_smooth_long)
    else:
        bmorph_corr_ts = bmorph_ts
    return bmorph_corr_ts[bmorph_window], bmorph_multipliers[bmorph_window]

def apply_interval_bmorph(raw_ts, train_ts, obs_ts,
        train_window, bmorph_window, reference_window, bmorph_interval,
        bmorph_overlap, n_smooth_long=None, n_smooth_short=5,
        raw_y=None, train_y=None, obs_y=None, bw=3, xbins=200, ybins=10,
        rtol=1e-6, atol=1e-8, method='hist', **kwargs):
    """Bias correction is performed by bmorph on user-defined intervals.

    Parameters
    ----------
    raw_ts : pandas.Series
        Raw flow timeseries.
    train_ts : pandas.Series
        Flow timeseries to train the bias correction model with.
    obs_ts : pandas.Series
        Observed/reference flow timeseries.
    train_window : pandas.date_range
        Date range to train the bias correction model.
    bmorph_window : pandas.date_range
        Date range to apply bmorph onto flow timeseries.
    reference_window : pandas.date_range
        Date range to smooth elements in 'raw_ts' and 'bmorph_ts'.
    bmorph_interval : pandas.DateOffset
        Difference between bmorph application intervals.
    bmorph_overlap : int
        Total overlap CDF windows have with each other, distributed evenly
        before and after the application window.
    n_smooth_long : int, optional
        Number of elements that will be smoothed in `raw_ts` and `bmorph_ts`.
        The nsmooth value in this case is typically much larger than the one
        used for the bmorph function itself. For example, 365 days.
    n_smooth_short : int, optional
        Number of elements that will be smoothed when determining CDFs used
        for the bmorph function itself.
    raw_y : pandas.Series, optional
        Raw time series of the second time series variable for conditioning.
    train_y : pandas.Series, optional
        Training second time series.
    obs_y : pandas.Series, optional
        Target second time series.
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

    Returns
    -------
    bmorph_corr_ts : pandas.Series
        Returns a time series of length of an interval in the bmoprh window
        with bmorphed values.
    bmorph_multipliers : pandas.Series
        Returns a time series of equal length to bc_totals used to scale the
        raw flow values into the bmorphed values returned in bc_totals.
    """
    assert isinstance(bmorph_interval, pd.DateOffset)

    if bmorph_interval == pd.DateOffset(days=1):
        raise Exception("Please enter a bmorph_interval greater than 1 day(s)")

    train_window = slice(*train_window)
    bmorph_window = slice(*bmorph_window)
    reference_window = slice(*reference_window)
    raw_ts_window = slice(pd.to_datetime(raw_ts.index.values[0]),
                          pd.to_datetime(raw_ts.index.values[-1]))

    # bmorph the series
    overlap_period = int(bmorph_overlap / 2)
    bmorph_ts = pd.Series([])
    bmorph_multipliers = pd.Series([])
    bmorph_range = pd.date_range(bmorph_window.start, bmorph_window.stop+bmorph_interval,
                                 freq=bmorph_interval)
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
                                          train_window, n_smooth_short, raw_y, obs_y, train_y,
                                          bw=bw, xbins=xbins, ybins=ybins, rtol=rtol, atol=atol,
                                          method=method)
        bmorph_ts = bmorph_ts.append(bc_total)
        bmorph_multipliers = bmorph_multipliers.append(bc_mult)


    # Apply the correction
    if n_smooth_long:
        nrni_mean = obs_ts[reference_window].mean()
        train_mean = train_ts[reference_window].mean()
        bmorph_corr_ts, corr_ts = bmorph.bmorph_correct(raw_ts, bmorph_ts, raw_ts_window,
                                                        nrni_mean, train_mean,
                                                        n_smooth_long)
    else:
        bmorph_corr_ts = bmorph_ts
    return bmorph_corr_ts[bmorph_window], bmorph_multipliers[bmorph_window]


def apply_annual_blendmorph(raw_upstream_ts, raw_downstream_ts,
                            train_upstream_ts, train_downstream_ts,
                            truth_upstream_ts, truth_downstream_ts,
                            train_window, bmorph_window, reference_window, bmorph_overlap,
                            blend_factor, n_smooth_long=None, n_smooth_short=5, train_on_year=False,
                            raw_upstream_y = None, raw_downstream_y = None,
                            train_upstream_y = None, train_downstream_y = None,
                            truth_upstream_y = None, truth_downstream_y = None,
                            bw=3, xbins=200, ybins=10, atol=0, rtol=1e-7, method='hist', **kwargs):
    """Bias correction is performed by blending bmorphed flows on yearly intervals.

    Blendmorph is used to perform spatially consistent bias correction, this function
    does so on an annual interval. This is done by performing bmorph bias correction
    for each site's timeseries according to upstream and downstream gauge sites
    (or proxies) where true flows are known. The upstream and downstream corrected
    timeseries are then multiplied by fractional weights, `blend_factor`, that sum
    to 1 between them so the corrected flows can be combined, or "blended," into one,
    representative corrected flow series for the site. It is thereby important to specify
    upstream and downstream values so bias corrections are performed with values that
    most closely represent each site being corrected.

    Parameters
    ----------
    raw_upstream_ts : pandas.Series
        Raw flow timeseries corresponding to the upstream  flows.
    raw_downstream_ts : pandas.Series
        Raw flow timerseries corresponding to the downstream  flows.
    train_upstream_ts : pandas.Series
        Flow timeseries to train the bias correction model with for the upstream flows.
    train_downstream_ts : pandas.Series
        Flow timeseries to train the bias correction model with for the downstream flows.
    truth_upstream_ts : pandas.Series
        Observed/reference flow timeseries corresponding to the upstream flows.
    truth_downstream_ts : pandas.Series
        Observed/reference flow timeseries corresponding to the downstream flows.
    train_window : pandas.date_range
        Date range to train the bias correction model.
    bmorph_window : pandas.date_range
        Date range to apply bmorph onto flow timeseries.
    reference_window : pandas.date_range
        Date range to smooth elements in 'raw_ts' and 'bmorph_ts'.
    bmorph_overlap : int
        Total overlap CDF windows have with each other, distributed evenly
        before and after the application window.
    blend_factor : numpy.array
        An array determining how upstream and downstream bmorphing is proportioned.
        This is determined by the fill_method used in mizuroute_utils. The blend_factor
        entries are the proportion of upstream multiplers and totals added with
        1-blend_factor of downstream multipliers and totals.
    n_smooth_long : int, optional
        This functionality is still to be implemented.

        Number of elements that will be smoothed in `raw_ts` and `bmorph_ts`.
        The nsmooth value in this case is typically much larger than the one
        used for the bmorph function itself. For example, 365 days.
    n_smooth_short : int, optional
        Number of elements that will be smoothed when determining CDFs used
        for the bmorph function itself.
    train_on_year : boolean, optional
        Fits a new target CDF for each year in the training period, rather than a
        single CDF for the entire period. This should only be used for testing
        purposes.
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
    truth_upstream_y : pandas.Series, optional
        Target second time series variable for conditioning corresponding to upstream flows.
    truth_downstream_y : pandas.Series, optional
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

    Returns
    -------
    bc_totals: pandas.Series
        Returns a time series of length of an interval in the bmoprh window
        with bmorphed values.
    bc_multipliers: pandas.Series
        Returns a time series of equal length to bc_totals used to scale the
        raw flow values into the bmorphed values returned in bc_totals.
    """

    bc_multipliers = pd.Series([])
    bc_totals = pd.Series([])

    train_window = slice(*train_window)
    bmorph_window = slice(*bmorph_window)
    reference_window = slice(*reference_window)
    raw_ts_window = slice(pd.to_datetime(raw_upstream_ts.index.values[0]),
                          pd.to_datetime(raw_upstream_ts.index.values[-1]))
    # Check if there is enough data input to run conditioning for both upstream
    # and downstream bmorphs. Boolean used here instead of later to make certain
    # both upstream and downstream use the same method and to minimze checks within
    # the for-loop
    run_mdcd = False
    y_varlist = [raw_upstream_y, train_upstream_y, truth_upstream_y,
                 raw_downstream_y, train_downstream_y, truth_downstream_y]
    if np.any(list(map(lambda x: x is not None, y_varlist))):
        run_mdcd = True

    # bmorph the series
    overlap_period = int(bmorph_overlap / 2)

    for year in range(bmorph_window.start.year, bmorph_window.stop.year+1):
        # set up annual windows
        if train_on_year:
            train_window = slice(pd.to_datetime('{}-10-01 00:00:00'.format(year-1)),
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
                                                    train_window, n_smooth_short,
                                                    raw_upstream_y, truth_upstream_y,
                                                    train_upstream_y, bw=bw, xbins=xbins,
                                                    ybins=ybins, rtol=rtol, atol=atol,
                                                    method=method)

            bc_down_total, bc_down_mult = bmorph.bmorph(raw_downstream_ts, raw_cdf_window,
                                                        raw_bmorph_window,
                                                        truth_downstream_ts, train_downstream_ts,
                                                        train_window, n_smooth_short,
                                                        raw_downstream_y, truth_downstream_y,
                                                        train_downstream_y, bw=bw, xbins=xbins,
                                                        ybins=ybins, rtol=rtol, atol=atol,
                                                        method=method)
        else:
            bc_up_total, bc_up_mult = bmorph.bmorph(raw_upstream_ts, raw_cdf_window,
                                                    raw_bmorph_window,
                                                    truth_upstream_ts, train_upstream_ts,
                                                    train_window, n_smooth_short)

            bc_down_total, bc_down_mult = bmorph.bmorph(raw_downstream_ts, raw_cdf_window,
                                                        raw_bmorph_window,
                                                        truth_downstream_ts, train_downstream_ts,
                                                       train_window, n_smooth_short)

        bc_multiplier = (blend_factor * bc_up_mult) + ((1 - blend_factor) * bc_down_mult)
        bc_total = (blend_factor * bc_up_total) + ((1 - blend_factor) * bc_down_total)

        bc_multipliers = bc_multipliers.append(bc_multiplier)
        bc_totals = bc_totals.append(bc_total)

    # Apply the correction to preserve the mean change
    if n_smooth_long:
        raw_ts = (blend_factor * raw_upstream_ts) + ((1 - blend_factor) * raw_downstream_ts)
        obs_ts = (blend_factor * truth_upstream_ts) + ((1 - blend_factor) * truth_downstream_ts)
        train_ts = (blend_factor * train_upstream_ts) + ((1 - blend_factor) * train_downstream_ts)
        nrni_mean = obs_ts[reference_window].mean()
        train_mean = train_ts[reference_window].mean()
        bc_totals, corr_ts = bmorph.bmorph_correct(raw_ts, bc_totals, raw_ts_window,
                                                   nrni_mean, train_mean,
                                                   n_smooth_long)
        bc_multipliers *= corr_ts

    return bc_totals[bmorph_window], bc_multipliers[bmorph_window]

def apply_interval_blendmorph(raw_upstream_ts, raw_downstream_ts,
                            train_upstream_ts, train_downstream_ts,
                            truth_upstream_ts, truth_downstream_ts,
                            train_window, bmorph_window, reference_window, bmorph_interval, bmorph_overlap,
                            blend_factor, n_smooth_long=None, n_smooth_short=5,
                            raw_upstream_y = None, raw_downstream_y = None,
                            train_upstream_y = None, train_downstream_y = None,
                            truth_upstream_y = None, truth_downstream_y = None,
                            bw=3, xbins=200, ybins=10, rtol=1e-6, atol=1e-8, method='hist', **kwargs):
    """Bias correction is performed by blending bmorphed flows on user defined intervals.

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
    truth_upstream_ts : pandas.Series
        Observed/reference flow timeseries corresponding to the upstream flows.
    truth_downstream_ts : pandas.Series
        Observed/reference flow timeseries corresponding to the downstream flows.
    train_window : pandas.date_range
        Date range to train the bias correction model.
    bmorph_window : pandas.date_range
        Date range to apply bmorph onto flow timeseries.
    reference_window : pandas.date_range
        Date range to smooth elements in 'raw_ts' and 'bmorph_ts'.
    bmorph_interval : pandas.DateOffset
        Difference between bmorph application intervals.
    bmorph_overlap: int
        Total overlap CDF windows have with each other, distributed evenly
        before and after the application window.
    blend_factor : numpy.array
        An array determining how upstream and downstream bmorphing is proportioned.
        This is determined by the fill_method used in mizuroute_utils. The blend_factor
        entries are the proportion of upstream multiplers and totals added with
        1-blend_factor of downstream multipliers and totals.
    n_smooth_long : int, optional
        This functionality is still to be implemented.

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
    truth_upstream_y : pandas.Series, optional
        Target second time series variable for conditioning corresponding to upstream flows.
    truth_downstream_y : pandas.Series, optional
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

    Returns
    -------
    bc_totals : pandas.Series
        Returns a time series of length of an interval in the bmoprh window
        with bmorphed values.
    bc_multipliers : pandas.Series
        Returns a time series of equal length to bc_totals used to scale the
        raw flow values into the bmorphed values returned in bc_totals.
    """
    assert isinstance(bmorph_interval, pd.DateOffset)

    if bmorph_interval == pd.DateOffset(days=1):
        raise Exception("Please enter a bmorph_interval greater than 1 day(s)")

    bc_multipliers = pd.Series([])
    bc_totals = pd.Series([])

    train_window = slice(*train_window)
    bmorph_window = slice(*bmorph_window)
    reference_window = slice(*reference_window)
    raw_ts_window = slice(pd.to_datetime(raw_upstream_ts.index.values[0]),
                          pd.to_datetime(raw_upstream_ts.index.values[-1]))

    # Check if there is enough data input to run conditioning for both upstream
    # and downstream bmorphs. Boolean used here instead of later to make certain
    # both upstream and downstream use the same method and to minimze checks within
    # the for-loop
    run_mdcd = False
    y_varlist = [raw_upstream_y, train_upstream_y, truth_upstream_y,
                 raw_downstream_y, train_downstream_y, truth_downstream_y]
    if np.any(list(map(lambda x: x is not None, y_varlist))):
        run_mdcd = True

    # bmorph the series
    overlap_period = int(bmorph_overlap / 2)
    bmorph_ts = pd.Series([])
    bmorph_multipliers = pd.Series([])
    bmorph_range = pd.date_range(bmorph_window.start, bmorph_window.stop+bmorph_interval,
                                 freq=bmorph_interval)
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
                                                    train_window, n_smooth_short,
                                                    raw_upstream_y, truth_upstream_y,
                                                    train_upstream_y, bw=bw, xbins=xbins,
                                                    ybins=ybins, rtol=rtol, atol=atol,
                                                    method=method)

            bc_down_total, bc_down_mult = bmorph.bmorph(raw_downstream_ts, raw_cdf_window,
                                                        raw_bmorph_window,
                                                        truth_downstream_ts, train_downstream_ts,
                                                        train_window, n_smooth_short,
                                                        raw_downstream_y, truth_downstream_y,
                                                        train_downstream_y, bw=bw, xbins=xbins,
                                                        ybins=ybins, rtol=rtol, atol=atol,
                                                        method=method)
        else:
            bc_up_total, bc_up_mult = bmorph.bmorph(raw_upstream_ts, raw_cdf_window,
                                                    raw_bmorph_window,
                                                    truth_upstream_ts, train_upstream_ts,
                                                    train_window, n_smooth_short)

            bc_down_total, bc_down_mult = bmorph.bmorph(raw_downstream_ts, raw_cdf_window,
                                                        raw_bmorph_window,
                                                        truth_downstream_ts, train_downstream_ts,
                                                       train_window, n_smooth_short)


        bc_multiplier = (blend_factor * bc_up_mult) + ((1 - blend_factor) * bc_down_mult)
        bc_total = (blend_factor * bc_up_total) + ((1 - blend_factor) * bc_down_total)

        bc_multipliers = bc_multipliers.append(bc_multiplier)
        bc_totals = bc_totals.append(bc_total)

    # Apply the correction to preserve the mean change
    if n_smooth_long:
        raw_ts = (blend_factor * raw_upstream_ts) + ((1 - blend_factor) * raw_downstream_ts)
        obs_ts = (blend_factor * truth_upstream_ts) + ((1 - blend_factor) * truth_downstream_ts)
        train_ts = (blend_factor * train_upstream_ts) + ((1 - blend_factor) * train_downstream_ts)
        nrni_mean = obs_ts[reference_window].mean()
        train_mean = train_ts[train_window].mean()
        bc_totals, corr_ts = bmorph.bmorph_correct(raw_ts, bc_totals, raw_ts_window,
                                                   nrni_mean, train_mean,
                                                   n_smooth_long)
        bc_multipliers *= corr_ts

    return bc_totals[bmorph_window], bc_multipliers[bmorph_window]


def _scbc_c_seg(ds, train_window, bmorph_window, reference_window,
               bmorph_interval, bmorph_overlap, condition_var, **kwargs):
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
               bmorph_interval, bmorph_overlap, condition_var=None, **kwargs):
    up_raw_ts =    ds['IRFroutedRunoff'].to_series()
    up_train_ts =  ds['up_raw_flow'].to_series()
    up_obs_ts =    ds['up_ref_flow'].to_series()
    up_seg =  int( ds['up_ref_seg'].values[()])
    dn_raw_ts =    ds['IRFroutedRunoff'].to_series()
    dn_train_ts =  ds['down_raw_flow'].to_series()
    dn_obs_ts =    ds['down_ref_flow'].to_series()
    dn_seg =  int( ds['down_ref_seg'].values[()])
    blend_factor = ds['cdf_blend_factor'].values[()]
    local_flow =   ds['dlayRunoff'].copy(deep=True)

    scbc_u_flows, scbc_u_mults = apply_interval_blendmorph(
            up_raw_ts, dn_raw_ts,
            up_train_ts, dn_train_ts,
            up_obs_ts, dn_obs_ts,
            train_window, bmorph_window, reference_window,
            bmorph_interval, bmorph_overlap, blend_factor)

    scbc_u_locals = scbc_u_mults * local_flow.sel(time=scbc_u_mults.index)
    return scbc_u_flows, scbc_u_mults, scbc_u_locals


def run_parallel_scbc(ds, mizuroute_exe, bmorph_config, client=None):
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

    if client:
        futures = [client.submit(scbc_fun, ds.sel(seg=seg)) for seg in ds['seg'].values]
        results = client.gather(futures)
    else:
        results = []
        for seg in tqdm(ds['seg'].values):
            results.append(scbc_fun(ds.sel(seg=seg)))
    unpack_and_write_netcdf(results, ds['seg'], f'{bmorph_config["data_path"]}/input/{bmorph_config["output_prefix"]}_local_{scbc_type}_scbc.nc')
    config_path, mizuroute_config = mizutil.write_mizuroute_config(bmorph_config["output_prefix"],
            scbc_type, bmorph_config['bmorph_window'],
            config_dir=bmorph_config['data_path']+'/mizuroute_configs/',
            topo_dir=bmorph_config['data_path']+'/topologies/',
            input_dir=bmorph_config['data_path']+'/input/',
            output_dir=bmorph_config['data_path']+'/output/',
            )
    mizutil.run_mizuroute(mizuroute_exe, config_path)
    region_totals = xr.open_mfdataset(f'{mizuroute_config["output_dir"]}{bmorph_config["output_prefix"]}_{scbc_type}_scbc*')
    region_totals = region_totals.sel(time=slice(*bmorph_config['bmorph_window']))
    region_totals['seg'] = region_totals['reachID'].isel(time=0)
    return region_totals.load()

def bmorph_to_dataarray(dict_flows, name):
    da = xr.DataArray(np.vstack(dict_flows.values()), dims=('site', 'time'))
    da['site'] = list(dict_flows.keys())
    da['time'] = list(dict_flows.values())[0].index
    da.name = name
    return da.transpose()
