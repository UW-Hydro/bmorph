import bmorph
import pandas as pd

def apply_annual_bmorph(raw_ts, train_ts, obs_ts,
        training_window, bmorph_window, reference_window,
        window_size, n_smooth_long, n_smooth_short, train_on_year=False):
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
    nrni_mean = obs_ts[reference_window].mean()
    train_mean = train_ts[reference_window].mean()
    bmorph_corr_ts = bmorph.bmorph_correct(raw_ts, bmorph_ts, raw_ts_window,
                                           nrni_mean, train_mean,
                                           n_smooth_long)
    return bmorph_corr_ts
