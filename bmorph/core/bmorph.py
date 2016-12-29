"""
bmorph: modify a time series by removing elements of persistent differences
        (aka bias correction)

Persistent differences are inferred by comparing a 'truth' sample with a
'training' sample. These differences are then used to correcta 'raw' sample
that is presumed to have the same persistent differences as the 'training'
sample. The resulting 'bmorph' sample should then be consistent with the
'truth' sample.

Here all samples are expected to be pandas dataframes and/or Series
"""

import scipy.stats


def presrat(raw_x, raw_cdf, train_cdf, truth_cdf):
    '''Calculate the multipliers using the preservation of ratio technique'''
    # Given raw_x and raw_cdf determine the quantiles u_t
    u_t = [scipy.stats.percentileofscore(raw_cdf, x, kind='mean')
           for x in raw_x]

    # Given u_t and train_cdf determine train_x
    train_x = scipy.stats.scoreatpercentile(train_cdf, u_t)

    # Given u_t and truth_cdf determine truth_x
    truth_x = scipy.stats.scoreatpercentile(truth_cdf, u_t)

    # Calculate multiplier
    multiplier = truth_x / train_x

    return multiplier


def bmorph(truth_ts, train_ts, training_window,
           raw_ts, raw_cdf_window, raw_bmorph_window,
           nsmooth):
    '''Morph raw_ts based on differences between truth_ts and train_ts

       truth_ts: Target time series
       train_ts: Training time series that is from the same source as raw_ts. A
                 mapping is created based on train_ts and truth_ts
       training_window: Slice used to subset truth_ts and train_ts
       raw_ts:   Raw time series that will be bmorphed
       raw_cdf_window: Slice used for CDF for raw_ts
       raw_bmorph_window: Slice of raw_ts that will be bmorphed
       nsmooth:  Number of elements that will be smoothed when determining CDF

       Returns a pandas time series of length raw_bmorph_window with bmorphed
       values
    '''

    # Create the CDFs that are used for morphing the raw_ts. The mapping is
    # based on the training_window
    truth_cdf = truth_ts[training_window].\
        rolling(window=nsmooth, min_periods=1, center=True).\
        mean().sort_values()
    train_cdf = train_ts[training_window].\
        rolling(window=nsmooth, min_periods=1, center=True).\
        mean().sort_values()

    # Smooth the raw Series
    raw_smoothed_ts = raw_ts.\
        rolling(window=nsmooth, min_periods=1, center=True).mean()

    raw_smoothed_cdf = raw_smoothed_ts[raw_cdf_window].sort_values()

    # Calculate the bmorph multipliers based on the smoothed time series and
    # PDFs
    bmorph_multipliers = presrat(raw_smoothed_ts[raw_bmorph_window],
                                 raw_smoothed_cdf, train_cdf, truth_cdf)

    # Apply the bmorph multipliers to the raw time series
    bmorph_ts = bmorph_multipliers * raw_ts[raw_bmorph_window]

    return bmorph_ts


def bmorph_correct(raw_ts, bmorph_ts, correction_window,
                   truth_mean, train_mean, nsmooth):
    '''Apply a correction to bmorph_ts to get closer to preserving the ratio
       of change between two periods in raw_ts

       raw_ts:    Raw time series
       bmorph_ts: Bmorphed version of raw_ts
       correction_window: Interval over which the correction is applied
       truth_mean: Mean of target time series for the base period
       train_mean: Mean of training time series for the base period

       Returns a pandas time series of length correction_window.

       The nsmooth value in this case is typically much larger than the one
       used for the bmorph function itself. For example, 365 days.
    '''

    # Smooth the raw and bmorphed time series
    raw_ts_smoothed = raw_ts.rolling(window=nsmooth, min_periods=1,
                                     center=True).mean()
    bmorph_ts_smoothed = bmorph_ts.rolling(window=nsmooth, min_periods=1,
                                           center=True).mean()

    # Calculate the correction factors
    correction_ts = raw_ts_smoothed[correction_window].\
        div(bmorph_ts_smoothed[correction_window], axis=0) * \
        truth_mean/train_mean

    # Apply the correction to the bmorph time series
    bmorph_corrected_ts = correction_ts.mul(bmorph_ts[correction_window],
                                            axis=0)

    return bmorph_corrected_ts
