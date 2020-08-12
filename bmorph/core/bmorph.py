"""
bmorph: modify a time series by removing elements of persistent differences
        (aka bias correction)

Persistent differences are inferred by comparing a 'truth' sample with a
'training' sample. These differences are then used to correct a 'raw' sample
that is presumed to have the same persistent differences as the 'training'
sample. The resulting 'bmorph' sample should then be consistent with the
'truth' sample.
"""

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.neighbors import KernelDensity

# Done fail silently on divide by zero, but raise an error instead
pd.np.seterr(divide='raise')



def kde2D(x, y, xbins=200, ybins=10, **kwargs):
    """ Estimate a 2 dimensional pdf via kernel density estimation """
    xx, yy = np.mgrid[x.min():x.max():(xbins * 1j), y.min():y.max():(ybins * 1j)]
    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T
    kde = KernelDensity(**kwargs)
    kde.fit(xy_train)
    # should this be exponential'd?
    z = np.exp(kde.score_samples(xy_sample))
    return xx[:, 0], yy[0, :], np.reshape(z, yy.shape)


def marginalize_cdf(y_raw, z_raw, vals):
    """Find the marginalized cdf by computing cumsum(P(x|y=val)) for each val"""
    locs =  np.argmin(np.abs(vals[:, np.newaxis] - y_raw), axis=1)
    z = np.cumsum(z_raw[:, locs], axis=0)
    z /= z[-1, :]
    return z


def mdcdedcdfm(raw_x: pd.Series, train_x: pd.Series, truth_x: pd.Series,
               raw_y: pd.Series, train_y: pd.Series, truth_y: pd.Series=None,
               bw=3, xbins=200, ybins=10, rtol=1e-6, atol=1e-8) -> pd.Series:
    """
    multiDimensional ConDitional EquiDistant CDF matching function
    \tilde{x_{mp}} = x_{mp} + F^{-1}_{oc}(F_{mp}(x_{mp}|y_{mp})|y_{oc})
                            - F^{-1}_{mc}(F_{mp}(x_{mp}|y_{mp})|y_{mc})
    """
    x_raw, y_raw, z_raw = kde2D(raw_x, raw_y, xbins, ybins,
                                bandwidth=bw, rtol=rtol, atol=atol)
    x_train, y_train, z_train = kde2D(train_x, train_y, xbins, ybins,
                                      bandwidth=bw, rtol=rtol, atol=atol)
    x_truth, y_truth, z_truth = kde2D(truth_x, truth_y, xbins, ybins,
                                      bandwidth=bw, rtol=rtol, atol=atol)

    nx = np.arange(len(raw_x))
    raw_cdfs = marginalize_cdf(y_raw, z_raw, raw_y)
    u_t = raw_cdfs[np.argmin(np.abs(raw_x[:, np.newaxis] - x_raw), axis=1), nx]
    u_t = u_t[:, np.newaxis]

    train_cdf = marginalize_cdf(y_train, z_train, train_y).T
    mapped_train = x_train[np.argmin(np.abs(u_t - train_cdf), axis=1)]

    truth_cdfs = marginalize_cdf(y_truth, z_truth, truth_y).T
    mapped_truth = x_truth[np.argmin(np.abs(u_t - truth_cdfs), axis=1)]

    return pd.Series(mapped_truth / mapped_train, index=raw_x.index)



def edcdfm(raw_x, raw_cdf, train_cdf, truth_cdf):
    '''Calculate  multipliers using an adapted version of the EDCDFm technique

    This routine implements part of the PresRat bias correction method from
    Pierce et al. (2015; http://dx.doi.org/10.1175/JHM-D-14-0236.1), which is
    itself an extension of the Equidistant quantile matching (EDCDFm) technique
    of Li et al. (2010; http://dx.doi.org/10.1029/94JD00483). The part that is
    implemented here is the amended form of EDCDFm that determines
    multiplicative changes in the quantiles of a CDF.

    In particular, if the value `raw_x` falls at quantile `u_t` (in `raw_cdf`),
    then the bias-corrected value is the value in `truth_cdf` at `u_t`
    (`truth_x`) multiplied by the model-predicted change at `u_t` evaluated as
    a ratio (i.e., model future (or `raw_x`) / model historical (or
    `truth_x`)). Thus, the bias-corrected value is `raw_x` multiplied by
    `truth_x`/`train_x`. Here we only return the multiplier
    `truth_x`/`train_x`. This method preserves the model-predicted median
    (not mean) change evaluated multiplicatively. Additional corrections
    are required to preserve the mean change. Inclusion of these additional
    corrections constitutes the PresRat method.

    Parameters
    ----------
    raw_x : pandas.Series
        Series of raw values that will be used to determine the quantile `u_t`
    raw_cdf : pandas.Series
        Series of raw values that represents the CDF that is used to
        determine the non-parametric quantile of `raw_x`
    train_cdf: pandas.Series
        Series of training values that represents the CDF based on
        the same process as `raw_cdf`, but overlapping in time with `truth_cdf`
    truth_cdf: pandas.Series
         Series of truth values that represents the truth CDF and that
         overlaps in time with `train_cdf`

    Returns
    -------
    multiplier : pandas.Series
        Multipliers for `raw_x`. The pandas.Series has the same index as
        `raw_x`
    '''
    # Type checking (note that the checking is more strict here then it
    # probably needs to be)
    assert isinstance(raw_x, pd.Series)
    assert isinstance(raw_cdf, pd.Series)
    assert isinstance(train_cdf, pd.Series)
    assert isinstance(truth_cdf, pd.Series)

    # Given raw_x and raw_cdf determine the quantiles u_t
    u_t = [scipy.stats.percentileofscore(raw_cdf, x, kind='mean')
           for x in raw_x]

    # Given u_t and train_cdf determine train_x
    train_x = pd.np.percentile(train_cdf, u_t)
    train_x[train_x < 1e-6] = 1e-6

    # Given u_t and truth_cdf determine truth_x
    truth_x = pd.np.percentile(truth_cdf, u_t)

    # Calculate multiplier
    multiplier = truth_x / train_x

    return pd.Series(multiplier, index=raw_x.index)

def bmorph(raw_ts, raw_cdf_window, raw_bmorph_window,
           truth_ts, train_ts, training_window,
           nsmooth, raw_y=None, truth_y=None, train_y=None,
           bw=3, xbins=200, ybins=10, rtol=1e-6, atol=1e-8):
    '''Morph `raw_ts` based on differences between `truth_ts` and `train_ts`

    bmorph is an adaptation of the PresRat bias correction procedure from
    Pierce et al. (2015; http://dx.doi.org/10.1175/JHM-D-14-0236.1), which is
    itself an extension of the Equidistant quantile matching (EDCDFm) technique
    of Li et al. (2010; http://dx.doi.org/10.1029/94JD00483). The method as
    implemented here uses a multiplicative change in the quantiles of a CDF,
    followed by a simple correction to preserve changes in the long-term mean.
    No further frequency-based corrections are applied.

    The method differs from PresRat in that it is not applied for fixed periods
    (but uses a moving window) to prevent discontinuities in the corrected time
    series and it does not apply a frequency-based correction.
    
    The method also allows changes to be made through an adapted version of the
    EDCDFm technique or through the multiDimensional ConDitional EquiDistant CDF
    matching function if a second timeseries variable is passed.

    Parameters
    ----------
    raw_ts : pandas.Series
        Raw time series that will be bmorphed
    raw_cdf_window : slice
        Slice used to determine the CDF for `raw_ts`
    raw_bmorph_window : slice
        Slice of `raw_ts` that will be bmorphed
    truth_ts : pandas.Series
        Target time series. This is the time series with truth values that
        overlaps with `train_ts` and is used to calculated `truth_cdf`
    train_ts : pandas.Series
        Training time series. This time series is generated by the same process
        as `raw_ts` but overlaps with `truth_ts`. It is used to calculate
        `train_cdf`
    training_window : slice
        Slice used to subset `truth_ts` and `train_ts` when the mapping between
        them is created
    nsmooth : int
        Number of elements that will be smoothed when determining CDFs
    raw_y : pandas.Series
        Raw time series of the second time series variable for mdcdedcdfm
    truth_y : pandas.Series
        Target second time series
    train_y : pandas.Series
        Training second time series
    bw : int
        bin width for both sets of time series
    xbins : int
        Bins for the flow time series
    ybins : int
        Bins for the second time series
        
    Returns
    -------
    bmorph_ts : pandas.Series
        Returns a time series of length `raw_bmorph_window` with bmorphed
        values
    '''
    # Type checking
    assert isinstance(raw_ts, pd.Series)
    assert isinstance(raw_cdf_window, slice)
    assert isinstance(raw_bmorph_window, slice)
    assert isinstance(truth_ts, pd.Series)
    assert isinstance(train_ts, pd.Series)
    assert isinstance(training_window, slice)

    # Smooth the raw Series
    raw_smoothed_ts = raw_ts.rolling(
        window=nsmooth, min_periods=1, center=True).mean()
    
    # Check if using edcdfm or mdcdedcdfm through second variable being added  
    # for the raw and train series because truth can be set as train later
    if (raw_y is None) or (truth_y is None) or (train_y is None):
        # Create the CDFs that are used for morphing the raw_ts. The mapping is
        # based on the training_window
        truth_cdf = truth_ts[training_window].rolling(
            window=nsmooth, min_periods=1, center=True).mean()
        train_cdf = train_ts[training_window].rolling(
            window=nsmooth, min_periods=1, center=True).mean()

        # Calculate the bmorph multipliers based on the smoothed time series and
        # PDFs
        bmorph_multipliers = edcdfm(raw_smoothed_ts[raw_bmorph_window],
                                    raw_smoothed_ts[raw_cdf_window],
                                    train_cdf, truth_cdf)

        # Apply the bmorph multipliers to the raw time series
        bmorph_ts = bmorph_multipliers * raw_ts[raw_bmorph_window]
        
    else:
        # Continue Type Checking additionally series
        assert isinstance(raw_y, pd.Series)
        assert isinstance(truth_y, pd.Series)
        assert isinstance(train_y, pd.Series)
        
        # smooth the raw y series
        raw_smoothed_y = raw_y.rolling(
            window=nsmooth, min_periods=1, center=True).mean()
        
        bmorph_multipliers = mdcdedcdfm(raw_smoothed_ts[raw_bmorph_window], 
                                        train_ts[training_window], truth_ts[training_window],
                                        raw_smoothed_y[raw_bmorph_window],
                                        train_y[training_window], truth_y[training_window],
                                        bw=bw, xbins=xbins, ybins=ybins,
                                        rtol=rtol, atol=atol)
        
        bmorph_ts = bmorph_multipliers * raw_ts[raw_bmorph_window]
        
        
    return bmorph_ts, bmorph_multipliers


def bmorph_correct(raw_ts, bmorph_ts, correction_window,
                   truth_mean, train_mean, nsmooth):
    '''Correct bmorphed values to preserve the ratio of change

    Apply a correction to bmorphed values to preserve the mean change over a
    `correction_window`. This is similar to teh correction applied in the
    original PresRat algorithm; Pierce et al. 2015;
    http://dx.doi.org/10.1175/JHM-D-14-0236.1), except that we use a rolling
    mean to determine the correction to avoid discontinuities on the
    boundaries.

    Parameters
    ----------
    raw_ts :    pandas.series
        Series of raw values that have not been bmorphed
    bmorph_ts : Bmorphed version of `raw_ts`
        Series of bmorphed values
    correction_window : slice
        Slice of `raw_ts` and `bmorph_ts` over which the correction is applied
    truth_mean : float
        Mean of target time series (`truth_ts`) for the base period
    train_mean : float
        Mean of training time series (`train_ts`) for the base period
    nsmooth : int
        Number of elements that will be smoothed in `raw_ts` and `bmorph_ts`.
        The nsmooth value in this case is typically much larger than the one
        used for the bmorph function itself. For example, 365 days.

    Returns
    -------
    bmorph_corrected_ts : pandas.Series
       Corrected series of length `correction_window`.

    '''
    # Type checking
    assert isinstance(raw_ts, pd.Series)
    assert isinstance(bmorph_ts, pd.Series)
    assert isinstance(correction_window, slice)

    # Smooth the raw and bmorphed time series
    raw_ts_smoothed = raw_ts.rolling(window=nsmooth, min_periods=1,
                                     center=True).mean()
    bmorph_ts_smoothed = bmorph_ts.rolling(window=nsmooth, min_periods=1,
                                           center=True).mean()

    # Calculate the correction factors
    correction_ts = raw_ts_smoothed[correction_window] / \
        bmorph_ts_smoothed[correction_window] * \
        truth_mean/train_mean

    # Apply the correction to the bmorph time series
    bmorph_corrected_ts = correction_ts * bmorph_ts[correction_window]

    return bmorph_corrected_ts
