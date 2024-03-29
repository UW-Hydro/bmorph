import numpy as np
import xarray as xr
import pandas as pd
from typing import List
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import probscale

import networkx as nx
import graphviz as gv
import pygraphviz as pgv
import probscale
from networkx.drawing.nx_agraph import graphviz_layout

from bmorph.evaluation.constants import colors99p99
from bmorph.evaluation import descriptive_statistics as dst

from statsmodels.distributions.empirical_distribution import ECDF

#*****************************************************************************************
# Plotting Helper Functions:
#      custom_legend
#      calc_water_year
#      find_index_water_year
#      determine_row_col
#      log10_1p
#      scatter_series_axes
#*****************************************************************************************

def custom_legend(names:List, colors=colors99p99):
    """Creates a list of patches to be passed in as `handles` for the plt.legends function.

    Parameters
    ----------
    names : list
        Legend names.
    colors : list
        A list of the colors corresponding to `names`.

    Returns
    -------
    handles
        Handle parameter for matplotlib.legend.
    """
    legend_elements = list()
    for i,name in enumerate(names):
        legend_elements.append(mpl.patches.Patch(facecolor = colors[i], label = name))
    return legend_elements

def calc_water_year(df: pd.DataFrame):
    """Calculates the water year.

    Parameters
    ----------
    df : pandas.DataFrame
        Flow timeseries with a DataTimeIndex.

    Returns
    -------
    pandas.DataFrame.index
        A pandas.DataFrame index grouped by water year.

    """
    return df.index.year + (df.index.month >= 10).astype(int)

def find_index_water_year(data: pd.DataFrame) -> np.int:
    """ Finds the index of the first hydrologic year.

    Parameters
    ----------
    data : pd.DataFrame
        Flow timeseries with a DateTime index.

    Returns
    -------
    int
        Index of the first hydrologic year.
    """
    water_year = pd.Timestamp(0)
    i = 0
    while water_year == pd.Timestamp(0):
        date = data.index[i]
        if date.month == 10 and date.day == 1:
            water_year = date
        else:
            i = i + 1
    return i

def determine_row_col(n:int, pref_rows = True):
    """Determines rows and columns for rectangular subplots

    Calculates a rectangular subplot layout that contains at least n subplots,
    some may need to be turned off in plotting. If a square configuration is
    possible, then a square configuration will be proposed. This helps automate
    the process of plotting a variable number of subplots.

    Parameters
    ----------
    n : int
        Total number of plots.
    pref_rows : boolean
        If True, and only a rectangular arrangment is possible, then put the
        longer dimension in n_rows. If False, then it is placed in the n_columns.

    Returns
    -------
    n_rows : int
        Number of rows for matplotlib.subplot.
    n_columns : int
        Number of columns for matplotlib.subplot.
    """
    if n < 0:
        raise Exception("Please enter a positive n")

    # use square root to test because we want a square
    # arrangment
    sqrt_n = np.sqrt(n)
    int_sqrt_n = int(sqrt_n)
    if sqrt_n == float(int_sqrt_n):
        return int_sqrt_n, int_sqrt_n
    elif int_sqrt_n*(int_sqrt_n+1)>=n:
        # see if a rectangular orientation would work
        # eg. sqrt(12) = 3.464, int(3.464) = 3
        # 3*(3+1) = 3*4 = 12
        if pref_rows:
            return int_sqrt_n+1, int_sqrt_n
        else:
            return int_sqrt_n, int_sqrt_n+1
    else:
        # since sqrt(n)*sqrt(n) = n,
        # (sqrt(n)+1)*sqrt(n) < n,
        # then (sqrt(n)+1)^2 > n
        return int_sqrt_n+1, int_sqrt_n+1

def log10_1p(x: np.ndarray):
    """Return the log10 of one plus the input array, element-wise.

    Parameters
    ----------
    x : numpy.ndarray
        An array of values greater than -1. If values are less than or
        equal to -1, then a domain error will occur in computing the
        logarithm.

    Returns
    --------
    y : numpy.ndarray
        Array of the values having the log10(element+1) computer.
    """
    y = np.nan*x
    for i, element in enumerate(x):
        y[i] = np.log10(element + 1)
    return y

def scatter_series_axes(data_x, data_y, label: str, color: str, alpha: float,
                        ax = None) -> plt.axes:
    """Creates a scatter axis for plotting.

    Parameters
    ----------
    data_x : array-like
        Data for the x series.
    data_y : array-like
        Data for the y series.
    label : str
        Name for the axes.
    color : str
        Color for the markers.
    alpha : float
        Transparency for the markers.

    Returns
    -------
    matplotlib.axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(data_x, data_y, label = label, color = color, alpha = alpha)
    return ax

#*****************************************************************************************
# General Bias Correction Summary Statistics:
#      pbias_sites
#      diff_maxflow_sites
#      pbias_plotter
#      diff_maxflow_plotter
#      site_diff_scatter
#      stat_corrections_scatter2D
#      anomaly_scatter2D
#      rmseFraePlot
#*****************************************************************************************

def pbias_sites(observed: pd.DataFrame, predicted: pd.DataFrame):
    """Calculates percent bias on a hydrologic year and site-by-site basis.

    Parameters
    ----------
    observed : pandas.DataFrame
        Dataframe containing all observations.
    predicted : pandas.DataFrame
        Dataframe containing all predictions.

    Returns
    -------
    pandas.DataFrame
        Dataframe contain the percent bias computed.
    """
    #finds the start of the first hydraulic year
    i = find_index_water_year(observed)
    water_year_start = observed.index[i]
    #counts the number of hydraylic years
    water_years = 0
    while i < len(observed):
        date = observed.index[i]
        if date.month == 9 and date.day == 30:
            water_years = water_years + 1
        i = i + 1

    pbias_site_df = pd.DataFrame(columns = observed.columns,
                                 index = pd.Series(range(0, water_years)))
    pbias_current_year = pd.DataFrame(columns = observed.columns)

    for i in range(0, water_years):
        #establish end of hydraulic year
        water_year_end = water_year_start + pd.Timedelta(364.25, unit = 'd')

        #need to truncate datetimes since time indicies do not align
        O = observed.loc[water_year_start : water_year_end]
        P = predicted.loc[water_year_start : water_year_end]
        O.index = O.index.floor('d')
        P.index = P.index.floor('d')

        pbias_current_year = dst.pbias(O, P)

        #puts the computations for the hydraulic year into our bigger dataframe
        pbias_site_df.iloc[i] = pbias_current_year.iloc[0].T

        #set up next hydraulic year
        water_year_start = water_year_start + pd.Timedelta(365.25, unit = 'd')

    return pbias_site_df

def diff_maxflow_sites(observed: pd.DataFrame, predicted: pd.DataFrame):
    """Calculates difference in maximum flows on a hydrologic year and site-by-site basis.

    Parameters
    ----------
    observed : pandas.DataFrame
        Dataframe containing all observations.
    predicted : pandas.DataFrame
        Dataframe containing all predictions.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the difference in maximum flows.
    """
    #finds the start of the first hydraulic year
    i = find_index_water_year(observed)
    water_year_start = observed.index[i]

    #counts the number of hydraylic years
    water_years = 0
    while i < len(observed):
        date = observed.index[i]
        if date.month == 9 and date.day == 30:
            water_years = water_years + 1
        i = i + 1

    diff_maxflow_sites_df = pd.DataFrame(columns = observed.columns,
                                         index = pd.Series(range(0,water_years)))
    diff_maxflow_current_year = pd.DataFrame(columns = observed.columns)

    for i in range(0,water_years):
        #establish end of hydraulic year
        water_year_end = water_year_start + pd.Timedelta(364.25, unit = 'd')

        #need to truncate datetimes since time indicies do not align
        O = observed.loc[water_year_start : water_year_end]
        P = predicted.loc[water_year_start : water_year_end]
        O.index = O.index.floor('d')
        P.index = P.index.floor('d')

        diff_maxflow_current_year = P.max().to_frame().T - O.max().to_frame().T

        #puts the computations for the hydraulic year into our bigger dataframe
        diff_maxflow_sites_df.iloc[i] = diff_maxflow_current_year.iloc[0].T

        #set up next hydraulic year
        water_year_start = water_year_start + pd.Timedelta(365.25, unit = 'd')


    return diff_maxflow_sites_df

def pbias_plotter(observed: pd.DataFrame, names: list, colors: list, *models: pd.DataFrame):
    """Plots box plots of numerous models grouped by site.

    Parameters
    ----------
    observed : pandas.Dataframe
        Dataframe containing observations.
    names  : list
        List of the model names.
    colors: list
        List of colors to be plotted.
    *models : List[pandas.DataFrame]
        Any number of pandas.DataFrame objects to be evaluated.

    Returns
    -------
    matplotlib.figure, matplotlib.axes
    """
    num_models = len(models)
    sites = observed.columns
    pbias_models = list()

    #runs each model through pbias_sites
    for model in models:
        pbias_models.append(pbias_sites(observed, model))

    fig = plt.figure()
    ax = plt.axes(xlabel = 'Sites', ylabel = 'Percent Bias')
    position = 1
    for site in sites:
        df = pd.DataFrame(index = pbias_models[0].index)

        #fill out the dataframe for a single site with each model's percent bias
        i = 0
        for model in pbias_models:
            entry = f"{site}:{names[i]}"
            df[entry] = model[site]
            i = i + 1

        boxplots = plt.boxplot(df.T, positions = np.arange(position, position + num_models),
                               patch_artist=True)

        for patch, color in zip(boxplots['boxes'], colors):
            patch.set_facecolor(color)

        position = position + num_models + 1

    tick_location = list()
    start_tick = int(np.ceil(len(models) / 2))
    tick_spacing = len(models) + 1
    for j in range(0, len(sites)):
        tick_location.append(start_tick + j * tick_spacing)

    ax.set(xticks = tick_location, xticklabels = sites)
    plt.xticks(rotation = 90)

    ax.legend(handles=custom_legend(names, colors), loc='upper left')
    return fig, ax

def diff_maxflow_plotter(observed: pd.DataFrame, names: list, colors: list, *models: pd.DataFrame):
    """Plots box plots of numerous models grouped by site.

    Parameters
    ----------
    observed : pandas.Dataframe
        a dataframe containing observations
    names : list
        List of the model names.
    colors: list
        List of colors to be plotted.
    *models : List[pandas.DataFrame]
        Any number of pandas.DataFrame objects to be evaluated.

    Returns
    -------
    matplotlib.figure, matplotlib.axes
    """
    num_models = len(models)
    sites = observed.columns
    diff_maxflow_models = list()

    #runs each model through pbSites
    for model in models:
        diff_maxflow_models.append(diff_maxflow_sites(observed, model))

    fig = plt.figure()
    ax = plt.axes(xlabel = 'Sites', ylabel = 'Difference in Max Flow')
    position = 1
    for site in sites:
        df = pd.DataFrame(index = diff_maxflow_models[0].index)

        #fill out the dataframe for a single site with each model's percent bias
        for i, model in enumerate(diff_maxflow_models):
            entry = f"{site}:{names[i]}"
            df[entry] = model[site]

        bp = plt.boxplot(df.T, positions = np.arange(position, position + num_models),
                         patch_artist = True)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        position = position + num_models + 1
        #follow plot style: https://stackoverflow.com/questions/16592222/matplotlib-group-boxplots

    tick_location = list()
    start_tick = int(np.ceil(len(models) / 2))
    tick_spacing = len(models) + 1
    for j in range(0, len(sites)):
        tick_location.append(start_tick + j * tick_spacing)

    ax.set(xticks = tick_location, xticklabels = sites)
    plt.xticks(rotation = 45)
    plt.title('Yearly Difference in Max Flow due to Correction')

    ax.legend(handles = custom_legend(names, colors), loc = 'upper left')

    return fig, ax

def site_diff_scatter(predictions: dict, raw_key: str, model_keys: list,
                      compare: dict, compare_key: str, site: str, colors = colors99p99):
    """Creates a scatter plot of Raw-BC versus some measure.

    Parameters
    ----------

    predictions : dict
        Expects {'Prediction Names' : Prediction pandas.DataFrame}.
        'Prediction Names' will be printed in the legend.
    raw_key : str
        The key for the predictions dictionary that directs to the
        raw data that each model will be subtracting.
    model_keys : list
        A list of dictoionary keys pertaining to the correction models
        that are wanting to be plotted.
    compare : dict
        Expecting {'Measure name' : measure pandas.DataFrame}.
        These are what is being plotted against on the horizontal-axis.
    compare_key : str
        The dictionary key for the measure desired in the compare dictionary.
        'compare_key' will be printed on the horizontal axis.
    site : str
        A single site designiation to be examined in the plot. This will
        be listed as the title of the plot.
    colors : List[str], optional
        Colors as strings to be plotted from.

    Returns
    -------
    matplotlib.figure, matplotlib.axes

    """
    #retreiving DataFrames and establishing data to be plotted
    raw = predictions[raw_key]
    raw = raw.loc[:, site]

    Y = list()
    for model_key in model_keys:
        predict = predictions[model_key]
        Y.append(raw - predict.loc[:, site])

    X = compare[compare_key]
    X = X.loc[:, site]
    fig,ax = plt.subplots()
    for i,y in enumerate(Y):
        scatter_series_axes(X, y, model_keys[i], color = colors[i], alpha = 0.05, ax = ax)
    plt.xlabel(compare_key)
    plt.ylabel('Raw-BC')
    plt.title(site)
    plt.axhline(0)
    plt.legend(handles = custom_legend(model_keys, colors))

    return fig, ax

def stat_corrections_scatter2D(computations: dict, baseline_key: str, cor_keys: list, uncor_key: str,
                               sites = [], multi = True, colors = colors99p99):
    """Creates a scatter plot of the flow before/after corrections relative to observations.

    Parameters
    ----------
    computations : dict
        Expecting {"Correction Name": correction pandas.DataFrame}.
    baseline_key : str
        Contains the dictionary key for the `computations` dictionary
        that accesses what baseline the corrections should be
        compared to. This is typically observations.
    cor_keys : list
        Dictionary keys accessing the correction DataFrames in `computations`.
        These will be printed in the legend.
    uncor_key : str
        The dictionary key that accesses the uncorrected data in `computations`.
    sites : list
        Site(s) to be compared in the plot, can have a size of 1.
        If `multi` is set to False and this is not changed to a
        single site, then the first value in the list will be
        chosen.
    multi : boolean, optional
        Determines whether the plot uses data from multiple sites or a
        single site, defaults as True.
    colors : List[str], optional
        Colors as strings to be plotted from.
        Plotting colors are different for each correction
        DataFrame, but same across sites for a singular
        correction. An error will be thrown if there are
        more `cor_keys` then `colors`.

    Returns
    -------
    matplotlib.figure, matplotlib.axes

    """
    #retreiving DataFrames and establishing data to be plotted
    datum = computations[baseline_key]
    X = datum - computations[uncor_key]

    #we need to make the values replacable by the sites,
    #hence making the max the min and the min the max
    xmax = X.min().min()
    xmin = X.max().max()

    ymax = 0
    ymin = 100

    #thrown in are Smax and Smin to determine what the max
    #and min values overall are so that the axi may be
    #appropriately scaled
    fig,ax = plt.subplots()
    for i, cor_key in enumerate(cor_keys):
        Y = datum - computations[cor_key]


        if multi == True:
            for site in sites:
                    x = X.loc[:, site]
                    y = Y.loc[:, site]

                    xmax_site = x.max()
                    xmin_site = x.min()
                    ymax_site = y.max()
                    ymin_site = y.min()
                    if xmax_site > xmax:
                        xmax = xmax_site
                    if xmin_site < xmin:
                        xmin = xmin_site
                    if ymax_site > ymax:
                        ymax = ymax_site
                    if ymin_site < ymin:
                        ymin = ymin_site

                    scatter_series_axes(x, y, site, colors[i], 0.05, ax)
        else: #meaning site should be set to a singular value
            #double check that this was actually changed, otherwise picks first value
            site = sites
            if isinstance(sites, list):
                site = sites[0]

            x = X.loc[:, site]
            y = Y.loc[:, site]
            xmax_site = x.max()
            xmin_site = x.min()
            ymax_site = y.max()
            ymin_site = y.min()
            if xmax_site > xmax:
                xmax = xmax_site
            if xmin_site < xmin:
                xmin = xmin_site
            if ymax_site > ymax:
                ymax = ymax_site
            if ymin_site < ymin:
                ymin = ymin_site

            scatter_series_axes(x, y, site, colors[i], 0.05, ax)


    #Sets up labels based on whether one or more sites were plotted
    plt.xlabel(f'{baseline_key}-Uncorrected')
    plt.ylabel(f'{baseline_key}-Corrected')

    if multi==True:
        plt.title("Statistical Corrections")
    else:
        plt.title(f'Statistical Corrections: {sites}')

    minlin = xmin
    maxlin = xmax

    if ymin < minlin:
        minlin = ymin
    if ymax > maxlin:
        maxlin = ymax

    minlin = minlin * 0.9
    maxlin = maxlin * 1.1

    plt.plot([minlin, maxlin], [minlin, maxlin])
    plt.legend(handles = custom_legend(cor_keys, colors))

    return fig, ax

def anomaly_scatter2D(computations: dict, baseline_key: str, vert_key: str, horz_key: str,
                      sites = [], multi = True, colors = colors99p99, show_legend = True):
    """Plots two correction models against each other after Raw is subracted from each.

    Parameters
    ----------
    computations : dict
        Expecting {"Correction Name": correction pandas.DataFrame}.
    baseline_key : str
        Dictionary key for the `computations` dictionary
        that accesses what baseline the corrections should be
        compared to. This is typically observations.
    vert_key : str
        Dictionary key for the `computations` dictionary that accesses the
        model to be plotted on the vertical axis.
    horz_key : str
        Dictionary key for the `computations` dictionary that accesses the
        model to be plotted on the horizontal axis.
    sites : list
        Site(s) to be compared in the plot, can have a size of 1.
        If `multi` is set to False and this is not changed to a
        single site, then the first value in the list will be
        chosen.
    multi : boolean, optional
        Whether the plot uses data from multiple sites or a single site.
    colors : list, optional
        Colors as strings to be plotted from. Plotting colors are different
        for each correction DataFrame, but same across sites for a singular
        correction. An error will be thrown if there are  more `cor_keys` then
        colors.
    show_legend : boolean, optional
        Whether or not to display the legend, defaults as True.
    """
    #retreiving DataFrames and establishing data to be plotted
    datum = computations[baseline_key]
    X = datum - computations[horz_key]
    Y = datum - computations[vert_key]

    i = 0
    fig, ax = plt.subplots()

    if multi == True:
        for site in sites:
            x = X.loc[:, site]
            y = Y.loc[:, site]
            scatter_series_axes(x, y, site, colors[i], 0.05, ax)
            i = i + 1

            if i >= len(colors):
                #recycles colors if all exhausted
                i = 0
    else: #meaning site should be set to a singular value
        #double check that this was actually changed, otherwise picks first value
        site = sites
        if type(sites) == list:
            site = sites[0]

        X = X.loc[:, site]
        Y = Y.loc[:, site]
        scatter_series_axes(X, Y, site, colors[i], 0.05, ax)

    plt.xlabel(f'{baseline_key}-{horz_key}')
    plt.ylabel(f'{baseline_key}-{vert_key}')
    plt.title("Statistical Correction Anomolies")
    plt.axhline(0)
    plt.axvline(0)
    if show_legend:
        plt.legend(handles = custom_legend(sites, colors))

def rmseFracPlot(data_dict: dict, obs_key: str, sim_keys: list,
                sites = [], multi = True, colors = colors99p99):
    """Root mean square values calculated by including descending values one-by-one.

    Parameters
    ----------
    data_dict : dict
        Expecting {"Data Name": data pandas.DataFrame}.
    obs_key : str
        Dictionary key for the `computations` dictionary that accesses the
        observations to be used as true in calculating root mean squares.
    sim_keys : list
        Dictionary keys accessing the simulated DataFrames in `computations`,
        used in predictions in calculating root mean squares.
    sites : list
        Site(s) to be compared in the plot, can have a size of 1.
        If `multi` is set to False and this is not changed to a
        single site, then the first value in the list will be
        chosen.
    multi : boolean, optional
        Whether the plot uses data from multiple sites or a single site.
    colors : list, optional
        Colors as strings to be plotted from. Plotting colors are different
        for each correction DataFrame, but same across sites for a singular
        correction. An error will be thrown if there are  more `sim_keys` then
        colors.
    """
    #retrieving data and flooring time stamps
    observations = data_dict[obs_key]
    observations.index = observations.index.floor('d')

    color_num = 0
    for sim_key in sim_keys:
        predictions = data_dict[sim_key]
        predictions.index = predictions.index.floor('d')

        N = len(predictions.index)
        rmse_tot = dst.rmse(observations, predictions)
        rmse_n = pd.DataFrame(index = np.arange(0, N))

        errors = predictions - observations
        square_errors = errors.pow(2)
        if multi == True:

            #constructs a dataframe where each column is independently sorted
            for site in sites:
                square_errors_site = square_errors.loc[:, site].sort_values(ascending = False)
                rmse_n[site]=square_errors_site.values


            #performs cumulative mean calcualtion
            for site in sites:
                vals = rmse_n[site].values
                mat = np.vstack([vals,] * N).T
                nantri = np.tri(N)
                nantri[nantri == 0] = np.nan
                mat_mean = np.nanmean(mat * nantri, axis = 1)
                mat_rmse = np.power(mat_mean, 0.5)
                rmse_n[site] = np.divide(mat_rmse, rmse_tot[site].values)
        else:
            site = sites
            if type(sites) == list:
                site = sites[0]

            square_errors_site = square_errors.loc[:, site].sort_values(ascending = False)
            rmse_n[site] = square_errors_site.values
            vals = rmse_n[site].values
            mat = np.vstack([vals,] * N).T
            nantri = np.tri(N)
            nantri[nantri == 0] = np.nan
            mat_mean = np.nanmean(mat * nantri, axis=1)
            mat_rmse = np.power(mat_mean, 0.5)
            rmse_n[site] = np.divide(mat_rmse, rmse_tot[site].values)

        rmse_n.index = rmse_n.index / N
        plt.plot(rmse_n, color = colors[color_num], alpha = 0.5)
        color_num = color_num + 1

    plt.xlabel('n/N')
    plt.xscale('log')
    plt.ylabel('RMSE_Cumulative/RMSE_total')
    plt.yscale('log')

    if multi == True:
        plt.title('RMSE Distribution in Descending Sort')
    else:
        plt.title(f'RMSE Distribution in Descending Sort: {site}')
    plt.axhline(1)
    plt.legend(handles = custom_legend(sim_keys , colors))

#*****************************************************************************************
# SimpleRiverNetwork Plots and Related NetworkX Functions:
#      find_upstream
#      find_all_upstream
#      create_adj_mat
#      create_nxgraph
#      organize_nxgraph
#      color_code_nxgraph
#      draw_dataset
#*****************************************************************************************

def find_upstream(topo: xr.Dataset, segID: int, return_segs: list = []):
    """Finds what river segment is directly upstream from the xarray.Dataset.

    Parameters
    ----------
    topo : xarray.Dataset
        Contains river network topography. Expecting each river segment's
        immeditate downstream river segment is designated by 'Tosegment'/
    segID : int
        Current river segment identification number.
    return_segs : list
        River segment identification numbers upstream from `segID`.
        This defaults as an empty list to be filled by the method.
    """
    upsegs = np.argwhere((topo['Tosegment'] == segID).values).flatten()
    upsegIDs = topo['seg_id'][upsegs].values
    return_segs += list(upsegs)

def find_all_upstream(topo: xr.Dataset, segID: int, return_segs: list = []) -> np.ndarray:
    """Finds all upstream river segments for a given river segment from the xarray.Dataset.

    Parameters
    ----------
    topo : xarray.Dataset
    segID : int
    return_segs : list

    Returns
    -------
    numpy.ndarray
    """
    upsegs = np.argwhere((topo['Tosegment'] == segID).values).flatten()
    upsegIDs = topo['seg_id'][upsegs].values
    return_segs += list(upsegs)
    for upsegID in upsegIDs:
        find_all_upstream(topo, upsegID, return_segs = return_segs)
    return np.unique(return_segs).flatten()

def create_adj_mat(topo: xr.Dataset) -> np.ndarray:
    """Forms the adjacency matrix for the graph of the topography.

    Note that this is independent of whatever the segments are called, it
    is a purely a map of the relative object locations.
    Parameters
    ----------
    topo : xarray.Dataset
        Describes the topograph of the river network.

    Returns
    --------
    numpy.ndarray
        An adjacency matrix describing the river network.
    """
    #creates the empty adjacency matrix
    N = topo.dims['seg']
    adj_mat = np.zeros(shape=(N, N), dtype = int)

    #builds adjacency matrix based on what segements are upstream
    i = 0
    for ID in topo['seg_id'].values:
        adj = list()
        find_upstream(topo, ID, adj)
        for dex in adj:
            adj_mat[i, dex] += 1
        i += 1
    return adj_mat

def create_nxgraph(adj_mat: np.ndarray) -> nx.Graph:
    """Creates a NetworkX Graph object given an adjacency matrix.

    Parameters
    ----------
    adj_mat : numpy.ndarray
        Adjacency matrix describing the river network.

    Returns
    -------
    networkx.graph
        NetworkX Graph of respective nodes.
    """
    topog = nx.from_numpy_matrix(adj_mat)
    return topog

def organize_nxgraph(topo: nx.Graph):
    """Orders the node positions into a hierarchical structure.

    Based on the "dot" layout and given topography.
    Parameters
    ----------
    topo : xarray.Dataset
        Contains river segment identifications and relationships.

    Returns
    -------
    networkx.positions
    """
    pos = nx.drawing.nx_agraph.graphviz_layout(topo, prog = 'dot')
    return pos

def color_code_nxgraph(graph: nx.graph, measure: pd.Series,
                       cmap=mpl.cm.get_cmap('coolwarm_r'),
                       vmin=None, vmax=None)-> dict:
    """Creates a dictionary mapping of nodes to color values.

    Parameters
    ----------
    graph : networkx.graph
        Graph to be color coded
    measure : pandas.Series
        Contains river segment ID's as the index and desired measures as
        values.
    cmap : matplotlib.colors.LinearSegmentedColormap, optional
        Colormap to be used for coloring the SimpleRiverNewtork
        plot. This defaults as 'coolwarm_r'.
    vmin: float, optional
        Minimum value for coloring
    vmax: float, optional
        Maximum value for coloring

    Returns
    -------
    dict
        Dictionary of {i:color} where i is the index of the river
        segment.

    """
    if np.where(measure < 0)[0].size == 0:
        # meaning we have only positive values and do not need to establish
        # zero at the center of the color bar
        segs = measure.index
        minimum = 0 #set to zero to preserve the coloring of the scale
        maximum = measure.max()

        color_vals = (measure.values) / (maximum)
        color_bar = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = minimum, vmax = maximum))

        color_dict =  {f'{seg}' : mpl.colors.to_hex(cmap(i)) for i, seg in zip(color_vals, segs)}
        return color_dict, color_bar
    else:
        segs = measure.index
        #determine colorbar range
        if vmin is None and vmax is None:
            extreme = np.max(np.abs([np.min(measure), np.max(measure)]))
            vmin = -extreme
            vmax = extreme
        elif vmin is None:
            vmin = measure.min()
        elif vmax is None:
            vmax = measure.max()

        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        color_bar = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

        color_dict =  {f'{seg}': mpl.colors.to_hex(cmap(norm(i)))
                       for i, seg in zip(measure.values, segs)}
        return color_dict, color_bar

def draw_dataset(topo: xr.Dataset, color_measure: pd.Series, cmap = mpl.cm.get_cmap('coolwarm_r')):
    """"Plots the river network through networkx.

    Draws a networkx graph from a topological xrarray.Dataset and color codes
    it based on a pandas.Series.

    Parameters
    ----------
    topo : xarray.Dataset
        Contains river segment identifications and relationships.
    color_measure : pandas.Series
        Indicies are concurrent with the number of segs in topo.
        Typically this contains statistical information about the flows
        that will be color coded by least to greatest value.
    cmap : matplotlib.colors.LinearSegmentedColormap, optional
        Colormap to be used for coloring the SimpleRiverNewtork
        plot. This defaults as 'coolwarm_r'.
    """
    topo_adj_mat = create_adj_mat(topo)
    topo_graph = create_nxgraph(topo_adj_mat)
    topo_positions = organize_nxgraph(topo_graph)
    topo_color_dict, topo_color_cbar = color_code_nxgraph(topo_graph, color_measure, cmap)
    topo_nodecolors = [topo_color_dict[f'{node}'] for node in topo_graph.nodes()]
    nx.draw_networkx(topo_graph, topo_positions, node_size = 200, font_size = 8, font_weight = 'bold',
                     node_shape = 's', linewidths = 2, font_color = 'w', node_color = topo_nodecolors)
    plt.colorbar(topo_color_cbar)

#*****************************************************************************************
# BMORPH Summary Statistics:
#      plot_reduced_flows
#      plot_spearman_rank_difference
#      correction_scatter
#      pbias_diff_hist
#      plot_residual_overlay
#      norm_change_annual_flow
#      pbias_compare_hist
#      compare_PDF
#      compare_CDF
#      spearman_diff_boxplots_annual
#      kl_divergence_annual_compare
#      spearman_diff_boxplots_annual_compare
#      compare_CDF_all
#      compare_mean_grouped_CPD
#*****************************************************************************************

def plot_reduced_flows(flow_dataset: xr.Dataset, plot_sites: list,
                        reduce_func=np.mean, interval = "day",
                        statistic_label='Mean',
                        units_label = r'$(m^3/s)$',
                        title_label=f'Annual Mean Flows',
                        raw_var = 'IRFroutedRunoff', raw_name = 'Mizuroute Raw',
                        ref_var = 'upstream_ref_flow', ref_name = 'upstream_ref_flow',
                        bc_vars = list(), bc_names = list(),
                        fontsize_title = 24, fontsize_legend = 20, fontsize_subplot = 20,
                        fontsize_tick = 20, fontcolor = 'black',
                        figsize_width = 20, figsize_height = 12,
                        plot_colors = ['grey', 'black', 'blue', 'red'],
                        return_reduced_flows = False):
    """Creates a series of subplots plotting statistical day of year flows per gauge site.

    Parameters
    ----------
    flow_dataset : xarray.Dataset
        Contatains raw, reference, and bias corrected flows.
    plot_sites : list
        Sites to be plotted.
    reduce_func : function, optional
        A function to apply to flows grouped by `interval`, defaults as np.mean.
    interval : str, optional
        What time interval annual `reduce_func` should be computed on. Currently supported
        is `day` for dayofyear (default), `week` for weekofyear, and `month` for monthly.
    statistic_label : str, optional
        Label for the statistic representing the `reduce_func`, defaults as
        'Mean' to fit `reduce_func` as np.mean.
    units_label : str, optional
        Label for the units of flow, defaults as r`$(m^3/s)$`.
    title_label : str
        Lable for the figure title representing the reduce_func, defaults as
        f'Annual Mean Flows' to fit `reduce_func` as np.mean.
    raw_var : str, optional
        The string to access the raw flows in `flow_dataset`, defaults as
        'IRFroutedRunoff'.
    raw_name : str, optional
        Label for the raw flows in the legend, defaults as 'Mizuroute Raw'.
    ref_var : str, optional
        The string to access the reference flows in `flow_dataset`, defaults
        as 'upstream_ref_flow'.
    ref_name : str, optional
        Label for the reference flows in the legend, defaults as 'upstream_ref_flow'.
    bc_vars : list
        The strings to access the bias corrected flows in `flow_dataset`.
    bc_names : list
        Labels for the bias corrected flows in the legend, expected in the same
        order as `bc_vars`.
    plot_colors : list, optional
        Colors to be plotted for `raw_var`, `ref_var`, `bc_vars` respectively.
        Defaults as ['grey', 'black', 'blue', 'red'].
    return_reduced_flows : boolean, optional
        If True, returns the reduced flows as calculated for plotting, defaults
        as False. This is typically used for debugging purposes.
    fontsize_title : int, optional
        Font size of the plot title, defaults as 80.
    fontsize_legend : int, optional
        Font size of the plot legend, defaults as 68.
    fontsize_subplot : int, optional
        Font size for the subplots, defaults as 60.
    fontsize_tick : int, optional
        Font size of the ticks, defaults 45.
    fontcolor : str, optional
        Color of the font, defaults as 'black'
    figsize_width : int, optional
        Width of the figure, defaults as 70.
    figusize_height : int, optional
        Height of the figure, defaults as 30.

    Returns
    -------
    xarray.Dataset or (matplotlib.figure, matplotlib.axes)
        If `return_reduced_flows` is False, matplotlib.figure and matplotlib.axes,
        otherwise the reduced_flows are returned as the xarray.Dataset.
    """

    if len(bc_vars) == 0:
        raise Exception("Please enter a non-zero number strings in bc_vars to be used")
    if len(bc_vars) != len(bc_names):
        raise Exception("Please have the same number of entries in bc_names as bc_names")
    if len(plot_colors) < 2 + len(bc_vars):
        raise Exception(f"Please enter at least {2 + len(bc_vars)} colors in plot_colors")

    interval = interval.lower()
    interval_name = "Day of Year"
    if interval == 'day':
        interval_name = "Day"
        raw_flow = flow_dataset[raw_var].groupby(
            flow_dataset['time'].dt.dayofyear).reduce(reduce_func)
        reference_flow = flow_dataset[ref_var].groupby(
            flow_dataset['time'].dt.dayofyear).reduce(reduce_func)
        bc_flows = list()
        for bc_var in bc_vars:
            bc_flows.append(flow_dataset[bc_var].groupby(flow_dataset['time'].dt.dayofyear).reduce(reduce_func))
        time = raw_flow['dayofyear'].values
    elif interval == 'week':
        interval_name = "Week of Year"
        raw_flow = flow_dataset[raw_var].groupby(
            flow_dataset['time'].dt.isocalendar().week).reduce(reduce_func)
        reference_flow = flow_dataset[ref_var].groupby(
            flow_dataset['time'].dt.isocalendar().week).reduce(reduce_func)
        bc_flows = list()
        for bc_var in bc_vars:
            bc_flows.append(flow_dataset[bc_var].groupby(flow_dataset['time'].dt.isocalendar().week).reduce(reduce_func))
        time = raw_flow['week'].values
    elif interval == 'month':
        interval_name = "Month"
        raw_flow = flow_dataset[raw_var].groupby(
            flow_dataset['time'].dt.month).reduce(reduce_func)
        reference_flow = flow_dataset[ref_var].groupby(
            flow_dataset['time'].dt.month).reduce(reduce_func)
        bc_flows = list()
        for bc_var in bc_vars:
            bc_flows.append(flow_dataset[bc_var].groupby(flow_dataset['time'].dt.month).reduce(reduce_func))
        time = raw_flow['month'].values
    else:
        raise Exception("Please use 'day', 'week', or 'month' for interval.")


    outlet_names = flow_dataset['seg'].values

    raw_flow_df = pd.DataFrame(data = raw_flow.values,
                                   index=time, columns = outlet_names)
    reference_flow_df = pd.DataFrame(data = reference_flow.values,
                                         index=time, columns = outlet_names)
    bc_flow_dfs = list()
    for bc_flow in bc_flows:
        bc_flow_dfs.append(pd.DataFrame(data = bc_flow.values, index = time, columns = outlet_names))

    plot_names = [raw_name, ref_name]
    plot_names.extend(bc_names)

    mpl.rcParams['figure.figsize'] = (figsize_width, figsize_height)
    n_rows, n_cols = determine_row_col(len(plot_sites))
    fig, axs = plt.subplots(n_rows, n_cols)

    for site, ax in zip(plot_sites, axs.ravel()):
        ax.plot(raw_flow_df[site], color = plot_colors[0], alpha = 0.8, lw = 4)
        ax.plot(reference_flow_df[site], color = plot_colors[1], alpha = 0.8, lw = 4)

        for i, bc_flow_df in enumerate(bc_flow_dfs):
            ax.plot(bc_flow_df[site], color = plot_colors[2+i], lw = 4, alpha = 0.8)

        ax.set_title(site, fontsize = fontsize_subplot, color = fontcolor)
        plt.setp(ax.spines.values(), color = fontcolor)
        ax.tick_params(axis = 'both', colors = fontcolor, labelsize = fontsize_tick)

    if len(plot_sites) < n_rows * n_cols:
        for ax_index in np.arange(len(plot_sites), n_rows * n_cols):
            axs.ravel().tolist()[ax_index].axis('off')

    fig.text(0.4, -0.02, interval_name, fontsize = fontsize_title, ha = 'center')
    fig.text(-0.02, 0.5, f"{statistic_label} {interval_name} Flow {units_label}", fontsize = fontsize_title,
             va = 'center', rotation = 'vertical')
    plt.subplots_adjust(wspace = 0.1, hspace = 0.3, left = 0.05, right = 0.8, top = 0.95)

    fig.legend(plot_names, fontsize = fontsize_legend, loc = 'center right');

    if return_reduced_flows:
        reduced_flows = xr.Dataset(coords={'site': plot_sites, 'time': time})
        reduced_flows[raw_var] = xr.DataArray(data = raw_flow_df.values, dims = ('time', 'site') ).transpose()
        reduced_flows[ref_var] = xr.DataArray(data = reference_flow_df.values, dims = ('time', 'site') ).transpose()
        for bc_var, bc_flow_df in zip(bc_vars, bc_flow_dfs):
            reduced_flows[bc_var] = xr.DataArray(data = bc_flow_df.values, dims = ('time', 'site') ).transpose()
        return reduced_flows

    return fig, ax


def plot_spearman_rank_difference(flow_dataset:xr.Dataset, gauge_sites: list,
                                  start_year: str, end_year: str,
                                  relative_locations_triu: pd.DataFrame, basin_map_png,
                                  cmap = mpl.cm.get_cmap('coolwarm_r'),
                                  blank_plot_color = 'w', fontcolor = 'black',
                                  fontsize_title=60, fontsize_tick = 30, fontsize_label = 45):
    """Creates a site-to-site rank correlation difference comparison plot with a map of the basin.

    Parameters
    ----------
    flow_dataset : xarray.Dataset
        Contains raw as 'raw_flow' and bias corrected as 'bias_corrected_total_flow'
        flow values, times of the flow values, and the names of the sites where those
        flow values exist
    gauge_sites : list
        Gauge sites to be plotted.
    start_year : str
        String formatted as 'yyyy-mm-dd' to start rank correlation window.
    end_year : str
        String formatted as 'yyyy-mm-dd' to end rank correlation window.
    relative_locations_triu : pandas.DataFrame
        Denotes which sites are connected with a '1' and has the lower
        triangle set to '0'.
    basin_map_png : png file
        The basin map with site values marked.
    cmap : matplotlib.colors.LinearSegmentedColormap, optional
        Colormap to be used for coloring the SimpleRiverNewtork
        plot. This defaults as 'coolwarm_r'.
    blank_plot_color : str, optional
        Color to set the lower extremes in `cmap` should be to keep extreme values from
        skewing the color map and hiding values. This is defaulted as 'w' for white. It
        should appear that this color matches the background plot color to make it appear
        as if no value is plotted here.
    font_color : str, optional
        Color of the font, defaulted as 'black'.
    fontsize_title : int, optional
        Font size of the title, defaults as 60.
    fontsize_tick : int, optional
        Font size of the ticks, defaults as 30.
    fontsize_label : int, optional
        Font size of the labels, defaults as 45.
    """

    mpl.rcParams['figure.figsize'] = (40, 20)
    cmap.set_under(color = blank_plot_color)

    fig, ax = plt.subplots()

    if int(end_year[:4]) - int(start_year[:4]) == 1:
        fig.suptitle(f'WY{start_year[:4]}: 'r'$r_{s}(Q_{raw}) - r_{s}(Q_{bc})$',
                     fontsize = fontsize_title, color = fontcolor, x = 0.6, y = 0.95)
    elif int(end_year[:4])-int(start_year[:4]) > 1:
        end_WY = int(end_year[:4])-1
        fig.suptitle(f'WY{start_year[:4]} to WY{end_WY}: 'r'$r_{s}(Q_{raw}) - r_{s}(Q_{bc})$',
                     fontsize = fontsize_title, color = fontcolor, x = 0.6, y = 0.95)
    else:
        raise Exception('Please check end_year is later than start_year')

    time_span = flow_dataset['time'].values
    outlet_names = flow_dataset['outlet'].values

    raw_flow = pd.DataFrame(data = np.transpose(flow_dataset['raw_flow'].values),
                            index = time_span, columns = outlet_names)
    ref_flow = pd.DataFrame(data = flow_dataset['reference_flow'].values,
                            index = time_span, columns = outlet_names)
    bc_flow = pd.DataFrame(data=np.transpose(flow_dataset['bias_corrected_total_flow'].values),
                           index = time_span, columns = outlet_names)

    raw_flow_spearman = filter_rank_corr(raw_flow.loc[start_year : end_year].corr(method = 'spearman'),
                                         rel_loc = relative_locations_triu)
    bc_flow_spearman = filter_rank_corr(bc_flow.loc[start_year : end_year].corr(method = 'spearman'),
                                        rel_loc = relative_locations_triu)
    raw_minus_bc_spearman = raw_flow_spearman - bc_flow_spearman

    vmin = np.min(raw_minus_bc_spearman.min())
    vmax = np.max(raw_minus_bc_spearman.max())
    vextreme = vmax
    if np.abs(vmin) > vextreme:
        vextreme = np.abs(vmin)

    # -10 is used to ensure that the squares not to be plotted are marked as such by
    # the cmap's under values
    vunder = np.abs(vmin) * -10

    im = ax.imshow(raw_minus_bc_spearman.fillna(vunder), vmin = -vextreme, vmax = vextreme, cmap = cmap)
    plt.setp(ax.spines.values(), color = fontcolor)
    ax.tick_params(axis='both', colors = fontcolor)
    tick_tot = len(gauge_sites)
    ax.set_xticks(np.arange(0, tick_tot, 1.0))
    ax.set_yticks(np.arange(0, tick_tot, 1.0))
    ax.set_ylim(tick_tot - 0.5, -0.5)
    ax.set_xticklabels(gauge_sites, rotation = 'vertical', fontsize = fontsize_label)
    ax.set_yticklabels(gauge_sites, fontsize = fontsize_tick);

    cb = fig.colorbar(im, pad = 0.01)
    cb.ax.yaxis.set_tick_params(color = fontcolor, labelcolor = fontcolor, labelsize = fontsize_tick)
    cb.outline.set_edgecolor(None)

    fig.text(0.6, -0.02, 'Site Abbreviation', fontsize = fontsize_label, ha = 'center')
    fig.text(0.32, 0.4, 'Site Abbreviation', fontsize = fontsize_label, va = 'center', rotation = 'vertical')

    newax = fig.add_axes([0.295, 0.4, 0.49, 0.49], anchor = 'NE', zorder = 2)
    newax.imshow(basin_map_png)
    newax.axis('off')
    newax.tick_params(axis = 'both')
    newax.set_xticks([])
    newax.set_yticks([])

    plt.tight_layout
    plt.show()

def correction_scatter(site_dict:dict, raw_flow:pd.DataFrame,
                       ref_flow:pd.DataFrame, bc_flow:pd.DataFrame,
                       colors:list, title='Flow Residuals',
                       fontsize_title=80, fontsize_legend=68,
                       fontsize_subplot=60, fontsize_tick=45, fontcolor='black',
                       pos_cone_guide=False, neg_cone_guide=False):
    """Difference from reference flows before and after correction.

    Plots differences between the raw and reference flows on the horizontal
    and differences between the bias corrected and refrerence on the vertical.
    This compares corrections needed before and after the bias correction method
    is applied.

    Parameters
    ----------
    site_dict : dict
        Expects {subgroup name: list of segments in subgroup} how sites are to be
        seperated.
    raw_flow : pandas.DataFrame
        Contains flows before correction.
    ref_flow : pandas.DataFrame
        Contains the reference flows to compare `raw_flow` and `bc_flow`.
    bc_flow : pandas.DataFrame
        Contains flows after correction.
    colors : list
        Colors to be plotted for each site in `site_dict`.
    title : str, optional
        Title label for the plot, defaults as 'Flow Residuals'.
    fontsize_title : int, optional
        Fontsize of the title, defaults as 80.
    fontsize_legend : int, optional
        Fontsize of the legend, defaults as 68.
    fontsize_subplot : int, optional
        Fontsize of the subplots, defaults as 60.
    fontsize_tick : int, optional
        Fontsize of the ticks, defaults as 45.
    fontcolor : str, optional
        Color of the font, defaults as 'black'.
    pos_cone_guide : boolean, optional
        If True, plots a postive 1:1 line through the origin for reference.
    neg_cone_guide : boolean, optional
        If True, plots a negative 1:1 line through the origin for reference.
    """
    num_plots = len(site_dict.keys())
    n_rows, n_cols = determine_row_col(num_plots)

    mpl.rcParams['figure.figsize']=(60,40)

    fig,axs = plt.subplots(nrows=n_rows, ncols=n_cols)
    plt.suptitle(title, fontsize= fontsize_title, color=fontcolor, y=1.05)

    ax_list = axs.ravel().tolist()

    before_bc = ref_flow-raw_flow
    after_bc = ref_flow-bc_flow

    for i, site_group_key in enumerate(site_dict.keys()):
        site_group = site_dict[site_group_key]
        group_before_bc = before_bc.loc[:, site_group]
        group_after_bc = after_bc.loc[:, site_group]
        scatter_series_axes(group_before_bc, group_after_bc, label=site_group_key,
                                     alpha=0.05, color=colors[i], ax=ax_list[i])
        ax_list[i].set_title(site_group_key, fontsize=fontsize_subplot)
        plt.setp(ax_list[i].spines.values(), color=fontcolor)
        ax_list[i].tick_params(axis='both', colors=fontcolor, labelsize=fontsize_tick)

    # add horizontal axis at 0 line and hide plots that are not in use
    for i, ax in enumerate(axs.ravel()):
        if i < num_plots:
            bottom, top = ax.get_ylim()
            left, right = ax.get_xlim()
            ref_line_max = np.max([bottom, top, left, right])
            ref_line_min = np.min([bottom, top, left, right])
            ref_line_ext = ref_line_max
            if np.abs(ref_line_min) > ref_line_ext:
                ref_line_ext = np.abs(ref_line_min)
            ax.plot([-ref_line_ext, ref_line_ext], [0,0], color='k', linestyle='--')

            if pos_cone_guide and neg_cone_guide:
                ax.plot([-ref_line_ext, ref_line_ext], [-ref_line_ext, ref_line_max], color='k', linestyle='--')
                ax.plot([-ref_line_ext, ref_line_ext], [ref_line_ext, -ref_line_max], color='k', linestyle='--')
            elif pos_cone_guide:
                ax.plot([0, ref_line_ext], [0, ref_line_ext], color='k', linestyle='--')
                ax.plot([0, ref_line_ext], [0, -ref_line_ext], color='k', linestyle='--')
            elif neg_cone_guide:
                ax.plot([0, -ref_line_ext], [0, ref_line_ext], color='k', linestyle='--')
                ax.plot([0, -ref_line_ext], [0, -ref_line_ext], color='k', linestyle='--')

        else:
            ax.axis('off')

    fig.text(0.5, -0.04, r'$Q_{ref} - Q_{raw} \quad (m^3/s)$', ha='center', va = 'bottom',
             fontsize=fontsize_title, color=fontcolor);
    fig.text(-0.02, 0.5, r'$Q_{ref} - Q_{bc} \quad (m^3/s)$', va='center', rotation = 'vertical',
             fontsize=fontsize_title, color=fontcolor);

    plt.tight_layout()
    plt.show

def compare_correction_scatter(flow_dataset: xr.Dataset, plot_sites:list,
                       raw_var= "raw", raw_name = "Mizuroute Raw",
                       ref_var="ref", ref_name = "Reference",
                       bc_vars = list(), bc_names = list(),
                       plot_colors = ['blue', 'purple','orange','red'],
                       title='Absolute Error in Flow'r'$(m^3/s)$',
                       fontsize_title=80, fontsize_legend=68, alpha = 0.05,
                       fontsize_subplot=60, fontsize_tick=45, fontcolor='black',
                       pos_cone_guide=False, neg_cone_guide=False, symmetry=True):
    """Difference from reference flows before and after correction.

    Plots differences between the raw and reference flows on the horizontal
    and differences between the bias corrected and refrerence on the vertical.
    This compares corrections needed before and after the bias correction method
    is applied.

    Parameters
    ----------
    flow_dataset : xarray.Dataset
        contains raw, reference, and bias corrected flows.
    plot_sites : list
        Sites to be plotted, expected as the `seg` coordinate in `flow_dataset`.
    raw_var : str, optional
        The string to access the raw flows in `flow_dataset`, defaults as `raw`.
    raw_name : str, optional
        Label for the raw flows in the legend, defaults as 'Mizuroute Raw'.
    ref_var : str, optional
        The string to access the reference flows in `ref`, defaults
        as 'upstream_ref_flow'.
    ref_name : str, optional
        Label for the reference flows in the legend, defaults as 'Reference'.
    bc_vars : list
        The strings to access the bias corrected flows in `flow_dataset`.
    bc_names : list
        Labels for the bias corrected flows in the legend, expected in the same
        order as `bc_vars`.
    plot_colors : list, optional
        Colors to be plotted for each site in `plot sites`.
        Defaults as ['blue', 'purple', 'orange, 'red'].
    fontsize_title : int, optional
        Fontsize of the title, defaults as 80.
    fontsize_legend : int, optional
        Fontsize of the legend, defaults as 68.
    fontsize_subplot : int, optional
        Fontsize of the subplots, defaults as 60.
    fontsize_tick : int, optional
        Fontsize of the ticks, defaults as 45.
    fontcolor : str, optional
        Color of the font, defaults as 'black'.
    pos_cone_guide : boolean, optional
        If True, plots a postive 1:1 line through the origin for reference.
    neg_cone_guide : boolean, optional
        If True, plots a negative 1:1 line through the origin for reference.
    symmetry : boolean, optional
        If True, the plot axis are symmetrical about the origin (default).
        If False, plotting limits will minimize empty space while not losing any data.
    """
    if len(bc_vars) == 0:
        raise Exception("Please enter a non-zero number strings in bc_vars to be used")
    if len(bc_vars) != len(bc_names):
        raise Exception("Please have the same number of entries in bc_names as bc_names")
    if len(plot_colors) < len(bc_vars):
        raise Exception(f"Please enter at least {len(bc_vars)} colors in plot_colors")

    num_plots = len(plot_sites)
    n_rows, n_cols = determine_row_col(num_plots)

    mpl.rcParams['figure.figsize']=(60,40)

    fig,axs = plt.subplots(nrows=n_rows, ncols=n_cols)
    plt.suptitle(title, fontsize= fontsize_title, color=fontcolor, y=1.05)

    ax_list = axs.ravel().tolist()

    plot_flows = flow_dataset.sel(seg=plot_sites).copy()
    plot_flows["before_bc"] = plot_flows[ref_var]-plot_flows[raw_var]

    for bc_var in bc_vars:
        plot_flows[f"after_{bc_var}"] = plot_flows[ref_var]-plot_flows[bc_var]

    #we need to figure out which plots have the most spread to make certain we hide the fewest

    spread = list()
    for bc_var in bc_vars:
        spread.append(plot_flows[f"after_{bc_var}"].values.max()-plot_flows[f"after_{bc_var}"].values.min())

    sorted_spread = np.flip(np.sort(spread.copy()))
    spread_ranks = [spread.index(val) for val in sorted_spread]


    for i, site in enumerate(plot_sites):
        before_bc = plot_flows["before_bc"].sel(seg=site).values
        for j in spread_ranks:
            after_bc = plot_flows[f"after_{bc_vars[j]}"].sel(seg=site).values
            scatter_series_axes(before_bc, after_bc, label=bc_names[j],
                                         alpha=alpha, color=plot_colors[j], ax=ax_list[i])
        ax_list[i].set_title(site, fontsize=fontsize_subplot)
        plt.setp(ax_list[i].spines.values(), color=fontcolor)
        ax_list[i].tick_params(axis='both', colors=fontcolor, labelsize=fontsize_tick)

    # add horizontal axis at 0 line and hide plots that are not in use
    for i, ax in enumerate(axs.ravel()):
        if i == len(axs.ravel())-1:
            ax.axis('off')
            ax.legend(handles=custom_legend(names = bc_names, colors=plot_colors), fontsize = fontsize_legend, loc='center')
        if i < num_plots:
            bottom, top = ax.get_ylim()
            left, right = ax.get_xlim()
            ref_line_max = np.max([bottom, top, left, right])
            ref_line_min = np.min([bottom, top, left, right])
            ref_line_ext = ref_line_max
            if np.abs(ref_line_min) > ref_line_ext:
                ref_line_ext = np.abs(ref_line_min)
            ax.plot([-ref_line_ext, ref_line_ext], [0,0], color='k', linestyle='--')

            if pos_cone_guide and neg_cone_guide:
                ax.plot([-ref_line_ext, ref_line_ext], [-ref_line_ext, ref_line_max], color='k', linestyle='--')
                ax.plot([-ref_line_ext, ref_line_ext], [ref_line_ext, -ref_line_max], color='k', linestyle='--')
            elif pos_cone_guide:
                ax.plot([0, ref_line_ext], [0, ref_line_ext], color='k', linestyle='--')
                ax.plot([0, ref_line_ext], [0, -ref_line_ext], color='k', linestyle='--')
            elif neg_cone_guide:
                ax.plot([0, -ref_line_ext], [0, ref_line_ext], color='k', linestyle='--')
                ax.plot([0, -ref_line_ext], [0, -ref_line_ext], color='k', linestyle='--')
            if not symmetry:
                ax.set_xlim(left=left,right=right)
                ax.set_ylim(top=top,bottom=bottom)

        else:
            ax.axis('off')

    fig.text(0.5, -0.04, r'$Q_{ref} - Q_{raw} \quad (m^3/s)$', ha='center', va = 'bottom',
             fontsize=fontsize_title, color=fontcolor);
    fig.text(-0.02, 0.5, r'$Q_{ref} - Q_{bc} \quad (m^3/s)$', va='center', rotation = 'vertical',
             fontsize=fontsize_title, color=fontcolor);
    plt.tight_layout()
    plt.show

def pbias_diff_hist(sites:list, colors:list, raw_flow:pd.DataFrame, ref_flow:pd.DataFrame,
                      bc_flow: pd.DataFrame, grouper=pd.Grouper(freq='M'), total_bins=None,
                      title_freq='Monthly', fontsize_title=90, fontsize_subplot_title=60,
                    fontsize_tick=40, fontsize_labels=84):
    """Histograms of differences in percent bias before/after bias correction.

    Creates a number of histogram subplots by each given site that
    plot the difference in percent bias before and after bias correction.

    Parameters
    ----------
    sites : list
        Sites that are the columns of the flow DataFrames, `raw_flow`,
        `ref_flow`, and `bc_flow`.
    colors : list
        Colors to plot the sites with, (do not have to be different),
        that are used in the same order as the list of sites.
    raw_flow : pandas.DataFrame
        Contains flows before bias correction.
    ref_flow : pandas.DataFrame
        Contains reference flows for comparison as true values.
    bc_flow : pandas.DataFrame
        Contains flows after bias correction.
    grouper : pandas.TimeGrouper, optional
        How flows should be grouped for bias correction. This defaults as
        monthly.
    total_bins : int, optional
        Number of bins to use in the histogram plots. If none specified,
        defaults to the floored square root of the number of pbias difference
        values.
    title_freq : str, optional
        An adjective description of the frequency with which the flows are
        grouped, should align with `grouper`, although there is no check to
        verify this. This defaults as 'Monthly'.
    fontsize_title : int, optional
        Font size of the title. Defaults as 90.
    fontsize_subplot_title : int, optional
        Font size of the subplots. Defaults as 60.
    fontsize_tick : int, optional
        Font size of the ticks. Defaults as 40.
    fontsize_labels : int, optional
        Font size of the labels. Defaults as 84.
    """

    if len(sites) != len(colors):
        raise Exception('Please enter the same number of colors as sites')

    mpl.rcParams['figure.figsize']=(60, 40)

    n_rows,n_cols = determine_row_col(len(sites))

    bc_m_pbias = pbias_by_index(
        observe=ref_flow.groupby(grouper).sum(),
        predict=bc_flow.groupby(grouper).sum())
    raw_m_pbias = pbias_by_index(
        observe=ref_flow.groupby(grouper).sum(),
        predict=raw_flow.groupby(grouper).sum())

    bc_pbias_impact = bc_m_pbias-raw_m_pbias

    if type(total_bins) is type(None):
        total_bins=int(np.sqrt(len(bc_pbias_impact.index)))

    fig, axs = plt.subplots(n_rows, n_cols)
    axs_list = axs.ravel().tolist()

    plt.suptitle(f"Change in Percent Bias from Bias Correction on a {title_freq} Basis",
                 fontsize=fontsize_title, y=1.05)
    i=0
    for site in sites:
        ax = axs_list[i]
        site_bc_pbias_impact=bc_pbias_impact[site]
        impact_extreme = np.max(np.abs(site_bc_pbias_impact))
        bin_step = (2*impact_extreme/total_bins)
        bin_range = np.arange(-impact_extreme, impact_extreme+bin_step, bin_step)

        site_bc_pbias_impact.plot.hist(ax=ax, bins=bin_range, color=colors[i])
        ax.vlines(0, 0, ax.get_ylim()[1], color='k', linestyle='--', lw=4)
        ax.set_xlim(left=-150, right=150)

        ax.set_title(site,fontsize=fontsize_subplot_title)
        ax.set_ylabel("")
        ax.tick_params(axis='both', labelsize=fontsize_tick)
        i+=1

    while i < len(axs_list):
        axs_list[i].axis('off')
        i+=1

    fig.text(0.5, -0.04, r'$(PBias_{BC})-(PBias_{raw}) \quad (\%)$',
             ha='center', va = 'bottom', fontsize=fontsize_labels);
    fig.text(-0.02, 0.5, 'Frequencey',
             va='center', rotation = 'vertical', fontsize=fontsize_labels);
    plt.tight_layout()

def plot_residual_overlay(flows: pd.DataFrame, upstream_sites: list, downstream_site: str,
                          start_year: int, end_year: int, ax=None, fontsize_title=40,
                          fontsize_labels=60, fontsize_tick= 30, linecolor='k', alpha=0.3):
    """Plots flow upstream/downstream residuals overlayed across one year span.

    Plots residuals from each hydrologic year on top of each
    other with a refence line at zero flow. Residuals are calculated as
    downstream flows - sum(upstream flows).

    Parameters
    ----------
    flows : pandas.DataFrame
        All flows to be used in plotting.
    upstream_sites : list
        Strings of the site names stored in `flows` to aggregate.
    downstream_sites : str
        Name of the downstream site stored in flows that will
        have the `upstream_sites` subtracted from it.
    start_year : int
        The starting year to plot.
    end_year : int
        The year to conclude on.
    ax : matplotlib.axes, optional
        Axes to plot on. If none is specified, a new one is created.
    fontsize_title : int, optional
        Font size of the title.
    fontsize_labels : int, optional
        Font size of the labels.
    fontsize_tick : int, optional
        Font size of the ticks.
    linecolor : str, optional
        Color of the lines plotted. Defaults as 'k' for black.
    alpha : float, optional
        Transparency of the lines plotted. Defaults as 0.3 to help
        see how residuals line up across many years.

    Returns
    -------
    matplotlib.axes
    """

    mpl.rcParams['figure.figsize']=(20,20)

    if type(ax) is type(None):
        fig,ax = plt.subplots()

    start_date='-10-01'
    end_date='-09-30'

    year = start_year
    upstream_site_string = ""

    residual_flow = pd.DataFrame(index=flows.index, columns=['Residuals'])

    residual_flow['Residuals'] = flows[downstream_site].values
    for upstream_site in upstream_sites:
        residual_flow['Residuals'] -= flows[upstream_site].values
        upstream_site_string += f'{upstream_site}, '

    upstream_site_string = upstream_site_string[:len(upstream_site_string)-2]

    while year < end_year:
        values = residual_flow[f'{str(year)}{start_date}':f'{str(year+1)}{end_date}'].values
        doy = np.arange(1, len(values)+1, 1)
        ax.plot(doy, values, color=linecolor, alpha=alpha)
        year += 1

    ax.plot([1,366],[0,0], color='k', linestyle='--', lw=4)
    ax.set_xlim(left=1, right=366)
    ax.set_ylabel(r"$Q_{downstream}-\sum{Q_{upstream}}$"" "r"$(m^3/s)$", fontsize=fontsize_labels);
    ax.set_xlabel("Day of Year", fontsize=fontsize_labels);
    ax.set_title(f"Upstream: {upstream_site_string} | Downstream: {downstream_site}",
                 fontsize=fontsize_title, y=1.04);
    ax.tick_params(axis='both', labelsize=fontsize_tick)

    return ax


def norm_change_annual_flow(sites: list, before_bc: pd.DataFrame, after_bc: pd.DataFrame, colors = list,
                            fontsize_title=60, fontsize_labels=40, fontsize_tick= 30):
    """Normalized change in annual flow volume.

    Plots a series of subplots containing bar charts that depict
    the differnece in normalized annual flow volume due to bias correction.

    Parameters
    ----------
    sites : list
        String names of all the sites to be plotted, matching sites contained in
        the DataFrames `before_bc` and `after_bc`.
    before_bc : pandas.DataFrame
        Contains flows, (not aggregated), before bias correction is applied.
    after_bc : pandas.DataFrame
        Contains flows, (not aggregated), after bias correction is applied.
    colors : list
        Ccolors to be used for each site's subplot in the same order
        as sites, (does not have to be unique).
    fontsize_title : int, optional
        Font size of the title. Defaults as 60.
    fontsize_labels : int, optional
        Font size of the labels. Defaults as 40.
    fontsize_tick : int, optional
        Font size of the ticks. Defaults as 30.
    """

    mpl.rcParams['figure.figsize']=(30,20)

    WY_grouper = calc_water_year(before_bc)
    after_bc_annual = after_bc.groupby(WY_grouper).sum()
    before_bc_annual = before_bc.groupby(WY_grouper).sum()

    after_bc_annual, before_bc_annual = dst.normalize_pair(data=after_bc_annual, norming_data=before_bc_annual)

    diff_annual = after_bc_annual - before_bc_annual

    max_mag = np.abs(np.max(diff_annual.max()))
    min_mag = np.abs(np.min(diff_annual.min()))
    extreme_mag = np.max([max_mag, min_mag])

    n_rows, n_cols = determine_row_col(len(sites))

    fig, axs = plt.subplots(n_rows, n_cols)

    plt.suptitle("Change in Annual Flow Volume from Bias Correction",
                     fontsize=fontsize_title, y=1.05)

    axs_list = axs.ravel().tolist()

    i=0
    for site in sites:
        ax = axs_list[i]
        ax.bar(diff_annual.index, diff_annual[site].values, color=color_list[i])
        ax.set_title(site, fontsize=fontsize_tick)
        ax.set_ylim(top = extreme_mag, bottom=-extreme_mag)
        ax.tick_params(axis='both', labelsize=fontsize_tick)
        i += 1

    while i < len(axs_list):
        axs_list[i].axis('off')
        i += 1

    fig.text(0.5, -0.04, 'Hydrologic Year',
                 ha='center', va = 'bottom', fontsize=fontsize_labels);
    fig.text(-0.04, 0.5, "Normalized Change in Annual Flow Volume",
                 va='center', rotation = 'vertical', fontsize=fontsize_labels);

    plt.tight_layout()

def pbias_compare_hist(sites: list, raw_flow: pd.DataFrame, ref_flow: pd.DataFrame,
                      bc_flow: pd.DataFrame, grouper=pd.Grouper(freq='Y'), total_bins=None,
                      title_freq='Yearly', fontsize_title=90, fontsize_subplot_title=60,
                    fontsize_tick=40,fontsize_labels=84, x_extreme=150):
    """Histograms comparing percent bias before/after bias correction.

    Creates a number of histogram subplots by each given sites that
    plot percent bias both before and after bias correction.

    Parameters
    ----------
    sites : list
        Sites corresponding to the columns of the flow DataFrames, `raw_flow`,
        `ref_flow`, and `bc_flow`, to be plotted.
    raw_flow : pandas.DataFrame
        Flows before bias correction.
    ref_flow : pandas.DataFrame
        Reference flows for comparison as true values.
    bc_flow : pandas.DataFrame
        Flows after bias correction.
    grouper : pandas.TimeGrouper
        How flows should be grouped for bias correction, defaults as yearly.
    total_bins : int, optional
        Number of bins to use in the histogram plots. if none specified,
        defaults to the floored square root of the number of pbias difference
        values.
    title_freq : str, optional
        An adjective description of the frequency with which the flows are
        grouped corresponding to `grouper`. Defaults as 'Yearly'.
    fontsize_title : int, optional
        Font size of the title, defaulting as 90.
    fontsize_subplot_title : int, optional
        Font size of the subplot title, defaulting as 60.
    fontsize_ticler : int, optional
        Font size of the ticks, defaulting as 40.
    fontsize_labels : int, optional
        Font size of the labels, defaulting as 84.
    x_extreme : float, optional
        Greatest magnitude on the horizontal axis to specify the range,
        defaulting as 150, which results in a range of (-150, 150). This is
        useful if desiring to zoom in closer to the origin and exclued
        outlying percent biases.
    """

    mpl.rcParams['figure.figsize']=(60,40)

    n_rows,n_cols = determine_row_col(len(sites))

    bc_m_pbias = dst.pbias_by_index(
        observe=ref_flow.groupby(grouper).sum(),
        predict=bc_flow.groupby(grouper).sum())
    raw_m_pbias = dst.pbias_by_index(
        observe=ref_flow.groupby(grouper).sum(),
        predict=raw_flow.groupby(grouper).sum())
    if type(total_bins)==type(None):
        total_bins=int(np.sqrt(len(bc_m_pbias.index)))

    fig, axs = plt.subplots(n_rows,n_cols)
    axs_list = axs.ravel().tolist()

    plt.suptitle(f"Change in Percent Bias from Bias Correction on a {title_freq} Basis",
                 fontsize=fontsize_title,y=1.05)
    i=0
    for site in sites:
        ax = axs_list[i]
        before_pbias=raw_m_pbias[site]
        after_pbias=bc_m_pbias[site]
        before_extreme = np.max(np.abs(before_pbias))
        after_extreme = np.max(np.abs(after_pbias))
        extreme=np.max([before_extreme,after_extreme])
        bin_step = (2*extreme/total_bins)
        bin_range = np.arange(-extreme,extreme+bin_step,bin_step)

        before_pbias.plot.hist(ax=ax,bins=bin_range,color='red',edgecolor='black',alpha=0.5)
        after_pbias.plot.hist(ax=ax,bins=bin_range,color='blue',edgecolor='black',alpha=0.5)
        ax.vlines(0,0,ax.get_ylim()[1], color='k',linestyle='--',lw=4)
        x_extreme = np.abs(x_extreme)
        ax.set_xlim(left=-x_extreme,right=x_extreme)

        ax.set_title(site,fontsize=fontsize_subplot_title)
        ax.set_ylabel("")
        ax.tick_params(axis='both',labelsize=fontsize_tick)
        i+=1

    while i < len(axs_list):
        axs_list[i].axis('off')
        i+=1

    fig.text(0.5, -0.04, r'$PBias_{monthly} \quad (\%)$',
             ha='center', va = 'bottom', fontsize=fontsize_labels);
    fig.text(-0.02, 0.5, 'Frequencey',
             va='center', rotation = 'vertical', fontsize=fontsize_labels);
    plt.legend(handles=custom_legend(['Before BC','After BC', 'Overlap'],['red','blue','purple']),
               loc='lower right',fontsize=fontsize_labels)
    plt.tight_layout()

def compare_PDF(flow_dataset:xr.Dataset, gauge_sites = list,
                raw_var='raw_flow', ref_var='reference_flow', bc_var='bias_corrected_total_flow',
                raw_name='Mizuroute Raw', ref_name='NRNI Reference', bc_name='BMORPH BC',
                fontsize_title=40, fontsize_labels=30, fontsize_tick= 20):
    """Compare probability distribution functions.

    Plots the PDF's of the raw, reference, and bias corrected flows for each gauge site.

    Parameters
    ----------
    flow_dataset : xarray.Dataset
        Contatains raw, reference, and bias corrected flows.
    gauges_sites : list
        Gauge sites to be plotted as used in the `flow_dataset`.
    raw_var : str, optional
        The string to access the raw flows in `flow_dataset` for flows before
        bias correction. Defaults as 'raw_flow'.
    ref_var : str, optional
        The string to access the reference flows in `flow_dataset` for true flows.
        Defaults as 'reference_flow'.
    bc_var : str, optional
        The string to access the bias corrected flows in flow_dataset for flows
        after bias correction. Defaults as 'bias_corrected_total_flow'.
    raw_name : str, optional
        Label for the raw flows before bias correction. Defaults as 'Mizuroute Raw'.
    ref_name : str, optional
        Label for the reference flows. Defaults as 'NRNI Reference'.
    bc_name : str, optional
        Label for the bias corrected flows after bias correction. Defaults as
        'BMORPH BC'.
    fontsize_title : int, optional
        Fontsize of the title. Defaults as 40.
    fontsize_labels : int, optional
        Fontsize of the lables. Defaults as 30.
    fontsize_tick : int, optional
        Fontsize of the ticks. Defaults as 20.
    """

    n_rows, n_cols = determine_row_col(len(gauge_sites))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20), sharex=False, sharey=False)
    axes = axes.flatten()

    fig.suptitle("Probability Distribution Functions", y=1.01,x=0.4, fontsize=fontsize_title)

    for i, site in enumerate(gauge_sites):
        cmp = flow_dataset.sel(outlet=site)
        ax=axes[i]
        sns.kdeplot(np.log10(cmp[raw_var].values), ax=ax, color='grey', legend=False, label=raw_name)
        sns.kdeplot(np.log10(cmp[ref_var].values), ax=ax, color='black', legend=False, label=ref_name)
        sns.kdeplot(np.log10(cmp[bc_var].values), ax=ax, color='red', legend=False, label=bc_name)
        ax.set_title(site, fontsize=fontsize_labels)
        ax.tick_params(axis='both', labelsize=fontsize_tick)
        # relabel axis in standard base 10
        labels = ax.get_xticks()
        ax.set_xticklabels(labels=["$" + str(10**j) + "$" for j in labels], rotation=30)

    axes[-1].axis('off')
    axes[i].legend(bbox_to_anchor=(1.1, 0.8), fontsize=fontsize_tick)

    fig.text(0.4, 0.04, r'Q [$m^3/s$]', ha='center', fontsize=fontsize_labels)
    fig.text(-0.04, 0.5, r'Density', va='center', rotation='vertical', fontsize=fontsize_labels)

    plt.subplots_adjust(wspace=0.3, hspace= 0.45, left = 0.05, right = 0.8, top = 0.95)

def compare_CDF(flow_dataset: xr.Dataset, plot_sites: list,
                      raw_var: str, raw_name: str,
                      ref_var: str, ref_name: str,
                      bc_vars: list, bc_names: list,
                      plot_colors: list, logit_scale = True,
                      logarithm_base = '10', units = r'Q [$m^3/s$]',
                      markers = ['o', 'x', '*', '*'],
                      figsize = (20,20), sharex = False, sharey = True,
                      fontsize_title = 40, fontsize_labels = 30, fontsize_tick = 20,
                      markersize = 1, alpha = 0.3):
    """Compare probability distribution functions on a logit scale.

    Plots the CDF's of the raw, reference, and bias corrected flows.

    Parameters
    ----------
    flow_dataset : xarray.Dataset
        Contatains raw, reference, and bias corrected flows.
    gauges_sites : list
        Gauge sites to be plotted contained in `flow_dataset`.
    raw_var : str
        Accesses the raw flows in `flow_dataset` for flows before bias correction.
    raw_name : str
        Label for the raw flows in the legend, corresponding to `raw_var`.
    ref_var : str
        Accesses the reference flows in `flow_dataset` for true flows.
    ref_name : str
        Label for the reference flows in the legend, corresponding to `ref_var`.
    bc_vars : list
        Accesses the bias corrected flows in `flow_dataset`, where each element
        accesses its own bias corrected flows. Can be a size of 1.
    bc_names : list
        Label for the bias corrected flows in the legend for each entry in `bc_var`,
        assumed to be in the same order. Can be a size of 1.
    logit_scale : boolean, optional
        Whether to plot the vertical scale on a logit axis (True) or not (False).
        Defaults as True.
    logarithm_base : str, optional
        The logarthimic base to use for the horizontal scale. Only the following
        are currently supported:
            '10' to use a log10 horizontal scale (default)
            'e' to use a natural log horizontal scale
    units : str, optional
        The horizontal axis's label for units, defaults as r'Q [$m^3/s$]'.
    plot_colors : list, optional
        Colors to be plotted for the flows corresponding to `raw_var`,
        `ref_var`, and `bc_var`, defaulting as ['grey', 'black', 'blue', 'red'],
        assuming there are two entries in `bc_var`.
    markers : list, optional
        Markers to be plotted for the flows corresponding to `raw_var`,
        `ref_var`, and `bc_var`, defaulting as ['o', 'x', '*', '*'],
        assuming there are two entries in `bc_var`.
    figsize : tuple, optional
        Figure size following matplotlib notation, defaults as (20, 20).
    sharex : boolean or str, optional
        Whether horizontal axis should be shared amongst subplots,
        defaulting as False.
    sharey : boolean or str, optional
        Whether vertical axis should be shared amongst subplots,
        defaulting as True.
    fontsize_title : int, optional
        Font size of the title, defaults as 40.
    fontsize_labels : int, optional
        Font size of the labels, defaults as 30.
    fontsize_tick : int, optional
        Font size of the ticks, defaults as 20.
    markersize : float, optional
        Size of the markers plotted, defaults as 1.
    alpha : float, optional
        Transparancy of the markers plotted, defaults as 0.3 to help
        see where markers clump together.

    Returns
    -------
    matplotlib.figure, matplotlib.axes
    """

    if len(bc_vars) == 0:
        raise Exception("Please enter a non-zero number strings in bc_vars to be used")
    if len(bc_vars) != len(bc_names):
        raise Exception("Please have the same number of entries in bc_names as bc_names")
    if len(plot_colors) < len(bc_vars):
        raise Exception(f"Please enter at least {len(bc_vars)} colors in plot_colors")

    if logarithm_base == '10':
        log_func = log10_1p
    elif logarithm_base == 'e':
        log_func = np.log1p
    else:
        raise Exception("Please enter logarithm_base as '10' or 'e'")

    n_rows, n_cols = determine_row_col(len(plot_sites))

    fig, axes = plt.subplots(n_rows, n_cols, figsize = figsize, sharex = sharex, sharey = sharey)
    axes = axes.flatten()

    fig.suptitle("Cumulative Distribution Functions", y = 1.01, x = 0.4, fontsize = fontsize_title)

    for i, site in enumerate(plot_sites):
        ax = axes[i]
        cmp = flow_dataset.sel(seg = site)

        raw = ECDF(log_func(cmp[raw_var].values))
        ref = ECDF(log_func(cmp[ref_var].values))

        cors = list()
        for bc_var in bc_vars:
            cors.append(ECDF(log_func(cmp[bc_var].values)))

        linewidth = markersize / 2

        ax.plot(raw.x, raw.y, color = plot_colors[0], label = raw_name, lw = linewidth,
                linestyle = '--', marker = 'o', markersize = markersize, alpha = alpha)
        ax.plot(ref.x, ref.y, color = plot_colors[1], label = ref_name, lw = linewidth,
                linestyle = '--', marker = 'x', markersize = markersize, alpha = alpha)

        for j, cor in enumerate(cors):
            ax.plot(cor.x, cor.y, color = plot_colors[2+j], label = bc_names[j], lw = linewidth,
                    linestyle = '--', marker = '*', markersize = markersize, alpha = alpha)

        ax.set_title(site, fontsize = fontsize_labels)
        ax.tick_params(axis = 'both', labelsize = fontsize_tick)

        # relabel axis to account for log_func
        xlabels = ax.get_xticks()
        if logarithm_base == '10':
            ax.set_xticklabels(labels = ["$" + "{:0.0f}".format(10**k-1) + "$" for k in xlabels],
                               rotation = 30)
        elif logarithm_base == 'e':
            ax.set_xticklabels(labels = ["$" + "{:0.2f}".format(exp(k)-1) + "$" for k in xlabels],
                               rotation=30)

        if logit_scale:
            ax.set_yscale('logit')
            ax.minorticks_off()
            ax.yaxis.set_major_formatter(mpl.ticker.LogitFormatter())
            ylabels = ax.get_yticks()
            new_ylabels = list()
            for k, ylabel in enumerate(ylabels):
                if k % 3 == 0:
                    new_ylabels.append(ylabel)

            ax.set_yticks(ticks = new_ylabels)

    axes[-1].axis('off')
    axes[i].legend(bbox_to_anchor = (1.1, 0.8), fontsize = fontsize_tick)

    fig.text(0.4, 0.04, units, ha = 'center', fontsize = fontsize_labels)
    fig.text(-0.04, 0.5, r'Non-exceedence probability', va = 'center',
             rotation = 'vertical', fontsize = fontsize_labels)
    plt.subplots_adjust(hspace = 0.35, left = 0.05, right = 0.8, top = 0.95)

    return fig, axes

def spearman_diff_boxplots_annual(raw_flows: pd.DataFrame, bc_flows: pd.DataFrame, site_pairings,
                                  fontsize_title=40, fontsize_tick=30, fontsize_labels=40,
                                  subtitle = None, median_plot_color = 'red'):

    """Annual difference in spearman rank as boxplots.

    Creates box plots for each stide pairing determing the difference in spearman
    rank for each year between the raw and bias corrected data.

    Parameters
    ----------
    raw_flows : pandas.DataFrame
        Raw flows before bias correction with sites in the columns and time in the index.
    bc_flows : pandas.DataFrame
        Bias corrected flows with sites in the columns and time in the index.
    site_pairings : List[List[str]]
        List of list of string site pairs e.g. [['downstream_name','upstream_name'],...].
        This is used to organize which sites should be paired together for computing the
        spearman rank difference.
    fontsize_title : int, optional
        Font size of the title, defaults as 40.
    fontsize_tick : int, optional
        Font size of the ticks, defaults as 30.
    fontsize_labels : int, optional
        Font size of the labels, defaults as 40.
    subtitle : str, optional
        Subtitle to include after "Annual Chnage in Speraman Rank: ". If no subtitle
        is is specified, none is included and only the title is plotted.
    median_plot_color : str
        Color to plot the boxplot's median as, defaults as 'red'.
    """

    if np.where(raw_flows.index != bc_flows.index)[0].size != 0:
        raise Exception('Please ensure raw_flows and bc_flows have the same index')

    WY_grouper = calc_water_year(raw_flows)

    annual_spearman_difference = pd.DataFrame(index=np.arange(WY_grouper[0], WY_grouper[-1],1),
                                             columns=[str(pairing) for pairing in site_pairings])

    for WY in annual_spearman_difference.index:
        raw_flow_WY = raw_flows[f"{WY}-10-01":f"{WY+1}-09-30"]
        bc_flow_WY = bc_flows[f"{WY}-10-01":f"{WY+1}-09-30"]

        for site_pairing in site_pairings:
            downstream = site_pairing[0]
            upstream = site_pairing[1]

            downstream_raw = raw_flow_WY[downstream]
            upstream_raw = raw_flow_WY[upstream]
            downstream_bc = bc_flow_WY[downstream]
            upstream_bc = bc_flow_WY[upstream]

            raw_spearman, raw_pvalue = scipy.stats.spearmanr(a=downstream_raw.values,b=upstream_raw.values)
            bc_spearman, bc_pvalue = scipy.stats.spearmanr(a=downstream_bc.values,b=upstream_bc.values)
            annual_spearman_difference.loc[WY][str(site_pairing)] = raw_spearman-bc_spearman

    fig, ax = plt.subplots(figsize=(20,20))

    if isinstance(subtitle, type(None)):
        plt.suptitle('Annual Change in Spearman Rank', fontsize=fontsize_title, y=1.05)
    else:
        assert isinstance(subtitle, str)
        plt.suptitle(f'Annual Change in Spearman Rank: {subtitle}', fontsize = fontsize_title, y = 1.05)

    max_vert = np.max(annual_spearman_difference.max().values)*1.1
    min_vert = np.min(annual_spearman_difference.min().values)*1.1
    ax.axhline(y=0, color='black', linestyle='--')
    ax.boxplot([annual_spearman_difference[site_pairing].values for
                site_pairing in annual_spearman_difference.columns],
               labels=[f'{site_pairing[0]},\n{site_pairing[1]}' for site_pairing in site_pairings],
               medianprops={'color':median_plot_color, 'lw':4}, boxprops={'lw':4}, whiskerprops={'lw':4},
               capprops={'lw':4})

    ax.set_ylim(top=max_vert, bottom=min_vert)
    ax.tick_params(axis='both', labelsize=fontsize_tick)

    fig.text(0.5, -0.04, "Gauge Site Pairs: Downstream, Upstream",
                 ha='center', va = 'bottom', fontsize=fontsize_labels);

    fig.text(-0.04, 0.5, r'$r_s(Q_{raw}^{up}, Q_{raw}^{down}) - r_s(Q_{bc}^{up}, Q_{bc}^{down})$',
             va='center', rotation = 'vertical', fontsize=fontsize_labels);

    plt.tight_layout()

def kl_divergence_annual_compare(flow_dataset: xr.Dataset, sites: list,
                                 raw_var: str, raw_name: str,
                                 ref_var: str, ref_name: str,
                                 bc_vars: list, bc_names: list,
                                 plot_colors: list, title = "Annual KL Diveregence Before/After Bias Correction",
                                 fontsize_title = 40, fontsize_tick = 30, fontsize_labels = 40,
                                 fontsize_legend = 30,
                                 showfliers = False, sharex = True, sharey = 'row', TINY_VAL = 1e-6,
                                 figsize = (30,20), show_y_grid = True):
    """Kullback-Liebler Divergence compared before and after bias correction as boxplots.

    Plots the KL divergence for each year per site as KL(P_{ref} || P_{raw}) and
    KL( P_{ref} || P_{bc}).

    Parameters
    ----------
    flow_dataset : xarray.Dataset
        Contains raw (uncorrected), reference (true), and bias corrected flows.
    sites : list
        Contains all the sites to be plotted as included in `flow_dataset`, (note that if
        the number of sites to be plotted is square or rectangular, the last site will not
        be plotted to save room for the legend).
    raw_var : str
        Accesses the raw flows in `flow_dataset`.
    raw_name : str
        Label for the raw flows in the legend and horizontal labels corresponding to `raw_var`.
    ref_var : str
        Accesses the reference flows in `flow_dataset`.
    ref_name : str
        Label the reference flows in the legend and horizontal labels corresponding to `ref_var`.
    bc_vars : list
        String(s) to access the bias corrected flows in `flow_dataset`.
    bc_names : list
        Label(s) for the bias corrected flows in the legend and horizontal labels, corresponding
        to each element in `bc_vars`.
    plot_colors : list
        Colors to be plotted for the raw and the bias corrected flows, respectively.
    title : str, optional
        Title to be plotted, defaults as "Annual KL Diveregence Before/After Bias Correction".
    fontsize_title : int, optional
        Font size of the title, defaults as 40.
    fontsize_tick : int, optional
        Font size of the ticks, defaults as 30.
    fontsize_labels : int, optional
        Font size of the labels, defaults as 40.
    fontsize_legend : int, optional
        Font size of the legend text, defaults as 30.
    showfliers : boolean, optional
        Whether to include fliers in the boxplots, defaults as False.
    sharex : boolean or str, optional
        Whether the horizontal axis is shared, defaults as True.
    sharey : boolean or str, optional
        Whether the vertical axis is shared, defaults as 'row' to share the vertical axis
        in the same row.
    TINY_VAL : float, optional
        Used to ensure there are no zero values in the data because zero values cause unsual
        behavior in calculating the KL Divergence. Defaults as 1E-6.
    figsize : tuple, optional
        Figure size following maptlotlib connventions, defaults as (30,20).
    show_y_grid : boolean, optional
        Whether to plot y grid lines, defaults as True.

    Returns
    -------
    maptlotlib.figure, matplotlib.axes
    """

    raw_flows = flow_dataset[raw_var].to_pandas()
    ref_flows = flow_dataset[ref_var].to_pandas()
    bc_flows = list()
    for bc_var in bc_vars:
        bc_flows.append(flow_dataset[bc_var].to_pandas())

    WY_grouper = calc_water_year(raw_flows)
    WY_array = np.arange(WY_grouper[0], WY_grouper[-1], 1)

    n_rows, n_cols = determine_row_col(len(sites))
    fig, axs = plt.subplots(n_rows, n_cols, figsize = figsize, sharex = sharex, sharey = sharey)
    axs_list = axs.ravel().tolist()

    kldiv_refraw_annual = pd.DataFrame(index = WY_array, columns = sites)
    kldiv_refbc_annuals = list()
    for bc_var in bc_vars:
        kldiv_refbc_annuals.append(pd.DataFrame(index = WY_array, columns = sites))

    plt.suptitle(title, fontsize = fontsize_title, y = 1.05)

    for WY in WY_array:
        raw_flow_WY = raw_flows[f"{WY}-10-01":f"{WY+1}-09-30"]
        ref_flow_WY = ref_flows[f"{WY}-10-01":f"{WY+1}-09-30"]
        bc_flow_WYs = list()
        for bc_flow in bc_flows:
            bc_flow_WYs.append(bc_flow[f"{WY}-10-01":f"{WY+1}-09-30"])
        total_bins = int(np.sqrt(len(raw_flow_WY.index)))

        for site in sites:
            ref_WY_site_vals = ref_flow_WY[site].values
            raw_WY_site_vals = raw_flow_WY[site].values
            bc_WY_site_vals = list()
            for bc_flow_WY in bc_flow_WYs:
                bc_WY_site_vals.append(bc_flow_WY[site].values)

            ref_WY_site_pdf, ref_WY_site_edges = np.histogram(ref_WY_site_vals, bins = total_bins,
                                                              density = True)
            raw_WY_site_pdf = np.histogram(raw_WY_site_vals, bins = ref_WY_site_edges, density = True)[0]
            bc_WY_site_pdfs = list()
            for bc_WY_site_val in bc_WY_site_vals:
                bc_WY_site_pdf = np.histogram(bc_WY_site_val, bins = ref_WY_site_edges, density = True)[0]
                bc_WY_site_pdf[bc_WY_site_pdf == 0] = TINY_VAL
                bc_WY_site_pdfs.append(bc_WY_site_pdf)

            ref_WY_site_pdf[ref_WY_site_pdf == 0] = TINY_VAL
            raw_WY_site_pdf[raw_WY_site_pdf == 0] = TINY_VAL

            kldiv_refraw_annual.loc[WY][site] = scipy.stats.entropy(pk = raw_WY_site_pdf, qk = ref_WY_site_pdf)
            for i, (kldiv_refbc_annual, bc_WY_site_pdf) in enumerate(zip(kldiv_refbc_annuals, bc_WY_site_pdfs)):
                kldiv_refbc_annual.loc[WY][site] = scipy.stats.entropy(pk = bc_WY_site_pdf, qk = ref_WY_site_pdf)
                kldiv_refbc_annuals[i] = kldiv_refbc_annual

    plot_labels = [raw_name]
    plot_labels.extend(bc_names)

    for i, site in enumerate(sites):
        ax=axs_list[i]
        plot_vals = [kldiv_refraw_annual[site].values]
        plot_vals.extend([kldiv_refbc_annual[site].values for kldiv_refbc_annual in kldiv_refbc_annuals])
        box_dict = ax.boxplot(plot_vals, patch_artist = True, showfliers = showfliers, widths = 0.8, notch = True)
        for item in ['boxes', 'fliers', 'medians', 'means']:
            for sub_item, color in zip(box_dict[item], plot_colors):
                plt.setp(sub_item, color = color)
        ax.set_title(site, fontsize = fontsize_labels)
        ax.set_xticks(np.arange(1, len(plot_labels)+1))
        ax.set_xticklabels(plot_labels, fontsize = fontsize_tick, rotation = 45)
        ax.tick_params(axis = 'both', labelsize = fontsize_tick)
        if show_y_grid:
            ax.grid(which = 'major', axis = 'y', alpha = 0.5)

    # gets rid of any spare axes
    i += 1
    while i < len(axs_list):
        axs_list[i].axis('off')
        i += 1
    # ensures last axes is off to make room for the legend
    axs_list[-1].axis('off')

    fig.text(-0.04, 0.5, "Annual KL Divergence",
             va = 'center', rotation = 'vertical', fontsize = fontsize_labels);

    fig.text(0.5, -0.04, r'$KL(P_{' + f'{ref_name}' + r'} || P_{scenario})$',
             va = 'bottom', ha = 'center', fontsize = fontsize_labels);

    axs_list[-1].legend(handles=custom_legend(names=plot_labels, colors = plot_colors),
                        fontsize = fontsize_legend, loc = 'center')

    plt.tight_layout()

    return fig, axs

def spearman_diff_boxplots_annual_compare(flow_dataset: xr.Dataset, site_pairings,
                                  raw_var: str, bc_vars: list, bc_names: list,
                                  plot_colors: list, showfliers =  True,
                                  fontsize_title=40, fontsize_tick=25, fontsize_labels=30,
                                  figsize = (20,20), sharey = 'row'):

    """Annual difference in spearman rank as boxplots.

    Creates box plots for each site pairing determining the difference in spearman
    rank for each year between the raw and the bias corrected data.

    Parameters
    ----------
    flow_dataset : xarray.Dataset
        Contains raw (uncorrected), reference (true), and bias corrected flows.
    site_pairings : List[List[str]]
        List of list of string site pairs e.g. [['downstream_name','upstream_name'],...].
        This is used to organize which sites should be paired together for computing the
        spearman rank difference.
    raw_var : str
        Accesses the raw (uncorrected) flows in `flow_dataset`.
    bc_vars : list
        Strings to access the bias corrected flows in `flow_dataset`.
    bc_names : list
        Labels for the bias corrected flows from `flow_dataset`, corresponding to
        each element in `bc_vars` in the same order.
    plot_colors : list
        Colors that are in the same order as the `bc_vars` and `bc_names`
        to be used in plotting.
    showfliers : boolean, optional
        Whether to show fliers on the boxplots, defaults as True.
    fontsize_title : int, optional
        Font size of the title, defaults as 40.
    fontsize_tick : int, optional
        Font size of the ticks, defaults as 25.
    fontsize_lables : int, optional
        Font size of the labels, defaults as 30.
    figsize : tuple
        Figure size following matplotlib connventions, defaults as (20, 20).
    sharey : boolean or str, optional
        Whether or how the vertical axis are to be shared, defaults as 'row' to
        have vertical axis in the same row shared.

    Returns
    -------
    matplotlib.figure, matplotlib.axes
    """

    if len(bc_vars) == 0:
        raise Exception("Please enter a non-zero number strings in bc_vars to be used")
    if len(bc_vars) != len(bc_names):
        raise Exception("Please have the same number of entries in bc_names as bc_names")
    if len(plot_colors) < len(bc_vars):
        raise Exception(f"Please enter at least {len(bc_vars)} colors in plot_colors")

    raw_flows = flow_dataset[raw_var].to_pandas()
    bc_flows = list()
    for bc_var in bc_vars:
        bc_flows.append(flow_dataset[bc_var].to_pandas())

    WY_grouper = calc_water_year(raw_flows)
    WY_index = np.arange(WY_grouper[0], WY_grouper[-1], 1)

    annual_spearman_differences = list()
    for bc_var in bc_vars:
        annual_spearman_differences.append(pd.DataFrame(index=WY_index,
                                                        columns=[str(pairing) for pairing in site_pairings]))

    for WY in WY_index:
        raw_flow_WY = raw_flows[f"{WY}-10-01":f"{WY+1}-09-30"]
        bc_flow_WY_list= list()
        for bc_flow in bc_flows:
            bc_flow_WY_list.append(bc_flow[f"{WY}-10-01":f"{WY+1}-09-30"])

        for site_pairing in site_pairings:
            downstream = site_pairing[0]
            upstream = site_pairing[1]

            downstream_raw = raw_flow_WY[downstream]
            upstream_raw = raw_flow_WY[upstream]
            downstream_bcs = list()
            upstream_bcs = list()
            for bc_flow_WY in bc_flow_WY_list:
                downstream_bcs.append(bc_flow_WY[downstream])
                upstream_bcs.append(bc_flow_WY[upstream])

            raw_spearman, raw_pvalue = scipy.stats.spearmanr(a=downstream_raw.values, b=upstream_raw.values)
            for i, (downstream_bc, upstream_bc, annual_spearman_difference) in enumerate(
                zip(downstream_bcs, upstream_bcs, annual_spearman_differences)):
                bc_spearman, bc_pvalue = scipy.stats.spearmanr(a=downstream_bc.values, b=upstream_bc.values)
                annual_spearman_difference.loc[WY][str(site_pairing)] = raw_spearman-bc_spearman
                annual_spearman_differences[i] = annual_spearman_difference

    n_rows, n_cols = determine_row_col(len(site_pairings))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, sharex = True, sharey = sharey)
    axs_list = axs.ravel().tolist()

    #rewrites wrt subplots for site_pairings

    fig.suptitle('Annual Change in Spearman Rank', fontsize=fontsize_title, y=1.05)

    max_vert = np.max([df.values for df in annual_spearman_differences])*1.1
    min_vert = np.min([df.values for df in annual_spearman_differences])
    min_vert  = np.min([min_vert*1.1, min_vert*0.9])

    for i, site_pairing in enumerate(site_pairings):
        ax = axs_list[i]

        ax.axhline(y=0, color='black', linestyle='--')
        ax.set_title(f"{site_pairing[0]}, {site_pairing[1]}", fontsize = fontsize_labels)
        box_dict = ax.boxplot([annual_spearman_difference[str(site_pairing)].values for
                               annual_spearman_difference in annual_spearman_differences],
                   patch_artist = True, showfliers = showfliers, widths = 0.8, notch = True)

        ax.set_ylim(top=max_vert, bottom=min_vert)
        ax.set_xticklabels(bc_names, fontsize = fontsize_tick)
        ax.tick_params(axis='both', labelsize=fontsize_tick)
        for item in ['boxes', 'fliers', 'medians', 'means']:
                for sub_item, color in zip(box_dict[item], plot_colors):
                    plt.setp(sub_item, color = color)

    fig.text(0.5, -0.04, "Bias Correction Scenario",
                 ha='center', va = 'bottom', fontsize=fontsize_labels);
    fig.text(-0.04, 0.5, r'$r_s(Q_{raw}^{up}, Q_{raw}^{down}) - r_s(Q_{bc}^{up}, Q_{bc}^{down})$',
             va='center', rotation = 'vertical', fontsize=fontsize_labels);

    plt.tight_layout()

    return fig, axs

def compare_CDF_all(flow_dataset:xr.Dataset, plot_sites: list,
                      raw_var: str, raw_name: str,
                      ref_var: str, ref_name: str,
                      bc_vars: list, bc_names: list,
                      plot_colors: list, logit_scale = True,
                      logarithm_base = '10', units = r'Q [$m^3/s$]',
                      figsize = (20,20),
                      fontsize_title = 40, fontsize_labels = 40, fontsize_tick = 40,
                      markersize = 1, alpha = 0.3):
    """Compare probability distribution functions as a summary statistic.

    Plots the CDF's of the raw, reference, and bias corrected flows with data
    from all sites in plot_sites combined for a summary statistic.

    Parameters
    ----------
    flow_dataset : xarray.Dataset
        Contains raw (uncorrected), reference (true), and bias corrected flows.
    plot_sites : list
        Gauge sites to be plotted.
    raw_var : str
        Accesses the raw (uncorrected) flows in `flow_dataset`.
    raw_name : str
       Label for the raw flows in the legend, corresponding to `raw_var`.
    ref_var : str
        Accesses the reference (true) flows in `flow_dataset`.
    ref_name : str
        Label for the reference flows in the legend, corresponding to `ref_var`.
    bc_vars : list
        Accesses the bias corrected flows in `flow_dataset`. Can be a size of 1.
    bc_names : list
        Label for the bias corrected flows in the legend, corresponding to `bc_var`.
        Can be a size of 1.
    plot_colors : list, optional
        Colors to be plotted for the flows corresponding to `raw_var`,
        `ref_var`, and `bc_var`, defaulting as ['grey', 'black', 'blue', 'red'],
        assuming there are two entries in `bc_var`.
    logit_scale : True, optional
        Whether to plot the vertical scale on a logit axis (True) or not (False).
        Defaults as True.
    logarithm_base : str, optional
        The logarthimic base to use for the horizontal scale. Only the following
        are currently supported:
            '10' to use a log10 horizontal scale (default)
            'e' to use a natural log horizontal scale
    units : str, optional
        The horizontal axis's label for units, defaults as r'Q [$m^3/s$]'.
    figsize : tuple, optional
        Figure size following matplotlib connventions, defaults as (20, 20).
    fontsize_title : int, optional
        Font size of the title, defaults as 40.
    fontsize_labels : int, optional
        Font size of the labels, defaults as 40.
    fontsize_tick : int, optional
        Font size of the ticks, defaults as 40.
    markersize : float, optional
        Size of the markers plotted, defaults as 1. Linewidth is half of this value.
    alpha : float, optional
        Transparancy of the lines and markers, defaults as 0.3.

    Returns
    -------
    matplotlib.figure, matplotlib.axes
    """

    if len(bc_vars) == 0:
        raise Exception("Please enter a non-zero number strings in bc_vars to be used")
    if len(bc_vars) != len(bc_names):
        raise Exception("Please have the same number of entries in bc_names as bc_names")
    if len(plot_colors) < len(bc_vars):
        raise Exception(f"Please enter at least {len(bc_vars)} colors in plot_colors")

    if logarithm_base == '10':
        log_func = log10_1p
    elif logarithm_base == 'e':
        log_func = np.log1p
    else:
        raise Exception("Please enter logarithm_base as '10' or 'e'")

    fig, ax = plt.subplots(figsize = figsize)

    fig.suptitle("Cumulative Distribution Function", y = 1.01, x = 0.4, fontsize = fontsize_title)

    cmp = flow_dataset.sel(seg = plot_sites)

    raw = ECDF(log_func(cmp[raw_var].values.flatten()))
    ref = ECDF(log_func(cmp[ref_var].values.flatten()))

    cors = list()
    for bc_var in bc_vars:
        cors.append(ECDF(log_func(cmp[bc_var].values.flatten())))

    linewidth = markersize / 2

    ax.plot(raw.x, raw.y, color = plot_colors[0], label = raw_name, lw = linewidth,
            linestyle = '--', marker = 'o', markersize = markersize, alpha = alpha)
    ax.plot(ref.x, ref.y, color = plot_colors[1], label = ref_name, lw = linewidth,
            linestyle = '--', marker = 'x', markersize = markersize, alpha = alpha)

    for j, cor in enumerate(cors):
        ax.plot(cor.x, cor.y, color = plot_colors[2+j], label = bc_names[j], lw = linewidth,
                linestyle = '--', marker = '*', markersize = markersize, alpha = alpha)

    ax.tick_params(axis = 'both', labelsize = fontsize_tick)

    # relabel axis to account for log_func
    xlabels = ax.get_xticks()
    if logarithm_base == '10':
        ax.set_xticklabels(labels = ["$" + "{:0.0f}".format(10**k-1) + "$" for k in xlabels])
    elif logarithm_base == 'e':
        ax.set_xticklabels(labels = ["$" + "{:0.2f}".format(exp(k)-1) + "$" for k in xlabels])

    if logit_scale:
        ax.set_yscale('logit')
        ax.minorticks_off()
        ax.yaxis.set_major_formatter(mpl.ticker.LogitFormatter())
        ylabels = ax.get_yticks()
        new_ylabels = list()
        for k, ylabel in enumerate(ylabels):
            if k % 3 == 0:
                new_ylabels.append(ylabel)

        ax.set_yticks(ticks = new_ylabels)

    plot_labels = [raw_name, ref_name]
    plot_labels.extend(bc_names)
    ax.legend(handles = custom_legend(plot_labels, plot_colors), fontsize = fontsize_tick)

    fig.text(0.4, 0.04, units, ha = 'center', fontsize = fontsize_labels)
    fig.text(-0.04, 0.5, r'Non-exceedence probability', va = 'center',
             rotation = 'vertical', fontsize = fontsize_labels)
    plt.subplots_adjust(hspace = 0.35, left = 0.05, right = 0.8, top = 0.95)

    return fig, axes

def compare_mean_grouped_CPD(flow_dataset: xr.Dataset, plot_sites: list, grouper_func,
                             raw_var: str, raw_name: str,
                             ref_var: str, ref_name: str,
                             bc_vars: list, bc_names: list,
                             plot_colors: list, subset_month = None,
                             units = r'Mean Annual Flow [$m^3/s$]',
                             figsize = (20,20), sharex = False, sharey = False,
                             pp_kws = dict(postype='cunnane'), fontsize_title = 80,
                             fontsize_legend = 68, fontsize_subplot = 60,
                             fontsize_tick = 45, fontsize_labels = 80,
                             linestyles = ['-','-','-'], markers = ['.','.','.'],
                             markersize = 30, alpha = 1, legend_bbox_to_anchor = (1, 1),
                             fig = None, axes = None, start_ax_index = 0, tot_plots = None
                      ):
    """
    Cumulative Probability Distributions
        plots the CPD's of the raw, reference, and bias corrected flows on a probability axis
    ----
    flow_dataset : xarray.Dataset
        Contatains raw, reference, and bias corrected flows.
    plot_sites : list
        A list of sites to be plotted.
    grouper_func
        Function to group a pandas.DataFrame index by to calculate the
        mean of the grouped values.
    raw_var : str
        The string to access the raw flows in flow_dataset.
    raw_name : str
        Label for the raw flows in the legend.
    ref_var : str
        The string to access the reference flows in flow_dataset.
    ref_name : str
        Label for the reference flows in the legend.
    bc_vars : list
        List of strings to access the bias corrected flows in flow_dataset.
    bc_names : list
        List of labels for the bias corrected flows in the legend.
    plot_colors : list
        Contains the colors to be plotted for `raw_var`, `ref_var`, and
        `bc_vars`, respectively.
    subset_month: int, optional
        The integer date of a month to subset out for plotting,
        (ex: if you want to subset out January, enter 1). Defaults as None
        to avoid subsetting and use all the data in the year.
    units : str, optional
        Vertical axis's label for units, defaults as r'Mean Annual Flow [$m^3/s$]'.
    pp_kws : dict, optional
        Plotting position computation as specified by
        https://matplotlib.org/mpl-probscale/tutorial/closer_look_at_plot_pos.html.
        Defaults as dict(postype='cunnane') for cunnane plotting positions.
    fontsize_title : int, optional
        Font size for the plot title, defaults as 80.
    fontsize_legend : int, optional
        Font size for the plot legend, defaults as 68.
    fontsize_subplot : int, optional
        Font size for the plot subplot text, default as 60.
    fontsize_tick : int, optional
        Font size for the plot ticks, defaults as 45.
    fontsize_labels : int, optional
        Font size for the horizontal and vertical axis labels, defaults as 80.
    linestyles : list, optional
        Linestyles for ploting `raw_var`, `ref_var`, and `bc_vars`, respectively.
        Defaults as ['-','-','-'], expecting one of each.
    markers : list, optional
        Markers for ploting `raw_var`, `ref_var`, and `bc_vars`, respectively.
        Defaults as ['.','.','.'], expecting one of each.
    markersize : int, optional
        Size of the markers for plotting, defaults as 30.
    alpha : float, optional
        Alpha transparency value for plotting, where 1 is opaque and 0 is transparent.
    legend_bbox_to_anchor : tuple, optional
        Box that is used to position the legend to the final axes. Defaults as (1,1).
        Modify this is the legend does not plot where you desire it to be.
    fig : matplotlib.figure, optional
        matplotlib figure object to plot on, defaulting as None and creating a new object
        unless otherwise specified.
    axes : matplotlib.axes, optional
        Array-like of matplotlib axes objet to plot multiple plots on, defaulting as None and creating
        a new object unless otherwise specified.
    start_ax_index : int, optional
        If the plots should not be plotted starting at the first ax in axes, specifiy the index
        that plotting should begin on. Defaults as None, assuming plotting should begin from
        the first ax.
    tot_plots : int, optional
        If more plotting is to be done than with the total data to be provided, describe how many
        total plots there should be. Defalts as None, assuming plotting should begin form the
        first ax.
    """

    if len(bc_vars) == 0:
        raise Exception("Please enter a non-zero number strings in bc_vars to be used")
    if len(bc_vars) != len(bc_names):
        raise Exception("Please have the same number of entries in bc_names as bc_names")
    if len(plot_colors) < 2 + len(bc_vars):
        raise Exception(f"Please enter at least {2+len(bc_vars)} colors in plot_colors")
    if len(linestyles) < 2 + len(bc_vars):
        raise Exception(f"Please enter at least {2+len(bc_vars)} styles in linestyles")
    if len(markers) < 2 + len(bc_vars):
        raise Exception(f"Please enter at least {2+len(bc_vars)} markers in markers")

    if not tot_plots:
        tot_plots = len(plot_sites)
    n_rows, n_cols = determine_row_col(tot_plots)
    if fig is None or axes is None:
        fig, axes = plt.subplots(n_rows, n_cols, figsize = figsize,
                                 sharex = sharex, sharey = sharey)
    axes = axes.flatten()

    time = flow_dataset['time'].values
    raw_flow_df = pd.DataFrame(data = flow_dataset[raw_var].sel(seg=plot_sites).values,
                               index=time, columns = plot_sites)
    ref_flow_df = pd.DataFrame(data = flow_dataset[ref_var].sel(seg=plot_sites).values,
                               index=time, columns = plot_sites)
    bc_flow_dfs = list()
    for bc_var in bc_vars:
        bc_flow_df = pd.DataFrame(data = flow_dataset[bc_var].sel(seg=plot_sites).values,
                                  index = time, columns = plot_sites)
        bc_flow_dfs.append(bc_flow_df)

    if not isinstance(subset_month, type(None)):
        raw_flow_df = raw_flow_df[raw_flow_df.index.month == subset_month]
        ref_flow_df = ref_flow_df[ref_flow_df.index.month == subset_month]
        for i, bc_flow_df in enumerate(bc_flow_dfs):
            bc_flow_dfs[i] = bc_flow_df[bc_flow_df.index.month == subset_month]

    WY_grouper = grouper_func(raw_flow_df)
    raw_flow_annual = raw_flow_df.groupby(WY_grouper).mean()
    ref_flow_annual = ref_flow_df.groupby(WY_grouper).mean()
    bc_flow_annuals = list()
    for bc_flow_df in bc_flow_dfs:
        bc_flow_annual = bc_flow_df.groupby(WY_grouper).mean()
        bc_flow_annuals.append(bc_flow_annual)

    for i, site in enumerate(plot_sites):
        ax = axes[i+start_ax_index]

        raw = raw_flow_annual[site].values
        ref = ref_flow_annual[site].values

        y_max_raw = scipy.stats.scoreatpercentile(raw, 99)
        y_max_ref = scipy.stats.scoreatpercentile(ref, 99)
        y_min_raw = scipy.stats.scoreatpercentile(raw, 1)
        y_min_ref = scipy.stats.scoreatpercentile(ref, 1)
        y_max = np.max([y_max_raw, y_max_ref])*1.1
        y_min = np.min([y_min_raw, y_min_ref])*0.9

        cors = list()
        for bc_flow_annual in bc_flow_annuals:
            bc = bc_flow_annual[site].values
            cors.append(bc)
            y_max_bc = scipy.stats.scoreatpercentile(bc, 99)
            y_min_bc = scipy.stats.scoreatpercentile(bc, 1)
            if y_max_bc > y_max:
                y_max = y_max_bc
            if y_min_bc < y_min:
                y_min = y_min_bc

        common_opts = dict(
            plottype='prob',
            probax='x'
        )

        probscale.probplot(raw, ax=ax, pp_kws=pp_kws,
                           scatter_kws=dict(linestyle=linestyles[0], marker=markers[0], markersize = markersize, alpha=alpha,
                                            color = plot_colors[0], label=raw_name), **common_opts)

        probscale.probplot(ref, ax=ax, pp_kws=pp_kws,
                           scatter_kws=dict(linestyle=linestyles[1], marker=markers[1], markersize = markersize*1.5, alpha=alpha,
                                            color = plot_colors[1], label=ref_name), **common_opts)
        for j, cor in enumerate(cors):
            probscale.probplot(cor, ax=ax, pp_kws=pp_kws,
                               scatter_kws=dict(linestyle=linestyles[2+j], marker=markers[2+j], markersize = markersize, alpha=alpha,
                                                color = plot_colors[2+j], label=bc_names[j]), **common_opts)

        ax.set_title(site, fontsize = fontsize_subplot)
        ax.tick_params(axis = 'both', labelsize = fontsize_tick)
        ax.set_ylim([y_min, y_max])
        ax.set_xlim(left=1, right=99)
        ax.set_xticks([1, 10, 20, 50, 80, 90, 95, 99])
        plt.setp(ax.get_xticklabels(), Rotation=45)

    axes[-1].axis('off')
    plot_names = [raw_name, ref_name]
    plot_names.extend(bc_names)
    axes[i].legend(handles=custom_legend(names = plot_names, colors=plot_colors),
                   bbox_to_anchor = legend_bbox_to_anchor, fontsize = fontsize_legend)

    fig.text(0.4, 0.04, 'Cumulative Percentile', ha = 'center', fontsize = fontsize_labels)
    fig.text(-0.01, 0.5, units, va = 'center',
             rotation = 'vertical', fontsize = fontsize_labels)

    plt.subplots_adjust(hspace = 0.35, left = 0.05, right = 0.8, top = 0.95)

    return fig, axes

