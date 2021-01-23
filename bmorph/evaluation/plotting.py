import numpy as np
import xarray as xr
import pandas as pd
from typing import List
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy

import networkx as nx
import graphviz as gv
import pygraphviz as pgv
from networkx.drawing.nx_agraph import graphviz_layout

from bmorph.evaluation.constants import colors99p99
from bmorph.evaluation import descriptive_statistics as dst

from statsmodels.distributions.empirical_distribution import ECDF

#*****************************************************************************************
# Plotting Helper Functions:
#      custom_legend
#      calc_water_year
#      find_index_hyear
#      determin_row_col
#      log10_1p
#      scatter_series_axes
#*****************************************************************************************

def custom_legend(names:List, colors=colors99p99):
    """Creates a list of patches to be passed in as `handels` for the plt.legends function.
    
    Parameters
    ----------
    names : list
        Legend names.
    colors : list
        A list of the colors corresponding to `names`.

    Returns
    -------
    handles
        ???
    """
    legend_elements = list()
    for i,name in enumerate(names):
        legend_elements.append(mpl.patches.Patch(facecolor = colors[i], label = name))
    return legend_elements

def calc_water_year(df: pd.DataFrame):
    """Calculates hydrologic year.
    
    Parameters
    ----------
    df: pandas.DataFrame 
        Flow timeseries with a DataTime index.

    Returns
    -------
    ???
        A pandas.DataFrame index grouped by hydrologic year.

    """
    return df.index.year + (df.index.month >= 10).astype(int)

def find_index_hyear(data: pd.DataFrame) -> np.int:
    """ Finds the index of the first hydrologic year.
    
    Parameters
    ----------
    data: pd.DataFrame
        Flow timeseries with a DateTime index.
        
    Returns
    -------
    int
        Index of the first hydrologic year.
    """
    hyear = pd.Timestamp(0)
    i = 0
    while hyear == pd.Timestamp(0):
        date = data.index[i]
        if date.month == 10 and date.day == 1:
            hyear = date
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
#      rmseFracePlot
#*****************************************************************************************

def pbias_sites(observed: pd.DataFrame, predicted: pd.DataFrame):
    """Calculates percent bias on a hydrologic year and site-by-site basis.
    
    Parameters
    ----------
    observed : pd.DataFrame
        Dataframe containing all observations.
    predicted : pd.DataFrame
        Dataframe containing all predictions.
        
    Returns
    -------
    pandas.DataFrame
        Dataframe contain the percent bias computed.
    """
    #finds the start of the first hydraulic year
    i = find_index_hyear(observed)
    hyear_start = observed.index[i]
    #counts the number of hydraylic years
    hyears = 0
    while i < len(observed):
        date = observed.index[i]
        if date.month == 9 and date.day == 30:
            hyears = hyears + 1
        i = i + 1

    pbias_site_df = pd.DataFrame(columns = observed.columns, 
                                 index = pd.Series(range(0, hyears)))
    pbias_current_year = pd.DataFrame(columns = observed.columns)

    for i in range(0, hyears):
        #establish end of hydraulic year
        hyear_end = hyear_start + pd.Timedelta(364.25, unit = 'd')

        #need to truncate datetimes since time indicies do not align
        O = observed.loc[hyear_start : hyear_end]
        P = predicted.loc[hyear_start : hyear_end]
        O.index = O.index.floor('d')
        P.index = P.index.floor('d')

        pbias_current_year = dst.pbias(O, P)

        #puts the computations for the hydraulic year into our bigger dataframe
        pbias_site_df.iloc[i] = pbias_current_year.iloc[0].T

        #set up next hydraulic year
        hyear_start = hyear_start + pd.Timedelta(365.25, unit = 'd')

    return pbias_site_df

def diff_maxflow_sites(observed: pd.DataFrame, predicted: pd.DataFrame):
    """Calculates difference in maximum flows on a hydrologic year and site-by-site basis.
    
    Parameters
    ----------
    observed : pandas.DataFrame
        Dataframe containing all observations.
    predicted: pandas.DataFrame
        Dataframe containing all predictions.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the difference in maximum flows.
    """
    #finds the start of the first hydraulic year
    i = find_index_hyear(observed)
    hyear_start = observed.index[i]

    #counts the number of hydraylic years
    hyears = 0
    while i < len(observed):
        date = observed.index[i]
        if date.month == 9 and date.day == 30:
            hyears = hyears + 1
        i = i + 1

    diff_maxflow_sites_df = pd.DataFrame(columns = observed.columns, 
                                         index = pd.Series(range(0,hyears)))
    diff_maxflow_current_year = pd.DataFrame(columns = observed.columns)

    for i in range(0,hyears):
        #establish end of hydraulic year
        hyear_end = hyear_start + pd.Timedelta(364.25, unit = 'd')

        #need to truncate datetimes since time indicies do not align
        O = observed.loc[hyear_start : hyear_end]
        P = predicted.loc[hyear_start : hyear_end]
        O.index = O.index.floor('d')
        P.index = P.index.floor('d')

        diff_maxflow_current_year = P.max().to_frame().T - O.max().to_frame().T

        #puts the computations for the hydraulic year into our bigger dataframe
        diff_maxflow_sites_df.iloc[i] = diff_maxflow_current_year.iloc[0].T

        #set up next hydraulic year
        hyear_start = hyear_start + pd.Timedelta(365.25, unit = 'd')


    return diff_maxflow_sites_df

def pbias_plotter(observed: pd.DataFrame, names: list, colors: list, *models: pd.DataFrame):
    """Plots box plots of numerous models grouped by site.
    
    Parameters
    ----------
    observed : pd.Dataframe
        Dataframe containing observations.
    names  : list
        List of the model names.
    colors: list
        List of colors to be plotted.
    *models : List[pandas.DataFrame]
        Any number of pandas.DataFrame objects to be evaluated.
        
    Returns
    -------
    fig : matplotlib.figure
    ax : matplotlib.axes
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
    observed : pd.Dataframe
        a dataframe containing observations
    names : list
        List of the model names.
    colors: list
        List of colors to be plotted.
    *models : List[pandas.DataFrame]
        Any number of pandas.DataFrame objects to be evaluated.
        
    Returns
    -------
    fig : matplotlib.figure
    ax : matplotlib.axes
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

    predictions: dictionary containing
        'Prediction Names':Prediction pandas DataFrame.
        'Prediction Names' will be printed in the legend
    raw_key: the key for the predictions dictionary
        that directs to the raw data that each model will
        be subtracting
    model_keys: a list of keys pertaining to the correction
        models that are wanting to be plotted
    compare: a dictionary containing
        'Measure name':measure pandas DataFrame.
        These are what is being plotted against on the
        horizontal-axis
    compare_key: the mkey for the measure desired in
        the compare dictionary. 'compare_key' will be
        printed on the horizontal axis
    site: a single site designiation to be examined in the
        plot. This will be listed as the title of the plot
        
    Returns
    -------
    fig : matplotlib.figure
    ax : matplotlib.axes

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

def stat_corrections_scatter2D(computations: dict, datum_key: str, cor_keys: list, uncor_key: str,
                               sites = [], multi = True, colors = colors99p99):
    """
    Statistical Corrections Plot 2D
        creates a scatter plot of the flow after corrections
        are applied and beforehand in relation to observations
    ----
    compuations: a dictionary containing
        "Correction Name": correction pandas DataFrame
    datum_key: contains the key for the compuations dictionary
        that accesses what baseline the corrections should be
        compared to. This is typically observations
    cor_keys: a list of the keys accessing the correction
        DataFrames in computations. These will be printed in
        the legend.
    uncor_key: the key that accesses the uncorrected data
        in computations
    sites: a list or a singular site to be compared in the plot.
        If multi is set to False and this is not changed to a
        single site, then the first value in the list will be
        chosen
    multi: boolean that determines whether the plot uses data
        from multiple sites or a single site
    colors: a list of colors as strings to be plotted from.
        plotting colors are different for each correction
        DataFrame, but same across sites for a singular
        correction. An error will be thrown if there are
        more cor_keys then colors

    """
    #retreiving DataFrames and establishing data to be plotted
    datum = computations[datum_key]
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
    plt.xlabel(f'{datum_key}-Uncorrected')
    plt.ylabel(f'{datum_key}-Corrected')

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

def anomaly_scatter2D(computations: dict, datum_key: str, vert_key: str, horz_key: str,
                      sites = [], multi = True, colors = colors99p99, show_legend = True):
    """
    Anomaly Plot 2D
        Plots two correction models against each other after
        each is subtracted from a set of Raw data
    ----
    compuations: a dictionary containing
        "Correction Name": correction pandas DataFrame
    datum_key: contains the key for the compuations dictionary
        that accesses what baseline the corrections should be
        compared to. This is typically observations
    vert_key: contains the key for the compuations dictionary
        that accesses the model to be plotted on the vertical
        axis
    horz_key: contains the key for the compuations dictionary
        that accesses the model to be plotted on the horizontal
        axis
    sites: a list or a singular site to be compared in the plot.
        If multi is set to False and this is not changed to a
        single site, then the first value in the list will be
        chosen
    multi: boolean that determines whether the plot uses data
        from multiple sites or a single site
    colors: a list of colors as strings to be plotted from.
        plotting colors are different for each correction
        DataFrame, but same across sites for a singular
        correction. An error will be thrown if there are
        more cor_keys then colors

    """
    #retreiving DataFrames and establishing data to be plotted
    datum = computations[datum_key]
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

    plt.xlabel(f'{datum_key}-{horz_key}')
    plt.ylabel(f'{datum_key}-{vert_key}')
    plt.title("Statistical Correction Anomolies")
    plt.axhline(0)
    plt.axvline(0)
    if show_legend:
        plt.legend(handles = custom_legend(sites, colors))

def rmseFracPlot(data_dict: dict, obs_key: str, sim_keys: list,
                sites = [], multi = True, colors = colors99p99):
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
#      color_code_nxgraph_sorted
#      color_code_nxgraph
#      draw_dataset
#*****************************************************************************************

def find_upstream(topo: xr.Dataset, segID: int, return_segs: list = []):
    """
    find_upstream
        finds what segID is directly upstream from
        the xarray Dataset
    ----
    topo: xarray Dataset of topography
    segID: current segID of interest
    return_segs: list of what segID's are upstream
    """
    upsegs = np.argwhere((topo['Tosegment'] == segID).values).flatten()
    upsegIDs = topo['seg_id'][upsegs].values
    return_segs += list(upsegs)

def find_all_upstream(topo: xr.Dataset, segID: int, return_segs: list = []) -> np.ndarray:
    upsegs = np.argwhere((topo['Tosegment'] == segID).values).flatten()
    upsegIDs = topo['seg_id'][upsegs].values
    return_segs += list(upsegs)
    for upsegID in upsegIDs:
        find_all_upstream(topo, upsegID, return_segs = return_segs)
    return np.unique(return_segs).flatten()

def create_adj_mat(topo: xr.Dataset) -> np.ndarray:
    """
    create_adj_mat
        Forms the adjacency matrix for the graph
        of the topography
        Note that this is independent of whatever
        the segments are called, it is a purely a
        map of the relative object locations
    ----
    topo: xarray Dataset containing topographical information

    return: ndarray that is the adjacency matrix
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
    """
    create_nxGraph
        creates a NetworkX Graph object given an
        adjacency matrix
    ----
    adj_mat: a numpy ndarray containing the desired
        adjacency matrix

    returns: NetworkX Graph of respective nodes
    """
    topog = nx.from_numpy_matrix(adj_mat)
    return topog

def organize_nxgraph(topo: nx.Graph):
    """
    organize_nxgraph
        orders the node positions hierarchical based on
        the "dot" layout and given topography Dataset
    ----
    topo: xarray Dataset containing segment identifications
    """
    pos = nx.drawing.nx_agraph.graphviz_layout(topo, prog = 'dot')
    return pos

def color_code_nxgraph_sorted(graph: nx.graph, measure: pd.Series,
                              cmap=mpl.cm.get_cmap('plasma'))-> dict:
    """
    color_cod_nxgraph
        creates a dictionary mapping of nodes
        to color values
    ----
    graph: nx.graph to be color coded

    measure: pd.Series with segment ID's as
        the index and desired measures as values
    """
    #sets up color diversity
    segs = measure.sort_values().index
    color_steps = np.arange(0, 1, 1 / len(segs))

    color_dict =  {f'{seg}': mpl.colors.to_hex(cmap(i)) for i, seg in zip(color_steps, segs)}
    return color_dict

def color_code_nxgraph(graph: nx.graph, measure: pd.Series,
                       cmap = mpl.cm.get_cmap('coolwarm_r'))-> dict:
    """
    color_code_nxgraph
        creates a dictionary mapping of nodes
        to color values
    ----
    graph: nx.graph to be color coded

    measure: pd.Series with segment ID's as
        the index and desired measures as values

    cmap: colormap to be used

    """
    if np.where(measure < 0)[0].size == 0:
        # meaning we have only positive values and do not need to establish
        # zero at the center of the color bar
        segs = measure.index
        minimum = 0 #set to zero to preserve the coloring of the scale
        maximum = measure.max()

        color_vals = (measure.values) / (maximum)
        color_bar = plt.cm.ScalarMappable(cmap=cmap, norm = plt.Normalize(vmin = minimum, vmax = maximum))

        color_dict =  {f'{seg}' : mpl.colors.to_hex(cmap(i)) for i, seg in zip(color_vals, segs)}
        return color_dict, color_bar
    else:
        #determine colorbar range
        extreme = abs(measure.max())
        if np.abs(measure.min()) > extreme:
            extreme = np.abs(measure.min())


        #sets up color values
        segs = measure.index
        color_vals = (measure.values + extreme) / (2 * extreme)
        color_bar = plt.cm.ScalarMappable(cmap=cmap, norm = plt.Normalize(vmin = -extreme, vmax = extreme))

        color_dict =  {f'{seg}': mpl.colors.to_hex(cmap(i)) for i, seg in zip(color_vals, segs)}
        return color_dict, color_bar

def draw_dataset(topo: xr.Dataset, color_measure: pd.Series, cmap = mpl.cm.get_cmap('coolwarm_r')):
    """
    draw_dataset
        draws a networkx graph from a topological
        xrarray Dataset and color codes it based on
        a pandas Series
    ----
    topo: xr.Dataset containing topologcial information

    color_measure: pd.Series where indicies are concurrent
        with the number of segs in topo. Typically this contains
        statistical information about the flows that will be
        color coded by least to greatest value

    cmap: a mpl colormap to use in conjunction with color_measure.
        The default is diverging color map, 'coolwarm'
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
#      plot_reduced_doy_flows
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

def plot_reduced_doy_flows(flow_dataset: xr.Dataset, plot_sites: list, 
                        reduce_func=np.mean, 
                        vertical_label=f'Mean Day of Year Flow 'r'$(m^3/s)$',
                        title_label=f'Annual Mean Flows',                           
                        raw_var = 'IRFroutedRunoff', raw_name = 'Mizuroute Raw', 
                        ref_var = 'upstream_ref_flow', ref_name = 'upstream_ref_flow', 
                        bc_vars = list(), bc_names = list(),                            
                        fontsize_title = 80, fontsize_legend = 68, fontsize_subplot = 60, 
                        fontsize_tick = 45, fontcolor = 'black', 
                        figsize_width = 70, figsize_height = 30,
                        plot_colors = ['grey', 'black', 'blue', 'red'],
                        return_reduced_flows = False):
    """
    Plot Mean Day of Year Flows
        creates a series of subplots that plot an average year's flows
        per gauge site
    ---
    flow_dataset: xr.Dataset
        contatains raw, reference, and bias corrected flows
    plot_sites: list
        a list of sites to be plotted
    reduce_func: function
        a function to apply to flows grouped by dayofyear,
        default = np.mean
    vertical_label: str
        a string label for the vertical axis representing
        the reduce_func, defaults as:
        f'Mean Day of Year Flow 'r'$(m^3/s)$' to fit np.mean
    title_label: str
        a string lable for the figure title representing
        the reduce_func, defaults as:
        f'Annual Mean Flows' to fit np.mean
    raw_var: str
        the string to access the raw flows in flow_dataset
    raw_name: str
        the string to label the raw flows in the legend
    ref_var: str
        the string to access the reference flows in flow_dataset
    ref_name: str
        the string to label the refernce flows in the legend
    bc_vars: list
        a list of strings to access the bias corrected flows in flow_dataset
    bc_names: list
        a list of strings to label the reference flows in the legend
    plot_colors: ['grey', 'black', 'blue', 'red']
        a list containing colors to be plotted for raw, ref, bc,
        and bc_alt, respectively
    return_reduced_flows : boolean
        if True, returns the reduced flows as calculated for plotting
    ----
    Returns: fig, axs
    """
    
    if len(bc_vars) == 0:
        raise Exception("Please enter a non-zero number strings in bc_vars to be used")
    if len(bc_vars) != len(bc_names):
        raise Exception("Please have the same number of entries in bc_names as bc_names")
    if len(plot_colors) < 2 + len(bc_vars):
        raise Exception(f"Please enter at least {2 + len(bc_vars)} colors in plot_colors")
    
    raw_flow_doy = flow_dataset[raw_var].groupby(
        flow_dataset['time'].dt.dayofyear).reduce(reduce_func)
    reference_flow_doy = flow_dataset[ref_var].groupby(
        flow_dataset['time'].dt.dayofyear).reduce(reduce_func)
    bc_flow_doys = list()
    for bc_var in bc_vars:
        bc_flow_doys.append(flow_dataset[bc_var].groupby(flow_dataset['time'].dt.dayofyear).reduce(reduce_func))        

    doy = raw_flow_doy['dayofyear'].values
    outlet_names = flow_dataset['seg'].values
   
    raw_flow_doy_df = pd.DataFrame(data = raw_flow_doy.values, 
                                   index=doy, columns = outlet_names)
    reference_flow_doy_df = pd.DataFrame(data = reference_flow_doy.values, 
                                         index=doy, columns = outlet_names)
    bc_flow_doy_dfs = list()
    for bc_flow_doy in bc_flow_doys:
        bc_flow_doy_dfs.append(pd.DataFrame(data = bc_flow_doy.values, index = doy, columns = outlet_names))
    
    plot_names = [raw_name, ref_name]
    plot_names.extend(bc_names)
    
    mpl.rcParams['figure.figsize'] = (figsize_width, figsize_height)
    n_rows, n_cols = determine_row_col(len(plot_sites))
    fig, axs = plt.subplots(n_rows, n_cols)
    fig.suptitle(title_label, fontsize = fontsize_title, color = fontcolor, y = 1.05)
                                          
    for site, ax in zip(plot_sites, axs.ravel()):
        ax.plot(raw_flow_doy_df[site], color = plot_colors[0], alpha = 0.8, lw = 4)
        ax.plot(reference_flow_doy_df[site], color = plot_colors[1], alpha = 0.8, lw = 4)
        
        for i, bc_flow_doy_df in enumerate(bc_flow_doy_dfs):
            ax.plot(bc_flow_doy_df[site], color = plot_colors[2+i], lw = 4, alpha = 0.8)
        
        ax.set_title(site, fontsize = fontsize_subplot, color = fontcolor)
        plt.setp(ax.spines.values(), color = fontcolor)
        ax.tick_params(axis = 'both', colors = fontcolor, labelsize = fontsize_tick)
        
    if len(plot_sites) < n_rows * n_cols:
        for ax_index in np.arange(len(plot_sites), n_rows * n_cols):
            axs.ravel().tolist()[ax_index].axis('off')

    fig.text(0.5, -0.02, 'Day of Year', fontsize = fontsize_title, ha = 'center')
    fig.text(-0.02, 0.5, vertical_label, fontsize = fontsize_title, 
             va = 'center', rotation = 'vertical')
    plt.subplots_adjust(wspace = 0.2, hspace = 0.5, left = 0.05, right = 0.8, top = 0.95)

    fig.legend(plot_names, fontsize = fontsize_legend, loc = 'center right');
    
    if return_reduced_flows:
        reduced_flows = xr.Dataset(coords={'site': plot_sites, 'time': doy})
        reduced_flows[raw_var] = xr.DataArray(data = raw_flow_doy_df.values, dims = ('time', 'site') ).transpose()
        reduced_flows[ref_var] = xr.DataArray(data = reference_flow_doy_df.values, dims = ('time', 'site') ).transpose()
        for bc_var, bc_flow_doy_df in zip(bc_vars, bc_flow_doy_dfs):
            reduced_flows[bc_var] = xr.DataArray(data = bc_flow_doy_df.values, dims = ('time', 'site') ).transpose()
        return reduced_flows
    
    return fig, ax
    
def plot_spearman_rank_difference(flow_dataset:xr.Dataset, gauge_sites: list, 
                                  start_year: str, end_year: str, 
                                  relative_locations_triu: pd.DataFrame, basin_map_png, 
                                  cmap = mpl.cm.get_cmap('coolwarm_r'),
                                  blank_plot_color = 'w', fontcolor = 'black',
                                  fontsize_title=60, fontsize_tick = 30, fontsize_label = 45):
    """
    Plot Differences in Spearman Rank
        creates a site-to-site rank correlation difference comparioson
        plot with a map of the given basin
    ----
    flow_dataset: xr.Dataset
        contains raw as 'raw_flow' and bias corrected as 'bias_corrected_total_flow'
        flow values, times of the flow values, and the names of the sites where those
        flow values exist
    gauge_sites: list
        a list of gauge sites to be plotted
    start_year: str
        string formatted as 'yyyy-mm-dd' to start rank correlation window
    end_year: str
        string formatted as 'yyyy-mm-dd' to end rank correlation winodw
    relative_locations_triu: pd.DataFrame
        denotes which sites are connected with a '1' and has the lower
        triangle set to '0'
    basin_map_png:
        a png file containing the basin map with site values marked    
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
    
def correction_scatter(site_dict: dict, raw_flow: pd.DataFrame, 
                       ref_flow: pd.DataFrame, bc_flow: pd.DataFrame, 
                       colors: list, title= 'Flow Residuals', 
                       fontsize_title = 80, fontsize_legend = 68, 
                       fontsize_subplot = 60, fontsize_tick = 45, fontcolor = 'black',
                       pos_cone_guide = False, neg_cone_guide = False):
    """
    Correction Scatter
        Plots differences between the raw and reference flows on the horizontal
        and differences between the bias corrected and refrerence on the vertical.
        This compares corrections needed before and after the bias correction method
        is applied.
    ----
    site_dict: dict {subgroup name: list of segments in subgroup}
        how sites are to be seperated
    raw_flow: pd.DataFrame
        accesses the raw flows in the flow_dataset
    ref_flow: pd.DataFrame
        accesses the reference flows in the flow_dataset
    bc_flow: pd.DataFrame
        accesses the bias corrected flows in the flow_dataset
    colors
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
            ax.plot([-ref_line_ext,ref_line_ext], [0,0], color='k', linestyle='--')
            
            if pos_cone_guide and neg_cone_guide:
                ax.plot([-ref_line_ext,ref_line_ext], [-ref_line_ext,ref_line_max], color='k', linestyle='--')
                ax.plot([-ref_line_ext,ref_line_ext], [ref_line_ext,-ref_line_max], color='k', linestyle='--')
            elif pos_cone_guide:
                ax.plot([0,ref_line_ext], [0,ref_line_ext], color='k', linestyle='--')
                ax.plot([0,ref_line_ext], [0,-ref_line_ext], color='k', linestyle='--')
            elif neg_cone_guide:
                ax.plot([0,-ref_line_ext], [0,ref_line_ext], color='k', linestyle='--')
                ax.plot([0,-ref_line_ext], [0,-ref_line_ext], color='k', linestyle='--')
            
        else:
            ax.axis('off')
    
    fig.text(0.5, -0.04, r'$Q_{ref} - Q_{raw} \quad (m^3/s)$', ha='center', va = 'bottom', 
             fontsize=fontsize_title, color=fontcolor);
    fig.text(-0.02, 0.5, r'$Q_{ref} - Q_{bc} \quad (m^3/s)$', va='center', rotation = 'vertical', 
             fontsize=fontsize_title, color=fontcolor);    
    
    plt.tight_layout()
    plt.show

def pbias_diff_hist(sites: list, colors: list, raw_flow: pd.DataFrame, ref_flow: pd.DataFrame, 
                      bc_flow: pd.DataFrame, grouper=pd.Grouper(freq='M'), total_bins=None,
                      title_freq='Monthly', fontsize_title=90, fontsize_subplot_title=60, 
                    fontsize_tick=40, fontsize_labels=84):
    """
    Percent Bias Difference Histogram
        creates a number of histogram subplots by each given site that 
        plot the difference in percent bias before and after bias correction 
    ----
    sites: list
        a list of sites that are the columns of the flow DataFrames
    colors: list
        a list of colors to plot the sites with, (do not have to be different),
        that are used in the same order as the list of sites
    raw_flow: pd.DataFrame
        contains the flows before bias correction
    ref_flow: pd.DataFrame
        contains the reference flows for comparison
    bc_flow: pd.DataFrame
        contains the flows after bias correction
    grouper: pd.Grouper(freq='M')
        how flows should be grouped for bia correction
    total_bins: None
        number of bins to use in the histogram plots. if none specified,
        defaults to the square root of the number of pbias difference
        values
    title_freq: 'Monthly'
        an adjective description of the frequency with which the flows are
        grouped
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
    """
    Plot Residual Overlay
        plots residuals from each hydrologic year on top of each
        other with a refence line at q=0
    ----
    flows: pd.DataFrame
        contains all flows to be used in plotting
    upstream_sites: list
        contains strings of the site names stored in flows to
        aggregate
    downstream_sites: str
        the name of the downstream site stored in flows that will
        have the upstream_sites subtracted from it
    start_year: int
        the starting year to plot
    end_year: int
        the year to conclude on
        
    Returns: ax containing this plot
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
    """
    Normalized Change in Annual Flow Volume
        plots a series of subplots containing bar charts that depict
        the differnece in normalized annual flow volume due to bias correction
    ----
    sites: list
        contatins the string names of all the sites to be plotted, matching sites
        contained in the DataFrames
    before_bc: pd.DataFrame
        contains flows, (not aggregated) before bias correction is applied
    after_bc: pd.DataFrame
        contains flows, (not aggregated) after bias correction is applied
    colors: list
        contains the colors to be used for each site's subplot in the same order
        as sites, (does not have to be unique)    
    """
    
    mpl.rcParams['figure.figsize']=(30,20)    

    WY_grouper = calc_water_year(before_bc)
    after_bc_annual = after_bc.groupby(WY_grouper).sum()
    before_bc_annual = before_bc.groupby(WY_grouper).sum()

    after_bc_annual, before_bc_annual = dst.normalize_flow_pair(data=after_bc_annual, norming_data=before_bc_annual)

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
    """
    Percent Bias Comparison Histogram
        creates a number of histogram subplots by each given sites that
        plot percent bias both before and after bias correction
    ----
    sites: list
        a list of sites that are the columns of the flow DataFrames
    raw_flow: pd.DataFrame
        contains the flows before bias correction
    ref_flow: pd.DataFrame
        contains the reference flows for comparison
    bc_flow: pd.DataFrame
        contains the flows after bias correction
    grouper: pd.Grouper(freq='M')
        how flows should be grouped for bia correction
    total_bins: None
        number of bins to use in the histogram plots. if none specified,
        defaults to the square root of the number of pbias difference
        values
    title_freq: 'Monthly'
        an adjective description of the frequency with which the flows are
        grouped
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
    """
    Compare Probability Distribution Functions
        plots the PDF's of the raw, reference, and bias corrected flows
        for each gauge site
    ----
    flow_dataset: xr.Dataset
        contatains raw, reference, and bias corrected flows
    gauges_sites: list
        a list of gauge sites to be plotted
    raw_var: str
        the string to access the raw flows in flow_dataset
    ref_var: str
        the string to access the reference flows in flow_dataset
    bc_var: str
        the string to access the bias corrected flows in flow_dataset
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

def compare_CDF(flow_dataset:xr.Dataset, plot_sites: list,
                      raw_var: str, raw_name: str,
                      ref_var: str, ref_name: str,
                      bc_vars: list, bc_names: list,
                      plot_colors: list, logit_scale = True,
                      logarithm_base = '10', units = r'Q [$m^3/s$]',
                      markers = ['o', 'x', '*', '*'],
                      figsize = (20,20), sharex = False, sharey = True,
                      fontsize_title = 40, fontsize_labels = 30, fontsize_tick = 20,
                      markersize = 1, alpha = 0.3):
    """
    Compare Probability Distribution Functions Logit
        plots the CDF's of the raw, reference, and bias corrected flows
    ----
    flow_dataset: xr.Dataset
        contatains raw, reference, and bias corrected flows
    gauges_sites: list
        a list of gauge sites to be plotted
    raw_var: str
        the string to access the raw flows in flow_dataset
    raw_name: str
        the string to label the raw flows in the legend
    ref_var: str
        the string to access the reference flows in flow_dataset
    ref_name: str
        the string to label the reference flows in the legend
    bc_var: str
        the string to access the bias corrected flows in flow_dataset
    bc_name: str
        the string to label the bias corrected flows in the legend
    bc_var_alt: None
        the string to access a second set of bias corrected flows in
        flow_dataset (optional)
    bc_var_name: None
        the string to label the second set of bias corrected flows in
        the legend (required if bc_var_alt is not None)
    logit_scale: True
        whether to plot the vertical scale on a logit axis (True) or not (False)
    logarithm_base: '10'
        '10' to use a log10 horizontal scale
        'e' to use a natural log horizontal scale
    units: r'Q [$m^3/s$]'
        the horizontal axis's label for units
    plot_colors: ['grey', 'black', 'blue', 'red']
        a list containing the colors to be plotted for raw, ref, bc,
        and bc_alt, respectively
    markers: ['o', 'x', '*', '*']
        a list containing the markers to be plotted for raw, ref, bc,
        and bc_alt, respectively
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
    
    n_rows, n_cols = bmorph.plotting.determine_row_col(len(plot_sites))
    
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
    
    """
    Spearman Rank Difference Boxplots Annual
        creates box plots for each stide pairing determing the difference in spearman
        rank for each year between the raw and bias corrected data
    ----
    raw_flows: pd.DataFrame
        contains the raw flows with sites in the columns and time in the index
    bc_flows: pd.DataFrame
        conains the bias corrected flows with sites in the columns and time in the index
    site_pairings: list of list of string site pairs
        e.g. [['downstream_name','upstream_name'],...]
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
                                 plot_colors: list,
                                 fontsize_title = 40, fontsize_tick = 30, fontsize_labels = 40, 
                                 fontsize_legend = 30,
                                 showfliers = False, sharex = True, sharey = 'row', TINY_VAL = 1e-6, 
                                 figsize = (30,20), show_y_grid = True):
    """
    Kullback Liebler Divergence Annual Comparison
        plots the KL divergence for each year per site as
        KL(P_{ref} || P_{raw}) and KL( P_{ref} || P_{bc})
    ----
    flow_dataset: xr.Dataset
        contains raw, reference, and bias corrected flows
    sites: list
        contains all the sites to be plotted, (note that if the number of sites to be plotted
        is square or rectangular, the last site will not be plotted to save room for the legend)    
    raw_var: str
        the string to access the raw flows in flow_dataset
    raw_name: str
        the string to label the raw flows in the legend and horizontal labels
    ref_var: str
        the string to access the reference flows in flow_dataset
    ref_name: str
        the string to label the reference flows in the legend and horizontal labels
    bc_vars: list
        a list of strings to access the bias corrected flows in flow_dataset
    bc_names: list
        a list of string to label the bias corrected flows in the legend and horizontal labels
    plot_colors: list
        a list containing colors to be plotted for raw and the bias corrected flows, respectively
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
    
    plt.suptitle("Annual KL Diveregence Before/After Bias Correction", fontsize = fontsize_title, y = 1.05)

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
    
    plt.legend(handles=custom_legend(names=plot_labels, colors = plot_colors), 
               fontsize = fontsize_legend, loc = 'lower right')

    plt.tight_layout()
    
    return fig, axs
    
def spearman_diff_boxplots_annual_compare(flow_dataset: xr.Dataset, site_pairings, 
                                  raw_var: str, bc_vars: list, bc_names: list,
                                  plot_colors: list, showfliers =  True,
                                  fontsize_title=40, fontsize_tick=25, fontsize_labels=30, 
                                  subtitle = None, figsize = (20,20), sharey = 'row'):
    
    """
    Spearman Rank Difference Boxplots Annual
        creates box plots for each stide pairing determing the difference in spearman
        rank for each year between the raw and bias corrected data
    ----
    flow_dataset: xr.Dataset
        contains raw and bias corrected flows
    site_pairings: list of list of string site pairs
        e.g. [['downstream_name','upstream_name'], ...]
    raw_var: str
        the string to access the raw flows in flow_dataset
    bc_vars: list
        a list of strings to access the bias corrected flows in flow_dataset
    bc_names: list
        a list of strings to name the bias corrected flows from flow_dataset
    plot_colors: list
        a list of colors that are in the same order as the bc_vars and bc_names
        to be used in plotting        
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
    """
    Compare Probability Distribution Functions All
        plots the CDF's of the raw, reference, and bias corrected flows with data 
        from all sites in plot_sites combined for a summary statistic
    ----
    flow_dataset: xr.Dataset
        contatains raw, reference, and bias corrected flows
    gauges_sites: list
        a list of gauge sites to be plotted
    raw_var: str
        the string to access the raw flows in flow_dataset
    raw_name: str
        the string to label the raw flows in the legend
    ref_var: str
        the string to access the reference flows in flow_dataset
    ref_name: str
        the string to label the reference flows in the legend
    bc_var: str
        the string to access the bias corrected flows in flow_dataset
    bc_name: str
        the string to label the bias corrected flows in the legend
    bc_var_alt: None
        the string to access a second set of bias corrected flows in
        flow_dataset (optional)
    bc_var_name: None
        the string to label the second set of bias corrected flows in
        the legend (required if bc_var_alt is not None)
    logit_scale: True
        whether to plot the vertical scale on a logit axis (True) or not (False)
    logarithm_base: '10'
        '10' to use a log10 horizontal scale
        'e' to use a natural log horizontal scale
    units: r'Q [$m^3/s$]'
        the horizontal axis's label for units
    plot_colors: ['grey', 'black', 'blue', 'red']
        a list containing the colors to be plotted for raw, ref, bc,
        and bc_alt, respectively
    markers: ['o', 'x', '*', '*']
        a list containing the markers to be plotted for raw, ref, bc,
        and bc_alt, respectively
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

def compare_mean_grouped_CPD(flow_dataset:xr.Dataset, plot_sites: list, grouper_func,
                             raw_var: str, raw_name: str,
                             ref_var: str, ref_name: str,
                             bc_vars: list, bc_names: list,
                             plot_colors: list, subset_month = None,
                             units = r'Mean Annual Flow [$m^3/s$]',
                             figsize = (20,20), sharex = False, sharey = True, 
                             pp_kws = dict(postype='cunnane'), fontsize_labels = 30, fontsize_tick = 20,
                             fontsize_legend = 24, markersize = 1, alpha = 0.3, legend_bbox_to_anchor = (1.4, 1)
                      ):
    """
    Cumulative Probability Distributions
        plots the CPD's of the raw, reference, and bias corrected flows on a probability axis
    ----
    flow_dataset: xr.Dataset
        contatains raw, reference, and bias corrected flows
    gauges_sites: list
        a list of gauge sites to be plotted
    grouper_func
        function to group a pandas.DataFrame index by to calculate the 
        mean of the grouped values
    raw_var: str
        the string to access the raw flows in flow_dataset
    raw_name: str
        the string to label the raw flows in the legend
    ref_var: str
        the string to access the reference flows in flow_dataset
    ref_name: str
        the string to label the reference flows in the legend
    bc_var: str
        the string to access the bias corrected flows in flow_dataset
    bc_name: str
        the string to label the bias corrected flows in the legend
    bc_var_alt: None
        the string to access a second set of bias corrected flows in
        flow_dataset (optional)
    bc_var_name: None
        the string to label the second set of bias corrected flows in
        the legend (required if bc_var_alt is not None)
    plot_colors: ['grey', 'black', 'blue', 'red']
        a list containing the colors to be plotted for raw, ref, bc,
        and bc_alt, respectively
    subset_month: None
        the integer date of a month to subset out for plotting,
        (ex: if you want to subset out January, enter 1)
    units:  r'Mean Annual Flow [$m^3/s$]'
        the horizontal axis's label for units
    pp_kws: dict(postype='cunnane')
        plotting position computation as specified by
        https://matplotlib.org/mpl-probscale/tutorial/closer_look_at_plot_pos.html
    """
    
    if len(bc_vars) == 0:
        raise Exception("Please enter a non-zero number strings in bc_vars to be used")
    if len(bc_vars) != len(bc_names):
        raise Exception("Please have the same number of entries in bc_names as bc_names")
    if len(plot_colors) < len(bc_vars):
        raise Exception(f"Please enter at least {len(bc_vars)} colors in plot_colors")
    
    n_rows, n_cols = bmorph.plotting.determine_row_col(len(plot_sites))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize = figsize, sharex = sharex, sharey = sharey)
    axes = axes.flatten()      
    
    time = flow_dataset['time'].values
    raw_flow_df = pd.DataFrame(data = flow_dataset[raw_var].values, index=time, columns = plot_sites)
    ref_flow_df = pd.DataFrame(data = flow_dataset[ref_var].values, index=time, columns = plot_sites)
    bc_flow_dfs = list()
    for bc_var in bc_vars:
        bc_flow_df = pd.DataFrame(data = flow_dataset[bc_var].values, index = time, columns = plot_sites)
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
        ax = axes[i]
        
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
                           scatter_kws=dict(linestyle='none', marker='.', alpha=alpha, color = plot_colors[0], label=raw_name), 
                           **common_opts)
        probscale.probplot(ref, ax=ax, pp_kws=pp_kws, 
                           scatter_kws=dict(linestyle='none', marker='.', alpha=alpha, color = plot_colors[1], label=ref_name), 
                           **common_opts)
        
        for j, cor in enumerate(cors):
            probscale.probplot(cor, ax=ax, pp_kws=pp_kws, 
                               scatter_kws=dict(linestyle='none', marker='.', alpha=alpha, color = plot_colors[2+j], label=bc_names[j]), 
                               **common_opts)
        
        ax.set_title(site, fontsize = fontsize_labels, position = (0.2, 0.8))
        ax.tick_params(axis = 'both', labelsize = fontsize_tick)
        ax.set_ylim([y_min, y_max])
        ax.set_xlim(left=0.01, right=99.99)
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

