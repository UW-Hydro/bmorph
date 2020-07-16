import numpy as np
import xarray as xr
import pandas as pd
from typing import List
import matplotlib as mpl
import matplotlib.pyplot as plt

import networkx as nx
import graphviz as gv
import pygraphviz as pgv
from networkx.drawing.nx_agraph import graphviz_layout

from bmorph.evaluation.constants import colors99p99
from bmorph.evaluation import descriptive_statistics as dst

def custom_legend(names: List,colors=colors99p99):
    """
    Custom Legend
        creates a list of patches to be passed in as
        `handels` for the plt.legends function
    ----
    names: a list of legend names
    colors: a list of colors

    It is assumed that the order of colors and corresponding
    names for plotted values match already
    """
    legend_elements = list()
    for i,name in enumerate(names):
        legend_elements.append(mpl.patches.Patch(facecolor=colors[i],label=name))
    return legend_elements

def calc_water_year(df: pd.DataFrame):
    """
    Calculate Hydrologic Year
    ----

    df: pandas DataFrame with a DataTime index

    Returns: an index grouped by Hydrologic Year

    """
    return df.index.year + (df.index.month >= 10).astype(int)

def find_index_hyear(data: pd.DataFrame) -> np.int:
    """
    finds the index of the first hydrologic year
    ----
    data: pd.DataFrame
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

def pbias_sites(observed: pd.DataFrame, predicted: pd.DataFrame):
    """
    Percent Bias Sites
        calculates percent bias on a hydrologic year
        and site-by-site basis
    ----
    observed: pd.DataFrame
        dataframe containing all observations
    predicted: pd.DataFrame
        dataframe containing all predictions
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

    pbias_site_df = pd.DataFrame(columns = observed.columns, index = pd.Series(range(0,hyears)))
    pbias_current_year = pd.DataFrame(columns = observed.columns)

    for i in range(0,hyears):
        #establish end of hydraulic year
        hyear_end = hyear_start + pd.Timedelta(364.25, unit = 'd')

        #need to truncate datetimes since time indicies do not align
        O = observed.loc[hyear_start:hyear_end]
        P = predicted.loc[hyear_start:hyear_end]
        O.index = O.index.floor('d')
        P.index = P.index.floor('d')

        pbias_current_year = dst.pbias(O,P)

        #puts the computations for the hydraulic year into our bigger dataframe
        pbias_site_df.iloc[i] = pbias_current_year.iloc[0].T

        #set up next hydraulic year
        hyear_start = hyear_start + pd.Timedelta(365.25,unit = 'd')

    return pbias_site_df

def diff_maxflow_sites(observed: pd.DataFrame, predicted: pd.DataFrame):
    """
    Difference Max Flow Sites
        calculates difference in maximum flows
        on a hydrologic year and site-by-site basis
    ----
    observed: pd.DataFrame
        dataframe containing all observations
    predicted: pd.DataFrame
        dataframe containing all predictions
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

    diff_maxflow_sites_df = pd.DataFrame(columns = observed.columns, index = pd.Series(range(0,hyears)))
    diff_maxflow_current_year = pd.DataFrame(columns = observed.columns)

    for i in range(0,hyears):
        #establish end of hydraulic year
        hyear_end = hyear_start + pd.Timedelta(364.25, unit = 'd')

        #need to truncate datetimes since time indicies do not align
        O = observed.loc[hyear_start:hyear_end]
        P = predicted.loc[hyear_start:hyear_end]
        O.index = O.index.floor('d')
        P.index = P.index.floor('d')

        diff_maxflow_current_year = P.max().to_frame().T - O.max().to_frame().T

        #puts the computations for the hydraulic year into our bigger dataframe
        diff_maxflow_sites_df.iloc[i] = diff_maxflow_current_year.iloc[0].T

        #set up next hydraulic year
        hyear_start = hyear_start + pd.Timedelta(365.25,unit = 'd')


    return diff_maxflow_sites_df

def pbias_plotter(observed: pd.DataFrame, names: list, colors: list, *models: pd.DataFrame):
    """
    Percent Bias Box Plotter
        plots box plots of numerous models grouped
        by site
    ----
    observed: pd.Dataframe
        a dataframe containing observations
    names: list
        a list of the model names
    colors: list
        a list of colors to be plotted
    *models: any number of pd.DataFrame objects
        to be evaluated
    """
    num_models = len(models)
    sites = observed.columns
    pbias_models = list()

    #runs each model through pbias_sites
    for model in models:
        pbias_models.append(pbias_sites(observed,model))

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
            i = i+1

        boxplots = plt.boxplot(df.T, positions = np.arange(position,position+num_models),patch_artist=True)

        for patch, color in zip(boxplots['boxes'], colors):
            patch.set_facecolor(color)

        position = position+num_models+1

    tick_location = list()
    start_tick = int(np.ceil(len(models)/2))
    tick_spacing = len(models)+1
    for j in range(0,len(sites)):
        tick_location.append(start_tick+j*tick_spacing)

    ax.set(xticks = tick_location, xticklabels = sites)
    plt.xticks(rotation = 90)

    ax.legend(handles=custom_legend(names,colors),loc='upper left')
    return fig, ax

def diff_maxflow_plotter(observed: pd.DataFrame, names: list, colors: list, *models: pd.DataFrame):
    """
    Difference Max Flow Box Plotter
        plots box plots of numerous models grouped
        by site
    ----
    observed: pd.Dataframe
        a dataframe containing observations
    names: list
        a list of the model names
    colors: list
        a list of colors to be plotted
    *models: any number of pd.DataFrame objects
        to be evaluated
    """
    num_models = len(models)
    sites = observed.columns
    diff_maxflow_models = list()

    #runs each model through pbSites
    for model in models:
        diff_maxflow_models.append(diff_maxflow_sites(observed,model))

    fig = plt.figure()
    ax = plt.axes(xlabel = 'Sites', ylabel = 'Difference in Max Flow')
    position = 1
    for site in sites:
        df = pd.DataFrame(index = diff_maxflow_models[0].index)

        #fill out the dataframe for a single site with each model's percent bias
        i = 0
        for model in diff_maxflow_models:
            entry = f"{site}:{names[i]}"
            df[entry] = model[site]
            i = i+1

        bp = plt.boxplot(df.T, positions = np.arange(position,position+num_models),patch_artist = True)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        position = position+num_models+1
        #follow plot style: https://stackoverflow.com/questions/16592222/matplotlib-group-boxplots

    tick_location = list()
    start_tick = int(np.ceil(len(models)/2))
    tick_spacing = len(models)+1
    for j in range(0,len(sites)):
        tick_location.append(start_tick+j*tick_spacing)

    ax.set(xticks = tick_location, xticklabels = sites)
    plt.xticks(rotation = 45)
    plt.title('Yearly Difference in Max Flow due to Correction')

    ax.legend(handles=custom_legend(names,colors),loc='upper left')

    return fig,ax

def scatter_series_axes(data_x,data_y,label:str,color:str,alpha:float,ax=None) -> plt.axes:
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(data_x,data_y,label=label,color=color,alpha=alpha)
    return ax

def site_diff_scatter(predictions: dict, raw_key: str, model_keys: list, compare: dict, compare_key: str,
        site: str, colors=colors99p99):
    """
    Site Differences Scatter Plot
        creates a scatter plot of Raw-BC versus some measure
    ----

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

    """
    #retreiving DataFrames and establishing data to be plotted
    raw = predictions[raw_key]
    raw = raw.loc[:,site]

    Y = list()
    for model_key in model_keys:
        predict = predictions[model_key]
        Y.append(raw-predict.loc[:,site])

    X = compare[compare_key]
    X = X.loc[:,site]
    fig,ax = plt.subplots()
    for i,y in enumerate(Y):
        scatter_series_axes(X,y,model_keys[i],color=colors[i],alpha=0.05,ax=ax)
    plt.xlabel(compare_key)
    plt.ylabel('Raw-BC')
    plt.title(site)
    plt.axhline(0)
    plt.legend(handles=custom_legend(model_keys,colors))

    return fig,ax

def stat_corrections_scatter2D(computations: dict, datum_key: str, cor_keys: list, uncor_key: str,
               sites=[], multi=True, colors=colors99p99):
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
    for i,cor_key in enumerate(cor_keys):
        Y = datum - computations[cor_key]


        if multi == True:
            for site in sites:
                    x = X.loc[:,site]
                    y = Y.loc[:,site]

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

                    scatter_series_axes(x,y,site,colors[i],0.05,ax)
        else: #meaning site should be set to a singular value
            #double check that this was actually changed, otherwise picks first value
            site = sites
            if isinstance(sites,list):
                site = sites[0]

            x = X.loc[:,site]
            y = Y.loc[:,site]
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

            scatter_series_axes(x,y,site,colors[i],0.05,ax)


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

    minlin = minlin*0.9
    maxlin = maxlin*1.1

    plt.plot([minlin, maxlin], [minlin, maxlin])
    plt.legend(handles=custom_legend(cor_keys,colors))

    return fig, ax

def anomaly_scatter2D(computations: dict, datum_key: str, vert_key: str, horz_key: str,
               sites=[], multi=True, colors=colors99p99,show_legend=True):
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
    fig,ax = plt.subplots()

    if multi == True:
        for site in sites:
            x = X.loc[:,site]
            y = Y.loc[:,site]
            scatter_series_axes(x,y,site,colors[i],0.05,ax)
            i = i + 1

            if i >= len(colors):
                #recycles colors if all exhausted
                i = 0
    else: #meaning site should be set to a singular value
        #double check that this was actually changed, otherwise picks first value
        site = sites
        if type(sites) == list:
            site = sites[0]

        X = X.loc[:,site]
        Y = Y.loc[:,site]
        scatter_series_axes(X,Y,site,colors[i],0.05,ax)

    plt.xlabel(f'{datum_key}-{horz_key}')
    plt.ylabel(f'{datum_key}-{vert_key}')
    plt.title("Statistical Correction Anomolies")
    plt.axhline(0)
    plt.axvline(0)
    if show_legend:
        plt.legend(handles=custom_legend(sites,colors))

def rmseFracPlot(data_dict: dict,obs_key:str,sim_keys:list,
                sites=[],multi=True,colors=colors99p99):
    #retrieving data and flooring time stamps
    observations = data_dict[obs_key]
    observations.index = observations.index.floor('d')

    color_num = 0
    for sim_key in sim_keys:
        predictions = data_dict[sim_key]
        predictions.index = predictions.index.floor('d')

        N = len(predictions.index)
        rmse_tot = dst.rmse(observations,predictions)
        rmse_n = pd.DataFrame(index = np.arange(0,N))

        errors = predictions - observations
        square_errors = errors.pow(2)
        if multi == True:

            #constructs a dataframe where each column is independently sorted
            for site in sites:
                square_errors_site = square_errors.loc[:,site].sort_values(ascending=False)
                rmse_n[site]=square_errors_site.values


            #performs cumulative mean calcualtion
            for site in sites:
                vals = rmse_n[site].values
                mat = np.vstack([vals,]*N).T
                nantri = np.tri(N)
                nantri[nantri==0]=np.nan
                mat_mean = np.nanmean(mat*nantri,axis=1)
                mat_rmse = np.power(mat_mean,0.5)
                rmse_n[site] = np.divide(mat_rmse,rmse_tot[site].values)
        else:

            site = sites
            if type(sites) == list:
                site = sites[0]

            square_errors_site = square_errors.loc[:,site].sort_values(ascending=False)
            rmse_n[site]=square_errors_site.values
            vals = rmse_n[site].values
            mat = np.vstack([vals,]*N).T
            nantri = np.tri(N)
            nantri[nantri==0]=np.nan
            mat_mean = np.nanmean(mat*nantri,axis=1)
            mat_rmse = np.power(mat_mean,0.5)
            rmse_n[site] = np.divide(mat_rmse,rmse_tot[site].values)

        rmse_n.index = rmse_n.index/N
        plt.plot(rmse_n, color = colors[color_num],alpha = 0.5)
        color_num = color_num+1

    plt.xlabel('n/N')
    plt.xscale('log')
    plt.ylabel('RMSE_Cumulative/RMSE_total')
    plt.yscale('log')

    if multi == True:
        plt.title('RMSE Distribution in Descending Sort')
    else:
        plt.title(f'RMSE Distribution in Descending Sort: {site}')
    plt.axhline(1)
    plt.legend(handles=custom_legend(sim_keys,colors))

def find_upstream(topo: xr.Dataset, segID: int,return_segs: list=[]):
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

def find_all_upstream(topo: xr.Dataset, segID: int, return_segs: list=[]) -> np.ndarray:
    upsegs = np.argwhere((topo['Tosegment'] == segID).values).flatten()
    upsegIDs = topo['seg_id'][upsegs].values
    return_segs += list(upsegs)
    for upsegID in upsegIDs:
        find_all_upstream(topo, upsegID, return_segs=return_segs)
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
    adj_mat = np.zeros(shape=(N,N),dtype=int)

    #builds adjacency matrix based on what segements are upstream
    i = 0
    for ID in topo['seg_id'].values:
        adj = list()
        find_upstream(topo,ID,adj)
        for dex in adj:
            #print(i,dex)
            adj_mat[i,dex] += 1
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
    pos = nx.drawing.nx_agraph.graphviz_layout(topo,prog='dot')
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
    color_steps = np.arange(0, 1, 1/len(segs))

    color_dict =  {f'{seg}': mpl.colors.to_hex(cmap(i)) for i, seg in zip(color_steps, segs)}
    return color_dict

def color_code_nxgraph(graph: nx.graph, measure: pd.Series,
                       cmap=mpl.cm.get_cmap('coolwarm_r'))-> dict:
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
    if np.where(measure<0)[0].size == 0:
        # meaning we have only positive values and do not need to establish
        # zero at the center of the color bar
        segs = measure.index
        minimum = 0 #set to zero to preserve the coloring of the scale
        maximum = measure.max()

        color_vals = (measure.values)/(maximum)
        color_bar = plt.cm.ScalarMappable(cmap=cmap, norm = plt.Normalize(vmin = minimum, vmax = maximum))

        color_dict =  {f'{seg}': mpl.colors.to_hex(cmap(i)) for i, seg in zip(color_vals, segs)}
        return color_dict, color_bar
    else:
        #determine colorbar range
        extreme = abs(measure.max())
        if np.abs(measure.min()) > extreme:
            extreme = np.abs(measure.min())


        #sets up color values
        segs = measure.index
        color_vals = (measure.values+extreme)/(2*extreme)
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
    topo_color_dict, topo_color_cbar = color_code_nxgraph(topo_graph,color_measure,cmap)
    topo_nodecolors = [topo_color_dict[f'{node}'] for node in topo_graph.nodes()]
    nx.draw_networkx(topo_graph,topo_positions,node_size = 200, font_size = 8, font_weight = 'bold',
                     node_shape = 's', linewidths = 2, font_color = 'w', node_color = topo_nodecolors)
    plt.colorbar(topo_color_cbar)

def plot_reduced_doy_flows(flow_dataset:xr.Dataset, gauge_sites:list, 
                        reduce_func=np.mean, vertical_label=f'Mean Day of Year Flow 'r'$(m^3/s)$',
                        title_label=f'Annual Mean Flows (WY1952 to WY2007)',
                        raw_var='raw_flow', ref_var = 'reference_flow', bc_var = 'bias_corrected_total_flow',
                        raw_name = 'Mizuroute Raw', ref_name = 'NRNI Reference', bc_name = 'BMORPH BC',
                        transpose_raw = True, transpose_ref = False, transpose_bc = True,
                        fontsize_title=80, fontsize_legend=68, fontsize_subplot=60, 
                        fontsize_tick = 45, fontcolor = 'black'):
    """
    Plot Mean Day of Year Flows
        creates a series of subplots that plot an average year's flows
        per gauge site
    ---
    flow_dataset: xr.Dataset
        contatains raw, reference, and bias corrected flows
    gauges_sites: list
        a list of gauge sites to be plotted, cannot exceed 12
    reduce_func: function
        a function to apply to flows grouped by dayofyear,
        default = np.mean
    vertical_label: str
        a string label for the vertical axis representing
        the reduce_func, defaults as:
        f'Mean Day of Year Flow 'r'$(m^3/s)$' to fit np.mean
    raw_var: str
        the string to access the raw flows in flow_dataset
    ref_var: str
        the string to access the reference flows in flow_dataset
    bc_var: str
        the string to access the bias corrected flows in flow_dataset
    transpose_*: boolean
        does this flow Dataset need to be transposed to fit? (no need to
        change unless prompted by error)
    """
    
    raw_flow_yak_doy = flow_dataset[raw_var].groupby(flow_dataset['time'].dt.dayofyear).reduce(reduce_func)
    reference_flow_yak_doy = flow_dataset[ref_var].groupby(flow_dataset['time'].dt.dayofyear).reduce(reduce_func)
    bc_flow_yak_doy = flow_dataset[bc_var].groupby(flow_dataset['index'].dt.dayofyear).reduce(reduce_func)

    doy = raw_flow_yak_doy['dayofyear'].values
    outlet_names = flow_dataset['outlet'].values
    
    if transpose_raw:
        raw_flow_doy_df = pd.DataFrame(data=np.transpose(raw_flow_yak_doy.values), index=doy,columns=outlet_names)
    else:
        raw_flow_doy_df = pd.DataFrame(data=raw_flow_yak_doy.values, index=doy,columns=outlet_names)
    if transpose_ref:
        reference_flow_doy_df = pd.DataFrame(data=np.transpose(reference_flow_yak_doy.values), index=doy,columns=outlet_names)
    else:
        reference_flow_doy_df = pd.DataFrame(data=reference_flow_yak_doy.values, index=doy,columns=outlet_names)
    if transpose_bc:
        bc_flow_doy_df = pd.DataFrame(data=np.transpose(bc_flow_yak_doy.values), index=doy,columns=outlet_names)
    else:
        bc_flow_doy_df = pd.DataFrame(data=bc_flow_yak_doy.values, index=doy,columns=outlet_names)
    
    mpl.rcParams['figure.figsize'] = (70,30)
    n_rows, n_cols = determine_row_col(len(gauge_sites))
    fig, axs = plt.subplots(n_rows,n_cols)

    fig.suptitle(title_label, fontsize = fontsize_title, color=fontcolor, y=1.05)

    for site, ax in zip(gauge_sites, axs.ravel()):
        ax.plot(raw_flow_doy_df[site],color='grey', alpha = 0.8, lw=4)
        ax.plot(reference_flow_doy_df[site],color='black', alpha = 0.8, lw=4)
        ax.plot(bc_flow_doy_df[site],color='red', lw=4)
        ax.set_title(site, fontsize=fontsize_subplot, color=fontcolor)
        plt.setp(ax.spines.values(),color=fontcolor)
        ax.tick_params(axis='both',colors=fontcolor,labelsize=fontsize_tick)
        
    if len(gauge_sites) < n_rows*n_cols:
        for ax_index in np.arange(len(gauge_sites),n_rows*n_cols):
            axs.ravel().tolist()[ax_index].axis('off')

    fig.text(0.5, -0.02, 'Day of Year', fontsize=fontsize_title, ha='center')
    fig.text(-0.02, 0.5, vertical_label, 
             fontsize=fontsize_title, va='center',rotation='vertical')
    plt.subplots_adjust(wspace=0.2, hspace= 0.5, left = 0.05, right = 0.8, top = 0.95)

    fig.legend([raw_name, ref_name, bc_name],fontsize=fontsize_legend, loc='center right');
    
def plot_spearman_rank_difference(flow_dataset:xr.Dataset, gauge_sites:list, start_year:str, end_year:str, 
                                  relative_locations_triu: pd.DataFrame, basin_map_png, 
                                  cmap=mpl.cm.get_cmap('coolwarm_r'), blank_plot_color='white', fontcolor='black',
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
    cmap.set_under(color=blank_plot_color)

    fig, ax = plt.subplots()
    
    if int(end_year[:4])-int(start_year[:4]) == 1:
        fig.suptitle(f'WY{start_year[:4]}: 'r'$r_{s}(Q_{raw}) - r_{s}(Q_{bc})$', 
                     fontsize=fontsize_title, color=fontcolor,x=0.6,y=0.95)
    elif int(end_year[:4])-int(start_year[:4]) > 1:
        end_WY = int(end_year[:4])-1
        fig.suptitle(f'WY{start_year[:4]} to WY{end_WY}: 'r'$r_{s}(Q_{raw}) - r_{s}(Q_{bc})$', 
                     fontsize=fontsize_title, color=fontcolor,x=0.6,y=0.95)
    else:
        raise Exception('Please check end_year is later than start_year')
        
    time_span = flow_dataset['time'].values
    outlet_names = flow_dataset['outlet'].values
        
    raw_flow = pd.DataFrame(data=np.transpose(flow_dataset['raw_flow'].values), 
                            index=time_span, columns=outlet_names)
    ref_flow = pd.DataFrame(data=flow_dataset['reference_flow'].values, 
                            index=time_span,columns=outlet_names)
    bc_flow = pd.DataFrame(data=np.transpose(flow_dataset['bias_corrected_total_flow'].values), 
                           index=time_span, columns=outlet_names)

    raw_flow_spearman = filter_rank_corr(raw_flow.loc[start_year:end_year].corr(method='spearman'), 
                                         rel_loc=relative_locations_triu)
    bc_flow_spearman = filter_rank_corr(bc_flow.loc[start_year:end_year].corr(method='spearman'), 
                                        rel_loc=relative_locations_triu)
    raw_minus_bc_spearman = raw_flow_spearman - bc_flow_spearman

    vmin = np.min(raw_minus_bc_spearman.min())
    vmax = np.max(raw_minus_bc_spearman.max())
    vextreme = vmax
    if np.abs(vmin) > vextreme:
        vextreme = np.abs(vmin)
        
    # -10 is used to ensure that the squares not to be plotted are marked as such by 
    # the cmap's under values
    vunder = np.abs(vmin)*-10

    im = ax.imshow(raw_minus_bc_spearman.fillna(vunder), vmin=-vextreme, vmax=vextreme, cmap=cmap)
    plt.setp(ax.spines.values(),color=fontcolor)
    ax.tick_params(axis='both',colors=fontcolor)
    tick_tot = len(gauge_sites)
    ax.set_xticks(np.arange(0,tick_tot,1.0))
    ax.set_yticks(np.arange(0,tick_tot,1.0))
    ax.set_ylim(tick_tot-0.5,-0.5)
    ax.set_xticklabels(gauge_sites,rotation='vertical',fontsize=fontsize_label)
    ax.set_yticklabels(gauge_sites,fontsize=fontsize_tick);

    cb = fig.colorbar(im, pad=0.01)
    cb.ax.yaxis.set_tick_params(color=fontcolor, labelcolor=fontcolor,labelsize=fontsize_tick)
    cb.outline.set_edgecolor(None)

    fig.text(0.6, -0.02, 'Site Abbreviation', fontsize=fontsize_label, ha='center')
    fig.text(0.32, 0.4, 'Site Abbreviation', fontsize=fontsize_label, va='center',rotation='vertical')

    newax = fig.add_axes([0.295, 0.4, 0.49, 0.49], anchor = 'NE', zorder = 2)
    newax.imshow(big_map_yak)
    newax.axis('off')
    newax.tick_params(axis='both')
    newax.set_xticks([])
    newax.set_yticks([])

    plt.tight_layout
    plt.show()
    
def determine_row_col(n:int, pref_rows = True):
    """
    Determine Rows and Columns
        calculates a rectangular subplot layout that
        contains at least n subplots, some may need to
        be turned off in plotting
    ----
    n: int
        total number of plots
    pref_rows: boolean
        If True, and only a rectangular arrangment
        is possible, then put the longer dimension
        in n_rows. If False, then it is
        placed in the n_columns
    return: int, int
        n_rows, n_columns
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
    
def correction_scatter(site_dict: dict, raw_flow: pd.DataFrame, ref_flow: pd.DataFrame, bc_flow: pd.DataFrame, 
                       colors: list, title= 'Flow Residuals', fontsize_title=80, fontsize_legend=68, 
                       fontsize_subplot=60, fontsize_tick = 45, fontcolor = 'black'):
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
        group_before_bc = before_bc.loc[:,site_group]
        group_after_bc = after_bc.loc[:,site_group]
        scatter_series_axes(group_before_bc,group_after_bc,label=site_group_key,
                                     alpha=0.05, color=colors[i],ax=ax_list[i])
        ax_list[i].set_title(site_group_key, fontsize=fontsize_subplot)
        plt.setp(ax_list[i].spines.values(),color=fontcolor)
        ax_list[i].tick_params(axis='both', colors=fontcolor, labelsize=fontsize_tick)
        
    # add 1:1 analysis line and hide plots that are not in use 
    for i, ax in enumerate(axs.ravel()):
        if i < num_plots:
            bottom, top = ax.get_ylim()
            left, right = ax.get_xlim()
            ref_line_max = np.max([bottom, top, left, right])
            ax.plot([0,ref_line_max], [0,ref_line_max], color='k', linestyle='--')
        else:
            ax.axis('off')
    
    fig.text(0.5, -0.04, r'$Q_{ref} - Q_{raw} \quad (m^3/s)$', ha='center', va = 'bottom', fontsize=fontsize_title);
    fig.text(-0.02, 0.5, r'$Q_{ref} - Q_{bc} \quad (m^3/s)$', va='center', rotation = 'vertical', fontsize=fontsize_title);    
    
    plt.tight_layout()
    plt.show