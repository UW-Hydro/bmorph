``bmorph`` Example Workflow
===========================

This notebook demonstrates how to setup data for and bias correct it through **bmorph**.

Import Packages and Load Data
-----------------------------

.. code:: ipython3
    
    # We will be using xarray and pandas in this example notebook
    # in addition to numpy, (which you could import directly intsead of using magic).
    %pylab inline
    import xarray as xr
    import pandas as pd
    
    """
    We will mainly deal with bmorph.workflows, but it is still
    helpful to distinguish bmorph.mizuroute_utils in specific because
    it is purely a data processing script.
    """
    import bmorph
    from bmorph.util import mizuroute_utils as mizutil
    
    """
    Setting up a client for parallelism can help spead up the process
    of bias correction immensely, espeically if working with large or numerous
    watersheds, which is useful if still calibrating which meterological variable
    you want to condition to, (or just saving time in general).
    """
    from dask.distributed import Client, progress
    

.. code:: ipython3
    
    """
    In case you are just copying this over, the client is only set up with
    one thread and one worker to prevent accidentally overburdening any
    machine this is running on. If you actually want to use parallelism, 
    be sure to change this!
    """
    client = Client(threads_per_worker=1, n_workers=1) #Increase for parallel power!!!
    client


.. raw:: html

    <table style="border: 2px solid white;">
    <tr>
    <td style="vertical-align: top; border: 0px solid white">
    <h3 style="text-align: left;">Client</h3>
    <ul style="text-align: left; list-style: none; margin: 0; padding: 0;">
      <li><b>Scheduler: </b>tcp://127.0.0.1:40411</li>
      <li><b>Dashboard: </b><a href='http://127.0.0.1:39355/status' target='_blank'>http://127.0.0.1:39355/status</a>
    </ul>
    </td>
    <td style="vertical-align: top; border: 0px solid white">
    <h3 style="text-align: left;">Cluster</h3>
    <ul style="text-align: left; list-style:none; margin: 0; padding: 0;">
      <li><b>Workers: </b>4</li>
      <li><b>Cores: </b>4</li>
      <li><b>Memory: </b>540.66 GB</li>
    </ul>
    </td>
    </tr>
    </table>



.. code:: ipython3

    # Here you provide the gauge site names and their respective river segment identification
    # numbers, or sites and segs. This will be used throughout to ensure there isn't data mismatch.
    site_to_seg = { site_0_name : site_0_seg, ...} # Input this mapping or read it from a text file before running!
    
    # Since it's always nice to be able to access the data you just filled out without much struggle, here we create
    # some other useful forms of these gauge sites for later use.
    seg_to_site = {seg: site for site, seg in site_to_seg.items()}
    ref_sites = list(site_to_seg.keys())
    ref_segs = list(site_to_seg.values())
    
    # Here we load in topographical data (topo), meterological data (met), 
    # uncorrected flows (raw) and reference flows (ref).
    basin_topo = xr.open_dataset('../topologies/basin_topology_file_name.nc').load() # loading the data will help speed up things later
    
    """
    Sometimes meterological data may only be available for a larger region
    or watershed than anlayzing, so the following data will be described under such
    an assumption.
    
    Here we load in some example meterological data: minimum temperature (tmin), seasonal precipitation (prec),
    and maximum temperature (tmax). You can use similar or completely different data, just note naming referencing
    should be universally updated.
    """
    watershed_met = xr.open_dataset('../input/tmin.nc').load()
    watershed_met['seasonal_precip'] = xr.open_dataset('../input/prec.nc')['prec'].load().rolling(time=30, min_periods=1).sum()
    watershed_met['tmax'] = xr.open_dataset('../input/tmax.nc')['tmax'].load()
    
    # Hydrualic residence units (hru's) are the typical coordinate for meteorlogical data. Later, mizuroute_utils
    # will take care of mapping these hru's to seg's.
    watershed_met['hru'] = (watershed_met['hru'] - 1.7e7).astype(np.int32)
    
    # And last not be certainly not least, we need the flows themselves! bmorph operates as a post-processing method,
    # meaning a streamflow routing through mizuroute should occur before running all this. As a result, loading
    # up the raw flows involves combining a number of flow netcdf files, hence the open_mfdataset.
    watershed_raw = xr.open_mfdataset('../input/first_route*.nc')[['IRFroutedRunoff', 'dlayRunoff', 'reachID']].load()
    watershed_raw['seg'] = watershed_raw.isel(time=0)['reachID'].astype(np.int32)
    watershed_ref = xr.open_dataset('../input/nrni_reference_flows.nc').load().rename({'outlet':'site'})[['seg', 'seg_id', 'reference_flow']]
    
    # And in order to select data for the basin of analysis from the larger watershed, we 
    # need the topology of the larger watershed as well.
    watershed_topo = xr.open_dataset('../topologies/watershed_topology_file_name.nc').load()
    watershed_topo = watershed_topo.where(watershed_topo['hru'] < 1.79e7, drop=True)
    
    # Here we jsut clean up a few naming conventions to get everything on the same page.
    if 'hru_id2' in basin_topo:
        basin_topo['hru'] = basin_topo['hru_id2']
    if 'seg_id' in basin_topo:
        basin_topo['seg'] = basin_topo['seg_id']




Convert ``mizuroute`` formatting to ``bmorph`` formatting
---------------------------------------------------------

``mizuroute_utils`` is our utility script that will handle converting
Mizuroute outputs to what we need for ``bmorph``. For more information
on what ``mizuroute_utils`` does and how to change its parameters, 
check out ``data.rst``.

.. code:: ipython3

    basin_ref = watershed_ref.sel(site=[r for r in ref_sites])
    
    for site, seg in site_to_seg.items():
        if site in basin_ref['site']:
            basin_ref['seg'].loc[{'site': site}] = seg
    
    # `mizuroute_to_blendmorph` is the primary utility function for automates
    # the preprocessing for bmorph.
    basin_met_seg = mizutil.mizuroute_to_blendmorph(
        basin_topo, watershed_raw.copy(), basin_ref, watershed_met, 
        fill_method='r2').ffill(dim='seg')

Apply ``bmorph`` bias correction
--------------------------------

.. code:: ipython3

    train_window = pd.date_range('1981-01-01', '1990-12-30')[[0, -1]]
    bmorph_window = pd.date_range('1991-01-01', '2005-12-30')[[0, -1]]
    reference_window = train_window
    
    interval = pd.DateOffset(years=1)
    overlap = 90
    #condition_var = 'tmax'
    #condition_var = 'seasonal_precip'
    condition_var = 'tmin'
    
    conditonal_config = {
        'train_window': train_window,
        'bmorph_window': bmorph_window,
        'reference_window': reference_window,
        'bmorph_interval': interval,
        'bmorph_overlap': overlap,
        'condition_var': condition_var
    }
    
    univariate_config = {
        'train_window': train_window,
        'bmorph_window': bmorph_window,
        'reference_window': reference_window,
        'bmorph_interval': interval,
        'bmorph_overlap': overlap,
    }

.. code:: ipython3

    ibc_u_flows = {}
    ibc_u_mults = {}
    ibc_c_flows = {}
    ibc_c_mults = {}
    
    raw_flows = {}
    ref_flows = {}
    
    for site, seg in site_to_seg.items():
        raw_ts = basin_met_seg.sel(seg=seg)['IRFroutedRunoff'].to_series()
        train_ts = basin_met_seg.sel(seg=seg)['IRFroutedRunoff'].to_series()
        obs_ts = basin_met_seg.sel(seg=seg)['up_ref_flow'].to_series()
        cond_var = basin_met_seg.sel(seg=seg)[f'up_{condition_var}'].to_series()
        ref_flows[site] = obs_ts
        raw_flows[site] = raw_ts
        
        ## IBC_U
        ibc_u_flows[site], ibc_u_mults[site] = bmorph.workflows.apply_interval_bmorph(
            raw_ts, train_ts, obs_ts, train_window, bmorph_window, reference_window, interval, overlap)
        
        ## IBC_C
        ibc_c_flows[site], ibc_c_mults[site] = bmorph.workflows.apply_interval_bmorph(
            raw_ts, train_ts, obs_ts, train_window, bmorph_window, reference_window, interval, overlap,
            raw_y=cond_var, train_y=cond_var, obs_y=cond_var)

.. code:: ipython3

    mizuroute_exe = '/pool0/data/steinjao/bmorph/docs/example/mizuroute'
    
    unconditioned_totals = {}
    conditioned_totals = {}
    region = 'basin'
    
    unconditioned_totals = bmorph.workflows.run_parallel_scbc(basin_met_seg, client, region, mizuroute_exe, univariate_config)
    conditioned_totals = bmorph.workflows.run_parallel_scbc(basin_met_seg, client, region, mizuroute_exe, conditonal_config)
    for site, seg in site_to_seg.items():
        unconditioned_totals[site] = unconditioned_totals['IRFroutedRunoff'].sel(seg=seg)
        conditioned_totals[site] = conditioned_totals['IRFroutedRunoff'].sel(seg=seg)



.. code:: ipython3

    scbc_c = bmorph.workflows.bmorph_to_dataarray(conditioned_totals, 'scbc_c')
    basin_analysis = xr.Dataset(coords={'site': list(site_to_seg.keys()), 'time': scbc_c['time']})
    basin_analysis['scbc_c'] = scbc_c
    basin_analysis['scbc_u'] = bmorph.workflows.bmorph_to_dataarray(unconditioned_totas, 'scbc_u')
    basin_analysis['ibc_u'] = bmorph.workflows.bmorph_to_dataarray(ibc_u_flows, 'ibc_u')
    basin_analysis['ibc_c'] = bmorph.workflows.bmorph_to_dataarray(ibc_c_flows, 'ibc_c')
    basin_analysis['raw'] = bmorph.workflows.bmorph_to_dataarray(raw_flows, 'raw')
    basin_analysis['ref'] = bmorph.workflows.bmorph_to_dataarray(ref_flows, 'ref')
    basin_analysis.to_netcdf(f'../output/{region.lower()}_data_processed.nc')

