``bmorph`` Example Workflow
===========================

This notebook demonstrates how to setup data for **bmorph**.

Import Packages and Load Data
-----------------------------

``mizuroute_utils`` is our utility script that will handle converting
Mizuroute outputs to what we need for ``bmorph``.

.. code:: ipython3

    %pylab inline
    %load_ext autoreload
    %autoreload 2
    %reload_ext autoreload
    import xarray as xr
    import pandas as pd
    import bmorph
    from bmorph.util import mizuroute_utils as mizutil
    from dask.distributed import Client, progress
    
    xr.set_options(display_style='html') # this is just to make viewing our data nicer


.. parsed-literal::

    Populating the interactive namespace from numpy and matplotlib




.. parsed-literal::

    <xarray.core.options.set_options at 0x7f71c1a7cb10>



.. code:: ipython3

    client = Client(threads_per_worker=1, n_workers=1)
    client


.. parsed-literal::

    /pool0/home/steinjao/.conda/envs/project/lib/python3.7/site-packages/distributed/dashboard/core.py:72: UserWarning: 
    Port 8787 is already in use. 
    Perhaps you already have a cluster running?
    Hosting the diagnostics dashboard on a random port instead.
      warnings.warn("\n" + msg)




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

    # providing the gauge site names and their respective river segment identification
    # numbers, or sites and segs
    site_to_seg = {'KEE' : 4175, 'KAC' : 4171, 'EASW': 4170,
                   'CLE' : 4164, 'YUMW': 4162, 'BUM' : 5231,
                   'AMRW': 5228,  'CLFW': 5224,  'RIM' : 5240,
                   'NACW': 5222, 'UMTW': 4139,  'AUGW': 594,
                   'PARW': 588,   'YGVW': 584,   'KIOW': 581, }
    
    # and creating some other useful forms of these gauge sites for later use
    seg_to_site = {seg: site for site, seg in site_to_seg.items()}
    ref_sites = list(site_to_seg.keys())
    ref_segs = list(site_to_seg.values())
    
    # loading in topographical data (topo), meterological data (met), 
    # uncorrected flows (raw) and reference flows (ref)
    yakima_topo = xr.open_dataset('../topologies/yakima_huc12_topology.nc').load()
    
    # the Yakima River Basin is part of the Columbia River Basin, so loading
    # data for the Columbia River Basin can later be subset for the Yakima
    columbia_met = xr.open_dataset('../input/tmin.nc').load()
    columbia_met['seasonal_precip'] = xr.open_dataset('../input/prec.nc')['prec'].load().rolling(time=30, min_periods=1).sum()
    columbia_met['tmax'] = xr.open_dataset('../input/tmax.nc')['tmax'].load()
    columbia_met['hru'] = (columbia_met['hru'] - 1.7e7).astype(np.int32)
    columbia_raw = xr.open_mfdataset('../input/first_route*.nc')[['IRFroutedRunoff', 'dlayRunoff', 'reachID']].load()
    columbia_raw['seg'] = columbia_raw.isel(time=0)['reachID'].astype(np.int32)
    
    columbia_ref = xr.open_dataset('../input/nrni_reference_flows.nc').load().rename({'outlet':'site'})[['seg', 'seg_id', 'reference_flow']]
    
    columbia_topo = xr.open_dataset('../topologies/columbia_huc12_topology.nc').load()
    columbia_topo = columbia_topo.where(columbia_topo['hru'] < 1.79e7, drop=True)
    
    # and cleaning up a few naming conventions
    if 'hru_id2' in yakima_topo:
        yakima_topo['hru'] = yakima_topo['hru_id2']
    if 'seg_id' in yakima_topo:
        yakima_topo['seg'] = yakima_topo['seg_id']


.. parsed-literal::

    /pool0/home/steinjao/.conda/envs/project/lib/python3.7/site-packages/ipykernel_launcher.py:24: FutureWarning: In xarray version 0.15 the default behaviour of `open_mfdataset`
    will change. To retain the existing behavior, pass
    combine='nested'. To use future default behavior, pass
    combine='by_coords'. See
    http://xarray.pydata.org/en/stable/combining.html#combining-multi
    
    /pool0/home/steinjao/.conda/envs/project/lib/python3.7/site-packages/xarray/backends/api.py:933: FutureWarning: The datasets supplied have global dimension coordinates. You may want
    to use the new `combine_by_coords` function (or the
    `combine='by_coords'` option to `open_mfdataset`) to order the datasets
    before concatenation. Alternatively, to continue concatenating based
    on the order the datasets are supplied in future, please use the new
    `combine_nested` function (or the `combine='nested'` option to
    open_mfdataset).
      from_openmfds=True,


Convert ``mizuroute`` formatting to ``bmorph`` formatting
---------------------------------------------------------

.. code:: ipython3

    yakima_ref = columbia_ref.sel(site=[r for r in ref_sites])
    
    for site, seg in site_to_seg.items():
        if site in yakima_ref['site']:
            yakima_ref['seg'].loc[{'site': site}] = seg
    
    # `mizuroute_to_blendmorph` is a utility function that automates
    # the preprocessing for bmorph
    yakima_met_seg = mizutil.mizuroute_to_blendmorph(
        yakima_topo, columbia_raw.copy(), yakima_ref, columbia_met, 
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
        raw_ts = yakima_met_seg.sel(seg=seg)['IRFroutedRunoff'].to_series()
        train_ts = yakima_met_seg.sel(seg=seg)['IRFroutedRunoff'].to_series()
        obs_ts = yakima_met_seg.sel(seg=seg)['up_ref_flow'].to_series()
        cond_var = yakima_met_seg.sel(seg=seg)[f'up_{condition_var}'].to_series()
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
    region = 'yakima'
    
    unconditioned_totals = bmorph.workflows.run_parallel_scbc(yakima_met_seg, client, region, mizuroute_exe, univariate_config)
    conditioned_totals = bmorph.workflows.run_parallel_scbc(yakima_met_seg, client, region, mizuroute_exe, conditonal_config)
    for site, seg in site_to_seg.items():
        unconditioned_totals[site] = unconditioned_totals['IRFroutedRunoff'].sel(seg=seg)
        conditioned_totals[site] = conditioned_totals['IRFroutedRunoff'].sel(seg=seg)


.. parsed-literal::

    /pool0/home/steinjao/.conda/envs/project/lib/python3.7/site-packages/distributed/worker.py:3285: UserWarning: Large object of size 2.28 MB detected in task graph: 
      (<xarray.Dataset>
    Dimensions:               (time: ... .. 13.49 13.7,)
    Consider scattering large objects ahead of time
    with client.scatter to reduce scheduler burden and 
    keep data on workers
    
        future = client.submit(func, big_data)    # bad
    
        big_future = client.scatter(big_data)     # good
        future = client.submit(func, big_future)  # good
      % (format_bytes(len(b)), s)


::


    ---------------------------------------------------------------------------

    OSError                                   Traceback (most recent call last)

    <ipython-input-7-03d1a2fe6003> in <module>
          5 region = 'yakima'
          6 
    ----> 7 unconditioned_totals = bmorph.workflows.run_parallel_scbc(yakima_met_seg, client, region, mizuroute_exe, univariate_config)
          8 conditioned_totals = bmorph.workflows.run_parallel_scbc(yakima_met_seg, client, region, mizuroute_exe, conditonal_config)
          9 for site, seg in site_to_seg.items():


    /pool0/data/steinjao/bmorph/bmorph/core/workflows.py in run_parallel_scbc(ds, client, region, mizuroute_exe, bmorph_config)
        674     config_path, mizuroute_config = mizutil.write_mizuroute_config(region, scbc_type, bmorph_config['bmorph_window'])
        675     mizutil.run_mizuroute(mizuroute_exe, config_path)
    --> 676     region_totals = xr.open_mfdataset(f'{mizuroute_config["output_dir"]}{region.lower()}_{scbc_type}_scbc*').load()
        677     region_totals = region_totals.sel(time=slice(*bmorph_config['bmorph_window']))
        678     region_totals['seg'] = region_totals['reachID'].isel(time=0)


    ~/.conda/envs/project/lib/python3.7/site-packages/xarray/backends/api.py in open_mfdataset(paths, chunks, concat_dim, compat, preprocess, engine, lock, data_vars, coords, combine, autoclose, parallel, join, **kwargs)
        868 
        869     if not paths:
    --> 870         raise OSError("no files to open")
        871 
        872     # If combine='by_coords' then this is unnecessary, but quick.


    OSError: no files to open


.. code:: ipython3

    scbc_c = bmorph.workflows.bmorph_to_dataarray(conditioned_totals, 'scbc_c')
    yakima_analysis = xr.Dataset(coords={'site': list(site_to_seg.keys()), 'time': scbc_c['time']})
    yakima_analysis['scbc_c'] = scbc_c
    yakima_analysis['scbc_u'] = bmorph.workflows.bmorph_to_dataarray(unconditioned_totas, 'scbc_u')
    yakima_analysis['ibc_u'] = bmorph.workflows.bmorph_to_dataarray(ibc_u_flows, 'ibc_u')
    yakima_analysis['ibc_c'] = bmorph.workflows.bmorph_to_dataarray(ibc_c_flows, 'ibc_c')
    yakima_analysis['raw'] = bmorph.workflows.bmorph_to_dataarray(raw_flows, 'raw')
    yakima_analysis['ref'] = bmorph.workflows.bmorph_to_dataarray(ref_flows, 'ref')
    yakima_analysis.to_netcdf(f'../output/{region.lower()}_data_processed.nc')

.. code:: ipython3

    scbc_type = 'univariate'

.. code:: ipython3

    config_path, mizuroute_config = mizutil.write_mizuroute_config(region, scbc_type, univariate_config['bmorph_window'])

.. code:: ipython3

    config_path




.. parsed-literal::

    '/pool0/data/steinjao/bmorph/docs/example/mizuroute_configs/reroute_yakima_univariate.control'



.. code:: ipython3

    mizuroute_config




.. parsed-literal::

    {'ancil_dir': '/pool0/data/steinjao/bmorph/docs/example/topologies/',
     'input_dir': '/pool0/data/steinjao/bmorph/docs/example/input/',
     'output_dir': '/pool0/data/steinjao/bmorph/docs/example/output/',
     'sim_start': '1991-01-01',
     'sim_end': '2005-12-30',
     'topo_file': 'yakima_huc12_topology_scaled_area.nc',
     'flow_file': 'yakima_local_univariate_scbc.nc',
     'out_name': 'yakima_univariate_scbc'}



.. code:: ipython3

    mizutil.run_mizuroute(mizuroute_exe, config_path)




.. parsed-literal::

    <subprocess.Popen at 0x7f6ff61d8e10>



.. code:: ipython3

    region_totals = xr.open_mfdataset(f'{mizuroute_config["output_dir"]}{region.lower()}_{scbc_type}_scbc*').load()


::


    ---------------------------------------------------------------------------

    OSError                                   Traceback (most recent call last)

    <ipython-input-16-50220390808c> in <module>
    ----> 1 region_totals = xr.open_mfdataset(f'{mizuroute_config["output_dir"]}{region.lower()}_{scbc_type}_scbc*').load()
    

    ~/.conda/envs/project/lib/python3.7/site-packages/xarray/backends/api.py in open_mfdataset(paths, chunks, concat_dim, compat, preprocess, engine, lock, data_vars, coords, combine, autoclose, parallel, join, **kwargs)
        868 
        869     if not paths:
    --> 870         raise OSError("no files to open")
        871 
        872     # If combine='by_coords' then this is unnecessary, but quick.


    OSError: no files to open


.. code:: ipython3

    f'{mizuroute_config["output_dir"]}{region.lower()}_{scbc_type}_scbc*'




.. parsed-literal::

    '/pool0/data/steinjao/bmorph/docs/example/output/yakima_univariate_scbc*'



