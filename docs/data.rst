Data Overview
=============

Directory Setup
---------------

The following directories and names are required to properly execute ``bmorph``:

|    input
|    mizuroute_configs
|    notebooks
|    output
|    topologies
    
Variable Naming Conventions
---------------------------

|    **seg** is an individual river segment containing a single reach
|    **hru** is a hydrologic response unit that feeds into a single seg, 
    but each seg could have multiple hru's feeding into it
|    **seg_id** is the identification number for a `seg`
|    **site** is the gauge site name for river segments with gauge data, not all segments have them
|    **raw** refers to uncorrected routed streamflows
|    **ref** refers to references streamflow data, such as that from a gauge site


Configuration Utilities
-----------------------
 
to mizuRoute
^^^^^^^^^^^^

``bmorph`` is designed to bias correct simulated streamflow as modeled by `mizuroute <https://mizuroute.readthedocs.io/en/latest/>`_.  `bmorph.util.mizuroute_utils.write_mizuroute_config <https://bmorph.readthedocs.io/en/develop/api.html#bmorph.util.mizuroute_utils.write_mizuroute_config>` automates writing a valid mizuroute configuration file for the purposes of ``bmorph``. 

Running ``mizuRoute`` during bias correction is only neccessary for spatially consistent methods were local flows need to be routed. As a result, writing the configuration file is automated through only the `bmorph.core.workflows.run_parallel_scbc <https://bmorph.readthedocs.io/en/develop/api.html#bmorph.core.workflows.run_parallel_scbc>`_. The configuration script recieves the name of the region, the type of bias correction, and the time window from ``run_parallel_scbc``, writing the rest of the configuration file assuming the `Directory Setup <Directory Setup>`_ described above.


to bmorph
^^^^^^^^^

`bmorph.utils.mizuroute_utils.mizuroute_to_blendmorph <https://bmorph.readthedocs.io/en/develop/api.html#bmorph.util.mizuroute_utils.mizuroute_to_blendmorph>` handles formatting the output of ``mizuroute`` for ``bmorph``. While configuring streamflows can be performed without this function, this helps to speed up the whole workflow with a number of customizable options. As detailed in the `tutorial <bmorph_tutorial.rst>`_, ``bmorph.utils.mizuroute_utils.mizuroute_to_blendmorph`` is the primary function for handling streamflow formatting:

.. code:: ipython3
    
    from bmorph.util import mizuroute_utils as mizutil
    
    basin_met_seg = mizutil.mizuroute_to_blendmorph(
        topo = basin_topo, #topology file for basin
        routed = watershed_raw.copy(), #uncorrected streamflows
        reference = basin_ref, #reference streamflows
        met_hru = watershed_met, #meteorologic variables
        route_var = "IRFroutedRunoff", #name of uncorrected flows
        fill_method = 'r2', #
    ).ffil(dim='seg')
    
Where ``topo`` is the topology file for the basin, ``routed`` are the uncorrected streamflows, ``reference`` are the reference streamflows from gauge sites, ``met_hru`` are the meteologic variables used in conditioning, ``route_var`` is the name of the uncorrected flows in ``routed``, and ``fill_method`` describes spatial consistency determining as described in `Spatial Consistency <bias_correction.rst/Spatial Consistency: Reference Site Selection & CDF Blend Factor>`_.

Input Specifications
====================

Input data will be need identical time indices as ``pandas.Series`` without null flow values. So long as the size of the ``pandas.Series`` are the same across all flow data, the magnitude of the length should not impact bias correction. Futher information on non-flow parameters can be found in `Implementation <bias_corrections.rst/Implementation>`_.

Output Specifications
=====================

``bmorph`` outputs a ``pandas.Series`` time series with flows as values indexed by time entires provided by given data. Total lenght of the output is the number of flow values provided as ``float32``. 

The `tutorial <bmorph_tutorial.rst>`_ stores each of these outputs in a dictionary with their site/seg being their corresponding keys. `bmoprh.workflows.bmorph_to_datarray <https://bmorph.readthedocs.io/en/develop/api.html#bmorph.core.workflows.bmorph_to_dataarray>`_ converts such a dictionary into an ``xarray.DataArray`` with coordinates ``site`` and ``time``, corresponding the the dictionary keys and the time of the ``pandas.Series`` that they access. From there, uncorrected and reference flows can be combined with the corrected flows into a singular ``xarray.DataArray`` and saved into a ``netCDF`` file if desired, storing it in the ``output`` directory if following the tutorial.

