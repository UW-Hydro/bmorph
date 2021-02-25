Input Specifications **(IN PROGRESS)**
======================================

Directory Setup
---------------

    input
    mizuroute_configs
    notebooks
    output
    topologies
    
Common Naming Conventions
-------------------------

`seg` is an individual river segment containing a single reach
`hru` is a hydrologic residence unit that feeds into a single seg, but each seg could have multiple hru's feeding into it
`seg_id` is the identification number for a `seg`
`site` is the gauge site name for river segments with gauge data, not all segments have them



Mizuroute
---------

configuration
^^^^^^^^^^^^^

should look like this ....

.. code:: ipython3

    <ancil_dir>         $ancil_dir !
    <input_dir>            $input_dir !
    <output_dir>           $output_dir !
    <sim_start>            $sim_start !
    <sim_end>              $sim_end !
    <fname_ntopOld>        $topo_file !
    <dname_nhru>           seg !
    <dname_sseg>           seg !
    <seg_outlet>           -9999 !
    <fname_qsim>           $flow_file !
    <vname_qsim>           scbc_flow !
    <vname_time>           time !
    <dname_time>           time !
    <dname_hruid>          seg !
    <vname_hruid>          seg !
    <units_qsim>           mm/d !
    <dt_qsim>              86400 !
    <is_remap>              F !
    <restart_opt>           F !
    <route_opt>             1 !
    <fname_output>          $out_name !
    <fname_state_out>       state.out.nc !
    <param_nml>             param.nml.default !
    <doesBasinRoute>        0 !
    <varname_area>          Contrib_Area !
    <varname_length>        Length !
    <varname_slope>         Slope !
    <varname_HRUid>         seg_id !
    <varname_segId>         seg_id !
    <varname_downSegId>     Tosegment !
    <varname_hruSegId>      seg_id !
    
    
Utilities
---------

mizuroute_to_blendmorph
^^^^^^^^^^^^^^^^^^^^^^^



reference site selection & cdf blend factor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^




Output Specifications
=====================

Rerouting Local Corrected Flows
-------------------------------


Citations
=========

.. [Ref] Mizukami, M. Clark, M. P., Sampson, K., Nijssen, B., Mao, Y., McMillan, H., Viger, R. J., Markstrom, S. L., Hay, L. E., Woods, R., Arnold, J. R., & Brekke, L. D. (2016). mizuRoute version 1: a river network routing tool for a continental domain water resources applications. *Geoscientific Model Development, 9*, 2223-2238. www.geosci-model-dev.net/9/2223/2016/doi:10.5194/gmd-9-2223-2016

