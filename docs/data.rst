Input Specifications
====================

Directory Setup
---------------

The following directories and names are  required to properly execute ``bmorph``:

    input
    mizuroute_configs
    notebooks
    output
    topologies
    
Common Naming Conventions
-------------------------

    **seg** is an individual river segment containing a single reach
    **hru** is a hydrologic response unit that feeds into a single seg, 
    but each seg could have multiple hru's feeding into it
    **seg_id** is the identification number for a `seg`
    **site** is the gauge site name for river segments with gauge data, not all segments have them


Mizuroute
---------

``bmorph`` is designed to bias correct simulated streamflow as modeled by mizuroute_. Following
is how to configure files for ``mizuroute`` in accoradance with the software.

.. _mizuroute: https://mizuroute.readthedocs.io/en/latest/

Configuration
^^^^^^^^^^^^^

Configuring files for ``mizuroute`` through ``bmorph`` should follow the following template:

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
    
Note that ``mizuroute_utils.write_mizuroute_config`` provides a template that can be used to write a valid mizuroute configuration as described above.
    
Utilities
---------

mizuroute_to_blendmorph
^^^^^^^^^^^^^^^^^^^^^^^

``mizuroute_to_blendmorph`` handles formatting the output of ``mizuroute`` for ``bmorph``. While configuring
streamflows can be performed without this function, this helps to speed up the whole workflow with a number of 
customizable options. As detailed in `bmorph_tutorial.rst <bmorph_tutorial.rst>`_, ``mizuroute_to_blendmorph`` 
is the primary function for handling streamflow formatting.

Spatial Consistency: Reference Site Selection & CDF Blend Factor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Spatial consistency is conserved by combining streamflows that are ``bmorph`` bias corrected with respect to
upstream and downstream reprentative sites. Ideally, if a seg has a gauge site directly upstream and downstream
of it, then a reference for that seg can be interpolated as a combination of those two gauge sites. Now because
there are not gauge sites everywhere, (which would render this method unncessary), the gauge sites used as the 
upstream/downstream need to be selected, this is where ``fill_method`` comes into play in ``mizuroute_to_blendmorph``. 
Segs that are gauge sites are simply assigned themselves as their upstream/downstream segments. Looking downstream can
typically yeild a gauge site as rivers do not typically branch out in the direction of flow. Looking upstream for a 
gauge site gets more complicated as a one:many relationship occurs. Hence, needing to "fill" in gauge sites that are
not simply found. There are a few different means of doing this: leaving the sites empty (``leave_null``), using xarray's
forward_fill_, or selecting based on different statistical measures of simularity (``r2``, ``kldiv``, ``kge``). 

.. image:: Figures/Blending_Diagram.png
    :alt: In blending, attributes from one gauge site are mixed with another gauge site depending on how close the intermediate seg is to each gauge site, (depicted left by 5 circles translating from pink to purple to blue across the segs). As a result, intermediate CDFs can be produced by transitioning from one gauge site CDF to another, (depicted right by pink CDF curves transforming into purple then blue CDFs curves).

Blend factor describes how upstream and downstream flows should be combined, or "blended" together.
Let

    UM, DM = Upstream Measure, Downstream Measure (length, r2, Kullback-Leibler Divergence, or Kling-Gupta Efficiency)    
    BF = Blend Factor    
    UF, DF, TF = Upstream Corrected Flow, Downstream Corrected Flow, Total Corrected Flow    

.. math:: 

    BF = \frac{UM}{UM+DM}
    TF = (BF*UF) + ((1-BF)*DF)
    

.. _forward_fill: http://xarray.pydata.org/en/stable/generated/xarray.DataArray.ffill.html

Output Specifications
=====================

Rerouting Local Corrected Flows
-------------------------------

``bmorph`` applies corrections to total flows, so rerouting ``bmorph`` outputs through ``mizuroute`` to retrieve local flows.

