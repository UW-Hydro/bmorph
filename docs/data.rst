Input Specifications
====================

Directory Setup
---------------

The following directories and names are  required to properly execute ``bmorph``:

|    input
|    mizuroute_configs
|    notebooks
|    output
|    topologies
    
Common Naming Conventions
-------------------------

|    **seg** is an individual river segment containing a single reach
|    **hru** is a hydrologic response unit that feeds into a single seg, 
    but each seg could have multiple hru's feeding into it
|    **seg_id** is the identification number for a `seg`
|    **site** is the gauge site name for river segments with gauge data, not all segments have them


Mizuroute
---------

``bmorph`` is designed to bias correct simulated streamflow as modeled by mizuroute_. Following
is how to configure files for ``mizuroute`` in accoradance with the software.

.. _mizuroute: https://mizuroute.readthedocs.io/en/latest/

Configuration
^^^^^^^^^^^^^
    
`bmorph.util.mizuroute_utils.write_mizuroute_config <https://bmorph.readthedocs.io/en/develop/api.html#bmorph.util.mizuroute_utils.write_mizuroute_config>` automate writing a valid mizuroute configuration file for the purposes of ``bmorph``.
    
Utilities
---------

mizuroute_to_blendmorph
^^^^^^^^^^^^^^^^^^^^^^^

`bmorph.utils.mizuroute_utils.mizuroute_to_blendmorph <https://bmorph.readthedocs.io/en/develop/api.html#bmorph.util.mizuroute_utils.mizuroute_to_blendmorph>` handles formatting the output of ``mizuroute`` for ``bmorph``. While configuring
streamflows can be performed without this function, this helps to speed up the whole workflow with a number of 
customizable options. As detailed in `bmorph_tutorial.rst <bmorph_tutorial.rst>`_, ``mizuroute_to_blendmorph`` 
is the primary function for handling streamflow formatting.

    
Output Specifications
=====================

Rerouting Local Corrected Flows
-------------------------------

``bmorph`` applies corrections to total flows, so rerouting ``bmorph`` outputs through ``mizuroute`` to retrieve local flows.

