Simple River Network (SRN)
==========================

Overview
--------

The Simple River Network, or SRN, is a graphical, psuedo-physical diagnostic tool used to visualize watershed models. Utilizing `NetworkX's <https://networkx.org/>`_ nodal network structure, SRNs represent each river segment, or `seg <https://bmorph.readthedocs.io/en/develop/data.html#variable-naming-conventions>`_, as a singular SegNode and connects them according to the watershed's topology. Each SRN is color-codable to assigned data values, such as percent bias, so you can visualize where issues may appear in the watershed during ``bmorph`` bias correction to more easily understand spatial patterns of bias correction in the network. 

.. image:: Figures/crb_srn_example.png
    :alt: Nodal network of the Columbia River Basin showing river segement connections and color-coded by Pfaffsetter basin.

SRN SegNode's contain identifying information that allow the network to be partitioned according to Pfaffstetter Codes (Verdin & Verdin 1999, Arge et. al. 2006). Pfaffstetter enconding not only allows the networks to be partitioned, but also to be "rolled up", effectively reducing the granularity of the network to simplify large watersheds. Data can also be subsected and split into new SRN's for simple manipulation.

SRN does not aim to supplant geographically accurate drawings of watershed networks. Instead it aims to provide a quicker, intermediate tool that allows for easy identification of spatial patterns within the network without having to configure spatial data. 

Construction
------------

[TBC]

Plotting on the SRN
-------------------

[TBC]

Citations
---------
Arge, L., Danner, A., Haverkort, H., & Zeh, N. (2006). I/O-Efficient Hierarchial Watershed Decomposition og Grid Terrain Models. In A. Riedl, W. Kainz, G.A. Elmes (Eds.), *Progress in Spatial Data Handling* (pp. 825-844). Springer, Berlin, Heidelberg. `https://doi.org/10.1007/3-540-35589-8_51 <https://doi.org/10.1007/3-540-35589-8_51>`_

Verdin, K.L., & Verdin, J. P. (1999). A topological system for delineation and codification of the Earth's river basins. *Elsevier Journal of Hydrology, 218*, 1-12. 

