Evaluation of Bias Correction **(IN PROGRESS)**
===============================================

Fundamental Statistics
----------------------

Below are the statistics used in ``bmorph`` to evaluate bias correction performance.
Let P be predicted values, such as the corrected flows, and O be the observed values, such as reference flows.

Mean Bias Error (MBE)
^^^^^^^^^^^^^^^^^^^^^

.. math::
    
    MBE = \frac{\Sigma_{i=0}^{n}(P_i - O_i)}{n}
    

Root Mean Square Error (RMSE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

    RMSE = \sqrt{\frac{\Sigma_{i=0}^{n}(P_i - O_i)^2}{n}}

Percent Bias (PB)
^^^^^^^^^^^^^^^^^

    PB = 100% * \frac{\Sigma_{i=0}^{n}(P_i - O_i)}{\Sigma_{i=0}^{n}O_i}
    
Kullback-Leibler Divergence (KL Divergence)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



Kling-Gupta Efficiency (KGE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^



Plotting
--------

The ``Evaluation`` package of ``bmorph`` comes with a number of helpful plotting tools to analyze bias correction performance. Whether you want to compare corrected to uncorrected values, contrast different bias correction parameters, or simply compile your results into a tidy plot across multiple sites, there are plenty of plotting functions to choose from.

Scatter Plots
^^^^^^^^^^^^^



Time Series
"""""""""""



Probabilty Distribtuions
""""""""""""""""""""""""



Box & Whisker Plots
^^^^^^^^^^^^^^^^^^^



Simple River Network
--------------------

The Simple River Network, or SRN, is a graphical, psuedo-physical diagnostic tool used to visualize watershed models. Utilizing <NetworkX's <https://networkx.org/>`_ nodal network structure, SRN's represent each river segment, or <seg <data.rst/Common Naming Conventions>`_, as a singular SegNode and connects them according to the watershed's topology. Each SRN is color-codable to assigned data values, such as percent bias, so you can visualize where issues may appear in ``bmorph`` bias correction the watershed to more easily understand spatial patterns in bias correction of the network. 

SRN SegNode's contain identifying information that allow the network to be partitioned according to Pfaffstetter Codes (Verdin & Verdin 1999, Arge et. al. 2006). Pfaffstetter enconding not only allows the networks to be partitioned, but also to be "rolled up", effectively reducing the granularity of the network to simplify large watersheds. Data can also be subsectted and split into new SRN's for simple manipulation.

SRN does not aim to supplant geographically accurate drawings of watershed networks. Instead it aims to provide a quicker, intermediate tool that allows for easy identification of spatial patterns within the network without having to configure spatial data. 

Citations
---------

Arge, L., Danner, A., Haverkort, H., & Zeh, N. (2006). I/O-Efficient Hierarchial Watershed Decomposition og Grid Terrain Models. In A. Riedl, W. Kainz, G.A. Elmes (Eds.), *Progress in Spatial Data Handling* (pp. 825-844). Springer, Berlin, Heidelberg. https://doi.org/10.1007/3-540-35589-8_51_

Knoben, W. J. M., Freer, J. E., & Woods, R. A. (2019). Technical note: Inherent benchmark or not? Comparing Nash-Sutcliffe and Kling-Gupta efficiency scores. *Hydrology and Earth System Sciences, 23*, 4323-4331.  https://doi.org/10.5194/hess-23-4323-2019_

Verdin, K.L., & Verdin, J. P. (1999). A topological system for delineation and codification of the Earth's river basins. *Elsevier Journal of Hydrology, 218*, 1-12. 




