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



Simple River Network
--------------------



Citations
---------

Arge, L., Danner, A., Haverkort, H., & Zeh, N. (2006). I/O-Efficient Hierarchial Watershed Decomposition og Grid Terrain Models. In A. Riedl, W. Kainz, G.A. Elmes (Eds.), *Progress in Spatial Data Handling* (pp. 825-844). Springer, Berlin, Heidelberg. https://doi.org/10.1007/3-540-35589-8_51_

Knoben, W. J. M., Freer, J. E., & Woods, R. A. (2019). Technical note: Inherent benchmark or not? Comparing Nash-Sutcliffe and Kling-Gupta efficiency scores. *Hydrology and Earth System Sciences, 23*, 4323-4331.  https://doi.org/10.5194/hess-23-4323-2019_

Verdin, K.L., & Verdin, J. P. (1999). A topological system for delineation and codification of the Earth's river basins. *Elsevier Journal of Hydrology, 218*, 1-12. 




