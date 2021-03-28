For Developers
==============

This file contains helpful tips & tricks found in the process of developing this project.

Converting notebooks to restructured text
-----------------------------------------

.. code:: ipython3

    jupyter nbconvert mynotebook.ipynb --to rst
    
For more information, check out `here <https://www.tutorialspoint.com/jupyter/jupyter_converting_notebooks.htm>`_

Helper functions
----------------

``log10_1p``
^^^^^^^^^^^^

Similar to numpy's `log1p <https://numpy.org/doc/stable/reference/generated/numpy.log1p.html>`_, `bmorph.evaluation.plotting.log10_1p <https://bmorph.readthedocs.io/en/develop/api.html#bmorph.evaluation.plotting.log10_1p>`_ has been added to address wanting to perform log10 computations on a dataset that contains zeros. It effectively adds 1 to the data, element-wise, and then takes the log10. This is useful in scientific plots where a log10 scale is desired yet zeros reside in the dataset.

``determine_row_col``
^^^^^^^^^^^^^^^^^^^^^

Tired of having to constantly reformat you subplots whenever you want to tack on one more plot or scratch off something you didnt' think you wanted? Well `bmorph.plotting.evaluation.determine_row_col <https://bmorph.readthedocs.io/en/develop/api.html#bmorph.evaluation.plotting.determine_row_col>` automates that process for you by calculating the tightest possible square/rectangular dimensions for your subplots. There may be some extra subplots leftover (and therefore we recommend turning off axis past the number you wish to plot), there will be at least enough subplots to fit all that you ask for.
