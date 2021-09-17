For Developers
==============

This file contains helpful tips & tricks found in the process of developing this project.

Converting notebooks to restructured text
-----------------------------------------

.. code:: ipython3

    jupyter nbconvert mynotebook.ipynb --to rst
    
    
Any images in the notebook will be saved as png files within a newly created ``mynotebook_files`` and automatically referenced within the ``rst`` file.
    
For more information, check out `here <https://www.tutorialspoint.com/jupyter/jupyter_converting_notebooks.htm>`_

Helper functions from ``bmorph``
--------------------------------

``log10_1p``
^^^^^^^^^^^^

Similar to numpy's `log1p <https://numpy.org/doc/stable/reference/generated/numpy.log1p.html>`_, `bmorph.evaluation.plotting.log10_1p <https://bmorph.readthedocs.io/en/develop/api.html#bmorph.evaluation.plotting.log10_1p>`_ has been added to address wanting to perform log10 computations on a dataset that contains zeros. It effectively adds 1 to the data, element-wise, and then takes the log10. This is useful in scientific plots where a log10 scale is desired yet zeros reside in the dataset.

``determine_row_col``
^^^^^^^^^^^^^^^^^^^^^

Tired of having to constantly reformat you subplots whenever you want to tack on one more plot or scratch off something you didn't think you wanted? Well `bmorph.plotting.evaluation.determine_row_col <https://bmorph.readthedocs.io/en/develop/api.html#bmorph.evaluation.plotting.determine_row_col>`_ automates that process for you by calculating the tightest possible square/rectangular dimensions for your subplots. There may be some extra subplots leftover (and therefore we recommend turning off axis past the number you wish to plot), there will be at least enough subplots to fit all that you ask for.

Progress Bars
-------------

``tqdm``
^^^^^^^^

`tqdm <https://tqdm.github.io/docs/tqdm/>`_ is a customizable progress bar for iterators, used here in ``apply_scbc`` for example. These allow for an easily updatable status of progress to be printed from your scripts or notebooks. A few helpful arguments include ``disable`` to turn them off, ``leave`` to determine whether to keep the bar after it completes, or ``desc`` to provide a label for the progress bar. 

Creating Documentation
----------------------

Documentation for this project was done with `Sphinx <https://www.sphinx-doc.org/en/master/index.html>`_'s `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_. Compiling of the documentation into an HTMl format was performed by `Read the Docs <https://docs.readthedocs.io/en/stable/intro/getting-started-with-sphinx.html>`_. Thanks to through software, documentation was made easily updatable through GitHub version control without needing to develop a website from scratch in HTML itself.

The tutorial is made runnable by `binder <https://mybinder.org>`_ while data is stored on `Hydroshare <https://www.hydroshare.org/>`_ for simple online access.

