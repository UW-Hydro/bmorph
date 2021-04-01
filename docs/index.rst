.. _index:

bmorph
######

``bmorph`` is repository of bias correction methodologies designed to reduce statistical bias in streamflow models for watersheds. As a post-processing method, ``bmorph`` works in tandem with the streamflow model `mizuroute <https://mizuroute.readthedocs.io/en/latest/>`_. ``bmorph`` provides methods for integrating meteorologic sensitivity into the bias correction process and preserves  spatial relationships imposed by the channel network to ensure spatial consistency between gauge sites throughout bias correction.

In `Bias Correction <https://bmorph.readthedocs.io/en/develop/bias_correction.html>`_ we discuss the theory behind ``bmorph`` to develop a more intrinsic understanding of how it performs bias correction. The `Tutorial <https://bmorph.readthedocs.io/en/develop/bmorph_tutorial.html>`_ walks you through an example implementation of ``bmorph``. While the `API Reference <api.rst>`_ delves into the functions themselves, `Package Overview <overview.rst>`_ will get you aquinted with the ``bmorph`` pack structure while `Data Overview <https://bmorph.readthedocs.io/en/develop/data.html>`_, `Input Specifications <https://bmorph.readthedocs.io/en/develop/data.html#input-specifications>`_, and `Output Specifications <https://bmorph.readthedocs.io/en/develop/data.html#output-specifications>`_ will cover the ins and outs of the overall workflow.

Installation
============
We provide a conda environment in ``environment.yml``. You can build the environment by running:

``conda env create -f environment.yml``

Then, to install ``bmorph`` run,

.. code-block::

   conda activate bmorph
   python setup.py develop
   python -m ipykernel install --user --name bmorph


Getting started
===============

A step-by-step tutorial can be found in documentation form `here <https://bmorph.readthedocs.io/en/develop/bmorph_tutorial.html>`_.
We also have an interactive instance of the tutorial `here <https://notebooks.gesis.org/binder/badge_logo.svg)](https://notebooks.gesis.org/binder/v2/gh/UW-Hydro/bmorph/develop>`_.


Sitemap
=======
.. toctree::
    :maxdepth: 2

    index
    bias_correction
    bmorph_tutorial
    overview
    data
    evaluation
    for_developers
    api
