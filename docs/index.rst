.. _index:

bmorph
######

``bmorph`` is repository of bias correction methodologies designed to reduce statistical bias in streamflow models for watersheds. As a post-processing method, ``bmorph`` works in tandem with the streamflow model `mizuroute <https://mizuroute.readthedocs.io/en/latest/>`_. ``bmorph`` provides methods for integrating meteorologic sensitivity into the bias correction process and preserves  spatial relationships imposed by the channel network to ensure spatial consistency between gauge sites throughout bias correction. Below ``bmorph``'s bias correction methods are discussed and compared with other statistical post-processing methods. 

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
TO DO

Sitemap
=======
.. toctree::
    :maxdepth: 3

    index
    bias_correction
    bmorph_tutorial
    overview
    data
    evaluation
    for_developers
    api
