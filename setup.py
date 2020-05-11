#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(name='bmorph',
      version='0.1',
      description='bias correction for time series',
      author='Bart Nijssen',
      author_email='nijssen@uw.edu',
      url='http://www.github.com/uw-hydro/bmorph',
      packages=find_packages(),
      install_requires=[
          'xarray',
          'pandas',
          'netcdf4',
          'numpy',
          'scipy',
          'dask',
          'toolz',
          'geopandas',
          'seaborn',
          'scikit-learn',
          'tqdm',
          'tensorflow>=2.0.0',
          'joblib',
          'networkx',
          'graphviz',
          'pygraphviz',
          'jupyter'
          ],
      keywords=['hydrology', 'streamflow', 'bias correction']
      )
