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
      )
