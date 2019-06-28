#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='lph',
      version='0.1',
      description='Learning to Plan Hierarchically from Curriculum',
      author='Philippe Morere',
      author_email='philippe.morere@sydney.edu.au',
      # packages=find_packages(),
      packages=[package for package in find_packages()
                if package.startswith('lph')],
      install_requires=[
          'sklearn',
          'gym',
          'tqdm',
          'matplotlib',
          'prettytable',
          'numpy'
      ],
      zip_safe=False)
