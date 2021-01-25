# -------------------------------------------------------------
# code developed by Michael Hartmann during his Ph.D.
# datareader: setup-file
#
# (C) 2020 Michael Hartmann, Graz, Austria
# Released under GNU GENERAL PUBLIC LICENSE
# email michael.hartmann@v2c2.at
# -------------------------------------------------------------

import setuptools
from setuptools import setup
import os
with open("README.md", "r") as fh:
    long_description = fh.read()

"""
    Function copied from https://stackoverflow.com/questions/27664504/how-to-add-package-data-recursively-in-python-setup-py
    18.01.2021 "How to add package data recursively in Python setup.py?"
"""
def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files('./datareader/data')

setup(name='datareader',
      version='0.0.9',
      description='Data reader',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/ga74kud/datareader',
      author='Michael Hartmann',
      author_email='michael.hartmann@v2c2.at',
      license='GNU GENERAL PUBLIC LICENSE',
      packages=setuptools.find_packages(),
      package_data={'': extra_files},
      include_package_data=True,
      install_requires=[
          "numpy",
          "pandas",
          "argparse",
          "scipy",
          "matplotlib",
          "opencv-python",
        ],
      zip_safe=False)
