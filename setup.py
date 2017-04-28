#!/usr/bin/env python
import os
import sys
from setuptools import setup

if "publish" in sys.argv[-1]:
    os.system("python setup.py sdist upload")
    sys.exit()

# Load the __version__ variable without importing the package already
exec(open('pyke3/version.py').read())

setup(name='pyke3',
      version=__version__,
      description="A backwards-incompatible, python3 compatible, pyraf-free "
                  "version of PyKE: a suite of tools to analyze Kepler/K2 "
                  "data",
      long_description=open('README.md').read(),
      author='KeplerGO',
      author_email='keplergo@mail.arc.nasa.gov',
      license='MIT',
      packages=['pyke3'],
      install_requires=['numpy>=1.11',
                        'astropy>=1.0',
                        'tqdm'],
      #entry_points=entry_points,
      include_package_data=True,
      classifiers=[
          "Development Status :: 5 - Production/Stable",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering :: Astronomy",
          ],
)
