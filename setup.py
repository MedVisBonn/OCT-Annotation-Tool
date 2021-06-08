#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['numpy==1.14.3', 'scipy==1.1.0', 'qimage2ndarray==1.6', 'bresenham==0.2','matplotlib==2.2.2','pillow==8.2.0','pandas==0.23.0','sklearn==0.0','h5py==2.7.1','image==1.5.24','scikit-image==0.14.0','opencv-python==3.4.1.15']

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Shekoufeh Gorgi Zadeh",
    author_email='gorgi@cs.uni-bonn.de',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
    ],
    description="This tool can be used to annotate OCT scanns",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='oct_annotation_tool',
    name='oct_annotation_tool',
      
    packages=find_packages('src'),
    package_dir = {'': 'src'},
    scripts = ['src/OCT/controller/oct_controller.py'],
      
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/shekoufeh/oct_annotation_tool',
    version='0.1',
    zip_safe=False,
)
