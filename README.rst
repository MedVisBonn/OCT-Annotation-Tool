-------------------
OCT-Annotation-Tool
-------------------
The OCT-Annotation Tool automaticall segments the RPE and BM layers in OCT scans using a convolution neuronal network. Uncertainty measures lead the user to those B-scans that probably need manual correction. In addition, the software automatically segments drusen from the RPE and BM layers and has tools for quickly editing them.

* Free software: GNU General Public License v3

Installation
---------------

This software requires python 2. Most of the required python modules are included in the setup.py file and can be installed by:

``pip install -e path``

where path is the path to the folder that contains the setup.py file.

Additionally required modules are PyQt4 and caffe with U-Net patch included. The following links can be used for the installation:

* PyQt4: http://pyqt.sourceforge.net/Docs/PyQt4/installation.html

* caffe: https://lmb.informatik.uni-freiburg.de/resources/opensource/caffe_unet_3D_v1.0.tar.gz


Getting Started
------------------

The software can be started by running ``oct_controller.py`` from ``src/OCT/controller``. To test the software, download the pretrained caffe model from
https://uni-bonn.sciebo.de/s/khbu3tAQ95a4oTy
for layer segmentation and store it in ``src/OCT/network`` folder. The path to the deep network can be set under the ``setting`` under ``File`` tab in the annotation software. For Caffe Path, use the link to ``<caffe_dir>/python``.

There are different tools available to manually modifiy the layer segmentation determined by the CNN:

1) **Pen**: Mark individual pixels as belonging to the selected layer

2) **Line Marker**: Mark evey pixel along a line segment as belonging to the selected layer

3) **Constrained Shortest Path Tool (CSP)**: Select a single point through which the new shortest path is enforced to run

4) **Local Smoothing Tool (LS)**: Select a layer and a horizontal interval on the image that is replaced by a smoothed version by fitting a low-degre polynomial to the pixel positions

Additionally, the uncertainty of the CNN layer segmentation can be displayed to guide the user's attentions to images or locations where manual correction is needed. Under view, it can be selected to show segmentation uncertainty in terms of probability and entropy. The uncertainty will be displayed as a color coded table at the bottom of the viewer and also the line representing the layer segmentation gets colored. Seperate color maps are used for the entropy (orange/red) and probability (blue). More saturated colors corresponds to higher uncertainty.  In the color coded table at the bottom, the color of the ith cell represents the uncertainty in the ith B-scan.

For a more detailed instruction, please refer to: link to pdf

Authors
----------

* Shekoufeh Gorgi Zadeh (https://github.com/shekoufeh)


Credits
-------
This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

