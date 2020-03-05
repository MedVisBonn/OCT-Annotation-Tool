-------------------
OCT-Annotation-Tool
-------------------
The OCT-Annotation Tool automatically segments the RPE and BM layers in OCT scans using a convolution neuronal network. Uncertainty measures lead the user to those B-scans that probably need manual correction. In addition, the software automatically segments drusen from the RPE and BM layers and has tools for quickly editing them.

* Free software: GNU General Public License v3

Installation
---------------

This software requires python 2. Most of the required python modules are included in the setup.py file and can be installed by:

``pip install -e path``

where path is the path to the folder that contains the setup.py file.

Additionally required modules are PyQt4 and caffe with U-Net patch included. The following links can be used for the installation:

* PyQt4: http://pyqt.sourceforge.net/Docs/PyQt4/installation.html

* caffe: You need to select the U-Net patch for your CUDA and cuDNN version from https://lmb.informatik.uni-freiburg.de/resources/opensource/unet/ .

For caffe installation, follow the instructions in the README file in the zipped file. 

Getting Started
------------------

The software can be started by running ``oct_controller.py`` from ``src/OCT/controller``. To test the software, download the pretrained caffe model from
https://uni-bonn.sciebo.de/s/vpVBVwUoXtQer8d
for layer segmentation and store it in ``src/OCT/network`` folder. The path to the deep network should be set under the ``setting`` under ``File`` tab in the annotation software. For Caffe Path, use the link to ``<caffe_dir>/python``.

There are different tools available to manually modifiy the layer segmentation determined by the CNN:

1) **Pen**: Mark individual pixels as belonging to the selected layer

2) **Line Marker**: Mark every pixel along a line segment as belonging to the selected layer

3) **Spline fit**: Fit a B-spline to the selected layer. User can drag, add and delete Spline knots

4) **Constrained Shortest Path Tool (CSP)**: Select a single point through which the new shortest path is enforced to run

5) **Local Smoothing Tool (LS)**: Select a layer and a horizontal interval on the image that is replaced by a smoothed version by fitting a low-degre polynomial to the pixel positions

6) **Layer Suggestion With Respect to Adjacent B-scans**: For RPE layer, uses the information of CSP tool used in the adjacent B-scan to suggest a new RPE layer. User can accept or ignore it. For BM layer, a 2D spline fit over the BM surface is used to provide suggestion for segmentation

7) **Fill**: Mark every pixel up to RPE layer in a closed region as drusen.

8) **Semi-automated Drusen Extraction (SDE)**: In a selected region by user, automatically extract the lower boundary of drusen and mark every pixel within the lower boundary up to RPE layer as drusen.

Additionally, the uncertainty of the CNN layer segmentation can be displayed to guide the user's attentions to images or locations where manual correction is needed. Under view, it can be selected to show segmentation uncertainty in terms of probability and entropy. The uncertainty will be displayed as a color coded table at the bottom of the viewer and also the line representing the layer segmentation gets colored. Seperate color maps are used for the entropy (orange/red) and probability (blue). More saturated colors corresponds to higher uncertainty.  In the color coded table at the bottom, the color of the ith cell represents the uncertainty in the ith B-scan.

More detailed information can be found in our publication (see below). The paper PDF is available from the authors' website http://cg.cs.uni-bonn.de/aigaion2root/attachments/gorgizadeh-guided-editing-2018.pdf and in the Eurographics Digital Library http://dx.doi.org/10.2312/vcbm.20181235

Dataset
------------------

To test the Annotation-Tool you can use the exemplary OCT volume scan in the ``data/OCT-Volume-Scan`` folder. To provide a reference for RPE and BM layer segmentation, a medical expert performed a careful manual correction of an initial segmentation, spending about 10 minutes per B-scan on average, adding up to a total of 26 hours. This ground truth is available at the ``data/RPE-BM-Layers`` directory, where each image is a segmentation map for RPE (upper layer) and BM (lower layer). Also, ``data/OCT-Info.xml`` file includes additional information about the OCT volume scan such as B-scan quality (marked by ``<ImageQuality>`` tag), size of each B-scan pixel in micrometer (marked by ``<ScaleX>``, ``<ScaleY>`` tags), and start and end location of B-scans (marked by ``<X>`` and ``<Y>`` tags in ) in the axial direction as represented here.

![](images/img.png)

Reference
----------

If you use our software as part of a scientific project, please cite the corresponding publication

* Shekoufeh Gorgi Zadeh, Maximilian W.M. Wintergerst, Thomas Schultz: Uncertainty-Guided Semi-Automated Editing of CNN-based Retinal Layer Segmentations in Optical Coherence Tomography. In: Proc. EG Workshop on Visual Computing for Biology and Medicine (VCBM), pp. 107-115, Eurographics, 2018.
  
Authors
----------

* Shekoufeh Gorgi Zadeh (https://github.com/shekoufeh)


Credits
-------
This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

