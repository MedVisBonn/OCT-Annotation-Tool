# -*- coding: utf-8 -*-
"""
Created in 2018

@author: Shekoufeh Gorgi Zadeh
"""
import logging

import re
import os
from os import listdir
from os.path import isfile, join
import copy
import pickle
import imageio
import numpy as np
import scipy as sc
import pandas as pd
from skimage import io

from scipy import misc
from scipy.interpolate import UnivariateSpline
import skimage.measure as skm
from PyQt4 import QtCore, QtGui

import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt

import deeplearning
import drusenextractor
import iovol

# Logging setup for file
logging.basicConfig(filename=os.path.join(os.path.expanduser('~'), 'octannotation.log'),
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    level=logging.DEBUG,
                    filemode='w')
logger = logging.getLogger('OCT')

octParams = dict()
octParams['bResolution'] = 'high'  # high
octParams['hx'] = 200. / 17.  # x axis in each B-scan
octParams['hy'] = 200. / 51.  # y axis in each B-scan
octParams['hz'] = 200. / 10.  # z axis in direction of B-scan slices
octParams['zRate'] = 2  # every 13 pixels for low resolution, every two pixels for high res


# ==============================================================================
# OCT class contains the OCT volume, and related segmentation maps
# ==============================================================================
class OCT:

    def __init__(self, controller):
        self.saveFormat = 'png'
        self.scanPath = ""
        self.bResolution = ''
        self.hx = 200. / 17.  # x axis in each B-scan
        self.hy = 200. / 51.  # y axis in each B-scan
        self.hz = 200. / 10.  # z axis in direction of B-scan slices
        self.zRate = 0.  # every 13 pixels for low resolution, every two pixels for high res
        self.scans = None
        self.layers = None
        self.hrfs = None
        self.gas = None
        self.ngas = None
        self.drusen = None
        self.enface = None
        self.enfaceDrusen = None
        self.hrfStatus = None
        self.hrfBBox = None
        self.ngaBBox = None
        self.enfaceBBox = None
        self.probmaps = None
        self.uncerEntRPE = None
        self.uncerProRPE = None
        self.uncerEntBM = None
        self.uncerProBM = None
        self.enfaceRPE = None
        self.enfaceBM = None

        # Probable layers
        self.rpeCSPs = None
        self.bmProbable = None
        self.rpeSuggestExtent = 0
        self.bmSuggestExtent = 0
        self.rpeSuggestShow = True
        self.bmSuggestShow = True

        self.numSlices = 0
        self.width = 0
        self.height = 0

        self.scanIDs = list()

        # Drusen analysis
        self.cx = None
        self.cy = None
        self.area = None
        self.height = None
        self.volume = None
        self.largeR = None
        self.smallR = None
        self.theta = None

        self.controller = controller

        self.layerSegmenter = None
        self.drusenSegmenter = None

        self.progressBarValue = 0

        self.certainSlices = list()
        self.editedLayers = list()

        self.GTlayers = None
        self.distances = list()
        self.overallDistance = 10000

        self.GTdrusen = None
        self.drusenDistances = list()
        self.drusenOverallDistance = 10000

        self.splineKnots = {'RPE': dict(), 'BM': dict()}

        self.evaluateLayers = False
        self.evaluateDrusen = False

        self.numTiles = None

    def set_num_of_tiles(self, num):
        self.numTiles = num

    def set_rpe_suggest_show(self, status):
        self.rpeSuggestShow = status

    def set_bm_suggest_show(self, status):
        self.bmSuggestShow = status

    def get_rpe_csps(self, sliceNumZ):
        if not self.rpeCSPs is None and not self.rpeCSPs[sliceNumZ] is None:
            return self.rpeCSPs[sliceNumZ]
        return None

    def get_enface_RPE(self):
        return self.enfaceRPE

    def get_enface_BM(self):
        return self.enfaceBM

    def get_num_scans(self):
        return self.numSlices

    def get_dim(self):
        return self.height, self.width

    def get_GT_layer(self, layerNumber):
        return np.copy(self.GTlayers[:, :, layerNumber - 1])

    def show_image(self, image, block=True):
        plt.imshow(image, cmap=plt.get_cmap('gray'))
        plt.show(block)

    def set_rpe_suggest_extent(self, val):
        self.rpeSuggestExtent = val

    def set_bm_suggest_extent(self, val):
        self.bmSuggestExtent = val

    def get_rpe_suggest_extent(self):
        return self.rpeSuggestExtent

    def get_bm_suggest_extent(self):
        return self.bmSuggestExtent

    def set_spline_knots(self, knots):
        self.splineKnots = knots

    def set_spline_knots_layer(self, knots, sliceZ, layerName):
        self.splineKnots[layerName][sliceZ] = knots

    def set_scan_path(self, scanPath):
        self.scanPath = scanPath

    def set_scan(self, scans):
        self.scans = scans

    def set_layers(self, layers):
        self.layers = layers

    def set_layer(self, layer, sliceNumZ):
        self.layers[:, :, sliceNumZ] = layer

    def set_prob_maps(self, probMaps):
        self.probmaps = probMaps

    def set_hrfs(self, hrfs):
        self.hrfs = hrfs

    def set_gas(self, gas):
        self.gas = gas

    def set_ga(self, s, e, color, sliceNum):
        self.gas[:, s:e + 1, sliceNum - 1] = color

    def set_ngas(self, ngas):
        self.ngas = ngas

    def set_nga(self, s, e, color, sliceNum):
        self.ngas[:, s:e + 1, sliceNum - 1] = color

    def set_drusen(self, drusen):
        self.drusen = drusen

    def set_enface(self, enface):
        self.enface = enface

    def set_enface_drusen(self, enfaceDrusen):
        self.enfaceDrusen = enfaceDrusen

    def set_drusen_b_scan(self, sliceD, sliceNumber):
        self.drusen[:, :, int(sliceNumber) - 1] = sliceD

    def set_HRF_status(self, index, status):
        self.hrfStatus[index] = status

    def set_hrf_bounding_boxes(self, bbox):
        self.hrfBBox = bbox

    def set_nga_bounding_boxes(self, bbox):
        self.ngaBBox = bbox

    def set_enface_bounding_boxes(self, bbox):
        self.enfaceBBox = bbox

    def set_smoothness_value(self, value):
        self.layerSegmenter.set_yLength(value)

    def set_progress_val(self, val):
        self.progressBarValue = val

    def set_uncertainty_color_map(self, umap, layerName, uncertaintyType):
        if layerName == 'RPE':
            if uncertaintyType == 'Entropy':
                self.uncerEntRPE = umap
            elif uncertaintyType == 'Probability':
                self.uncerProRPE = umap
        elif layerName == 'BM':
            if uncertaintyType == 'Entropy':
                self.uncerEntBM = umap
            elif uncertaintyType == 'Probability':
                self.uncerProBM = umap

    def get_uncer_map(self, layerName, uncertaintyType):
        if layerName == 'RPE':
            if uncertaintyType == 'Entropy':
                return self.uncerEntRPE
            elif uncertaintyType == 'Probability':
                return self.uncerProRPE
        elif layerName == 'BM':
            if uncertaintyType == 'Entropy':
                return self.uncerEntBM
            elif uncertaintyType == 'Probability':
                return self.uncerProBM
        return None

    ## Saving

    def save_bscans(self, savePath):
        if self.scans is not None:
            logger.debug('{},{}'.format(np.max(self.scans), self.scans.dtype))
            for i in range(self.scans.shape[2]):
                scan_path = os.path.join(savePath, str(i) + '.tif')
                imageio.imwrite(scan_path, self.scans[:, :, i].astype(np.uint8), format='.tif')

    def save_drusen(self, savePath):
        savePath = os.path.join(savePath, 'drusen')

        if self.drusen is not None:
            self.create_directory(savePath)
            if self.saveFormat == 'pkl':
                druLoc = np.where(self.drusen > 0)
                self.write_pickle_data(os.path.join(savePath, 'drusen.pkl'), druLoc)
            else:
                self.controller.show_progress_bar("Saving")
                pStep = 100 / float(max(1, self.drusen.shape[2]))
                for s in range(self.drusen.shape[2]):
                    self.controller.update_progress_bar_value(pStep)
                    misc.imsave(os.path.join(savePath, str(self.scanIDs[s]) + \
                                             '-drusen.png'), self.drusen[:, :, s])
                self.controller.hide_progress_bar()

    def save_layers(self, saveP):

        savePath = os.path.join(saveP, 'layers')
        savePath2 = os.path.join(saveP, 'probabilityMaps')

        if self.layers is not None:
            self.create_directory(savePath)
            self.create_directory(savePath2)
            layers = self.change_layers_format_for_saving()
            if self.saveFormat == 'pkl':
                layerLoc = dict()
                vs = np.unique(layers)
                for v in vs:
                    if v == 0:
                        continue
                    layerLoc[v] = np.where(layers == v)
                self.write_pickle_data(os.path.join(savePath, 'layers.pkl'), layerLoc)
            else:
                self.controller.show_progress_bar("Saving")
                pStep = 100 / float(max(1, layers.shape[2]))
                for s in range(layers.shape[2]):
                    self.controller.update_progress_bar_value(pStep)
                    misc.imsave(os.path.join(savePath, str(self.scanIDs[s]) + \
                                             '-layers.png'), layers[:, :, s])
                    if self.probmaps is not None:
                        io.imsave(os.path.join(savePath2, str(self.scanIDs[s]) + \
                                               '-0-layers.tif'), self.probmaps[:, :, 0, s])
                        io.imsave(os.path.join(savePath2, str(self.scanIDs[s]) + \
                                               '-1-layers.tif'), self.probmaps[:, :, 1, s])
                        io.imsave(os.path.join(savePath2, str(self.scanIDs[s]) + \
                                               '-2-layers.tif'), self.probmaps[:, :, 2, s])
                        io.imsave(os.path.join(savePath2, str(self.scanIDs[s]) + \
                                               '-3-layers.tif'), self.probmaps[:, :, 3, s])
                uc = self.controller.get_uncertainties()
                if not uc is None:
                    u1, u2, u3 = uc
                    if (not u1 is None) and (not u2 is None) and (not u3 is None):
                        np.savetxt(os.path.join(savePath2, 'prob-entropy.txt'), np.asarray(u1))
                        np.savetxt(os.path.join(savePath2, 'prob.txt'), np.asarray(u2))
                        np.savetxt(os.path.join(savePath2, 'entropy.txt'), np.asarray(u3))

                self.controller.hide_progress_bar()

    def save_bbox(self, fileName, bboxesIn):
        f = open(fileName, 'w')
        if bboxesIn is not None:
            for sliceNum in bboxesIn.keys():
                strline = str(sliceNum + 1) + ':'
                bboxes = bboxesIn[sliceNum]
                for bbox in bboxes:
                    tl = bbox.topLeft()
                    br = bbox.bottomRight()
                    strline = strline + str(tl.x()) + ',' + str(tl.y()) + ',' + str(br.x()) + \
                              ',' + str(br.y()) + '|'
                f.write(strline + '\n')
        f.close()

    def save_hrfs(self, savePath):
        savePath = os.path.join(savePath, 'HRF')
        if self.hrfs is not None:
            self.create_directory(savePath)
            if self.saveFormat == 'pkl':
                hrfLoc = np.where(self.hrfs > 0)
                self.write_pickle_data(os.path.join(savePath, 'hrfs.pkl'), hrfLoc)
            else:
                self.controller.show_progress_bar("Saving")
                pStep = 100 / float(max(1, self.hrfs.shape[2]))
                for s in range(self.hrfs.shape[2]):
                    self.controller.update_progress_bar_value(pStep)
                    imageio.imwrite(os.path.join(savePath, str(self.scanIDs[s]) + '-hrf.png'),
                                    self.hrfs[:, :, s])
                self.controller.hide_progress_bar()
            np.savetxt(os.path.join(savePath, 'hrfs.txt'), self.hrfStatus)
            self.save_bbox(os.path.join(savePath, 'hrfs-bounding-box.txt'), \
                           self.hrfBBox)

    def save_gas(self, savePath):
        savePath = os.path.join(savePath, 'GA')
        if self.gas is not None:
            self.create_directory(savePath)
            if self.saveFormat == 'pkl':
                gaLoc = np.where(self.gas > 0)
                self.write_pickle_data(os.path.join(savePath, 'gas.pkl'), gaLoc)
            else:
                self.controller.show_progress_bar("Saving")
                pStep = 100 / float(max(1, self.gas.shape[2]))
                for s in range(self.gas.shape[2]):
                    self.controller.update_progress_bar_value(pStep)
                    misc.imsave(os.path.join(savePath, str(self.scanIDs[s]) + \
                                             '-ga.png'), self.gas[:, :, s])
                self.controller.hide_progress_bar()

    def save_ngas(self, savePath):
        savePath = os.path.join(savePath, 'nGA')
        if self.ngas is not None:
            self.create_directory(savePath)
            if self.saveFormat == 'pkl':
                gaLoc = np.where(self.ngas > 0)
                self.write_pickle_data(os.path.join(savePath, 'ngas.pkl'), gaLoc)
            else:
                self.controller.show_progress_bar("Saving")
                pStep = 100 / float(max(1, self.ngas.shape[2]))
                for s in range(self.ngas.shape[2]):
                    self.controller.update_progress_bar_value(pStep)
                    misc.imsave(os.path.join(savePath, str(self.scanIDs[s]) + \
                                             '-nga.png'), self.ngas[:, :, s])
                self.controller.hide_progress_bar()
            self.save_bbox(os.path.join(savePath, 'ngas-bounding-box.txt'), \
                           self.ngaBBox)

    def save_enface(self, savePath):
        savePath = os.path.join(savePath, 'reticular-drusen')
        self.create_directory(savePath)
        self.save_bbox(os.path.join(savePath, \
                                    'reticular-drusen-bounding-box.txt'), self.enfaceBBox)

    def save_drusen_quantification(self, savePath, unit='pixel'):
        savePath = os.path.join(savePath, 'drusen-analysis')
        self.create_directory(savePath)

        if unit == 'pixel':
            cxM = self.cx
            cyM = self.cy
            areaM = self.area
            heightM = self.height
            volumeM = self.volume
            largeM = self.largeR
            smallM = self.smallR
            saveName = os.path.join(savePath, 'drusen-analysis-pixel.xlsx')

        elif unit == 'micrometer':
            cxM, cyM, areaM, heightM, volumeM, largeM, smallM, \
            theta = self.convert_from_pixel_size_to_meter()
            saveName = os.path.join(savePath, 'drusen-analysis-micrometer.xlsx')

        drusenInfo = dict()
        drusenInfo['Center'] = list()
        drusenInfo['Area'] = list()
        drusenInfo['Height'] = list()
        drusenInfo['Volume'] = list()
        drusenInfo['Diameter'] = list()

        for i in range(len(cxM)):
            drusenInfo['Center'].append((int(cxM[i]), int(cyM[i])))
            drusenInfo['Area'].append(areaM[i])
            drusenInfo['Height'].append(heightM[i])
            drusenInfo['Volume'].append(volumeM[i])
            drusenInfo['Diameter'].append((largeM[i], smallM[i]))

        df = pd.DataFrame(drusenInfo, index=(np.arange(len(areaM)) + 1), \
                          columns=['Center', 'Area', 'Height', 'Volume', 'Diameter'])
        df.to_csv(saveName, sep='\t')
        return

    ## Loading

    def load_probmaps(self):
        path = os.path.join(self.scanPath, 'probabilityMaps')

        try:
            nameing = 'layers.tif'
            self.controller.show_progress_bar()
            self.progressBarValue = 1
            self.controller.set_progress_bar_value(self.progressBarValue)

            self.probmaps = np.zeros(self.scans.shape[:2], +(4,), self.scans.shape[-1])
            files = [f for f in listdir(path) if isfile(os.path.join(path, f))]
            stepsize = 100 / len(files)
            for filename in files:
                i, j, ftype = filename.split('-')
                if ftype == nameing:
                    self.probmaps[:, :, j, i] = io.imread(filename)

                self.controller.update_progress_bar_value(stepsize)

            self.controller.hide_progress_bar()
        except:
            raise Exception('No probability maps found for loading')

        self.layerSegmenter.layers = self.layers

    def load_bscans(self, savePath):
        # bscans are saved like 10.tif but we do not load enface.tif for example
        scanregex = re.compile(r'\d+.tif')
        files = [f for f in listdir(savePath) if isfile(join(savePath, f)) and scanregex.search(f) is not None]
        files.sort(key=self.natural_keys)
        img0 = imageio.imread(join(savePath, files[0]))

        self.scans = np.zeros(img0.shape + (len(files),))
        self.scans[..., 0] = img0

        for i in range(1, len(files)):
            self.scans[..., i] = imageio.imread(join(savePath, files[i]))

        self.scanIDs = list(range(1, len(files)+1))
        self.numSlices = self.scans.shape[2]
        self.width = self.scans.shape[1]
        self.height = self.scans.shape[0]

        if self.numSlices > 50:
            self.zRate = 2
            self.bResolution = 'high'
        else:
            self.zRate = 13
            self.bResolution = 'low'

    def load_hrfs(self):
        path = os.path.join(self.scanPath, 'HRF')

        if self.saveFormat == 'pkl':
            hrfLoc = self.read_pickle_data(os.path.join(path, 'hrfs.pkl'))
            self.hrfs = np.zeros(self.scans.shape)
            self.hrfs[hrfLoc] = 255

        else:
            if self.saveFormat == 'png':
                nameing = 'hrf.png'
                self.hrfs = self.load_images(path, nameing)

        return self.hrfs

    def load_gas(self):
        path = os.path.join(self.scanPath, 'GA')

        if self.saveFormat == 'pkl':
            gasLoc = self.read_pickle_data(os.path.join(path, 'gas.pkl'))
            self.gas = np.zeros(self.scans.shape)
            self.gas[gasLoc] = 255

        else:
            if self.saveFormat == 'png':
                nameing = 'ga.png'
                self.gas = self.load_images(path, nameing)
        return self.gas

    def load_ngas(self):
        path = os.path.join(self.scanPath, 'nGA')

        if self.saveFormat == 'pkl':
            ngasLoc = self.read_pickle_data(os.path.join(path, 'ngas.pkl'))
            self.ngas = np.zeros(self.scans.shape)
            self.ngas[ngasLoc] = 255

        else:
            if self.saveFormat == 'png':
                nameing = 'nga.png'
                self.ngas = self.load_images(path, nameing)
        return self.ngas

    def load_images(self, path, naming):
        self.controller.show_progress_bar()
        self.progressBarValue = 1
        self.controller.set_progress_bar_value(self.progressBarValue)

        images = np.zeros(self.scans.shape)
        regex = re.compile('\d-{}'.format(naming))
        files = [f for f in listdir(path) if os.path.isfile(os.path.join(path, f)) and regex.search(f) is not None]
        files.sort(key=self.natural_keys)

        for i in range(len(files)):
            filepath = join(path, files[i])
            images[:, :, i] = imageio.imread(filepath)

        self.controller.hide_progress_bar()
        return images

    def load_layers(self):
        path = os.path.join(self.scanPath, 'layers')

        if self.saveFormat == 'pkl':
            layersLoc = self.read_pickle_data(os.path.join(path, 'layers.pkl'))
            self.layers = np.zeros(self.scans.shape)
            keys = layersLoc.keys()
            for v in keys:
                self.layers[layersLoc[v]] = v

        else:
            if self.saveFormat == 'png':
                nameing = 'layers.png'
                self.layers = self.load_images(path, nameing)
            else:
                try:
                    naming = 'BinSeg.tif'
                    self.layers = self.load_images(path, naming)
                except:
                    raise Exception('No layers found for loading')

        self.change_layers_format_for_GUI()

        return self.layers

    def load_drusen(self, hfilter):
        path = os.path.join(self.scanPath, 'drusen')
        if not os.path.exists(path) or not os.listdir(path):
            raise Exception('No saved drusen available')

        if self.saveFormat == 'pkl':
            druLoc = self.read_pickle_data(os.path.join(path, 'drusen.pkl'))
            self.drusen = np.zeros(self.scans.shape)
            self.drusen[druLoc] = 255

        else:
            if self.saveFormat == 'png':
                nameing = 'drusen.png'
                drusen = self.load_images(path, nameing)
            else:
                try:
                    naming = 'binmask.tif'
                    drusen = self.load_images(path, naming)
                except:
                    raise Exception('No layers found for loading')

        drusen[drusen > 0] = 1
        # drusen = self.filter_drusen_by_size(drusen)
        # if (hfilter > 0):
        #    drusen = self.filter_druse_by_max_height(drusen, hfilter)

        self.drusen = drusen * 255

        if self.evaluateDrusen:
            self.get_GT_drusen()

        return self.drusen

    ## Getters

    def get_scan_path(self):
        return self.scanPath

    def get_scan(self):
        return self.scans

    def get_slo(self):
        return self.slo

    def get_probmaps(self):
        """

        :return:
        """
        if self.probmaps is None:
            try:
                self.probmaps = self.load_probmaps()
            except:
                self.layers, self.probmaps = self.compute_layers()

        if self.layerSegmenter is None:
            self.layerSegmenter = deeplearning.DeepLearningLayerSeg(self)

        return self.probmaps

    def get_bbox_from_file_reticular(self, fileName):
        """
        Load the bounding box annotation for reticular drusen if there are any.
        """
        retBBox = dict()
        try:
            f = open(fileName, 'r')
            lines = f.readlines()
            for l in lines:
                l = l.rstrip()
                boxes = l.split(':')[1]
                boxes = boxes.split('|')
                for box in boxes:
                    pnts = box.split(',')
                    if len(pnts) == 4:
                        rect = QtCore.QRect(QtCore.QPoint(int(pnts[0]), \
                                                          int(pnts[1])), QtCore.QPoint(int(pnts[2]), \
                                                                                       int(pnts[3])))
                        if not 0 in retBBox.keys():
                            retBBox[0] = list()
                        retBBox[0].append(rect)
            return retBBox
        except:
            return retBBox

    def get_bbox_from_file(self, fileName):
        """
        Load the bounding box annotation for HRFs if there are any.
        """
        hrfBBox = dict()
        try:
            f = open(fileName, 'r')
            lines = f.readlines()
            for l in lines:
                l = l.rstrip()
                sliceNum = int(l.split(':')[0])
                boxes = l.split(':')[1]
                boxes = boxes.split('|')
                for box in boxes:
                    pnts = box.split(',')
                    if len(pnts) == 4:
                        rect = QtCore.QRect(QtCore.QPoint(int(pnts[0]), \
                                                          int(pnts[1])), QtCore.QPoint(int(pnts[2]), \
                                                                                       int(pnts[3])))
                        if not sliceNum - 1 in hrfBBox.keys():
                            hrfBBox[sliceNum - 1] = list()
                        hrfBBox[sliceNum - 1].append(rect)
            return hrfBBox
        except:
            return hrfBBox

    def get_hrfs(self):
        """
        Read the HRF segmentation from the disk if they exist.
        """
        try:
            path = os.path.join(self.scanPath, 'HRF')
        except:
            path = None

        if self.hrfs is None:
            try:
                self.hrfs = self.load_hrfs()
            except:
                self.hrfs = np.zeros(self.scans.shape)

            try:
                self.hrfStatus = np.loadtxt(os.path.join(path, 'hrfs.txt')).astype(bool)
            except:
                self.hrfStatus = np.full(self.scans.shape[2], False, dtype=bool)

            try:
                self.hrfBBox = self.get_bbox_from_file(os.path.join(path, 'hrfs-bounding-box.txt'))
            except:
                self.hrfBBox = dict()

        logger.debug('HRFs types: {}, {}, {}'.format(type(self.hrfs), type(self.hrfStatus), type(self.hrfBBox)))
        return self.hrfs, self.hrfStatus, self.hrfBBox



    def get_gas(self):
        """
        Read the GA segmentation maps from the disk if they exist.
        """
        if self.gas is None:
            try:
                self.gas = self.load_gas()
            except:
                self.gas = np.zeros(self.scans.shape)

        return self.gas

    def get_ngas(self):
        """
        Read the NGA segmentation maps from the disk if they exist.
        """
        try:
            path = os.path.join(self.scanPath, 'nGA')
        except:
            path = None

        if self.ngas is None:
            try:
                self.ngas = self.load_gas()
            except:
                self.ngas = np.zeros(self.scans.shape)

            try:
                self.ngaBBox = self.get_bbox_from_file(os.path.join(path, 'ngas-bounding-box.txt'))
            except:
                self.ngaBBox = dict()

        return self.ngas, self.ngaBBox

    def get_progress_val(self):
        return self.progressBarValue

    def get_uncertainties_per_bscan(self):
        return self.layerSegmenter.get_uncertainties_per_bscan()

    def get_edited_layers(self):
        return self.editedLayers

    def get_GT_drusen(self):
        """
        Read ground truth for the layer segmentation. Useful for the software
        evaluation.
        """
        if not self.GTdrusen is None:
            return
        scanPath = os.path.join(self.scanPath, 'GTdrusen')
        showDrusenOnScan = False
        d2 = [f for f in listdir(scanPath) if isfile(join(scanPath, f))]
        rawstack = list()
        ind = list()
        rawStackDict = dict()
        rawSize = ()
        for fi in range(len(d2)):
            filename = os.path.join(scanPath, d2[fi])
            ftype = d2[fi].split('-')[-1]
            if ftype == 'drusen.png':
                ind.append(int(d2[fi].split('-')[0]))

                raw = io.imread(filename)
                rawSize = raw.shape
                rawStackDict[ind[-1]] = raw
        if len(rawSize) > 0:
            rawstack = np.empty((rawSize[0], rawSize[1], len(ind)))
            keys = rawStackDict.keys()
            keys.sort()
            i = 0
            for k in keys:
                rawstack[:, :, i] = rawStackDict[k]
                if showDrusenOnScan:
                    y, x = np.where(rawstack[:, :, i] > 0)
                    self.scans[y, x, i] = 255
                i += 1

            self.GTdrusen = np.copy(rawstack)
            self.compute_distances_drusen()

    def get_GT_layers(self):
        """
        Read ground truth for the layer segmentation. Useful for the software
        evaluation.
        """
        if not self.GTlayers is None:
            return
        scanPath = os.path.join(self.scanPath, 'GTlayers')
        showLayersOnScan = False
        d2 = [f for f in listdir(scanPath) if isfile(join(scanPath, f))]
        rawstack = list()
        ind = list()
        rawStackDict = dict()
        rawSize = ()
        for fi in range(len(d2)):
            filename = os.path.join(scanPath, d2[fi])
            ftype = d2[fi].split('-')[-1]
            if ftype == 'BinSeg.tif':
                ind.append(int(d2[fi].split('-')[0]))

                raw = io.imread(filename)
                rawSize = raw.shape
                rawStackDict[ind[-1]] = raw
        if len(rawSize) > 0:
            rawstack = np.empty((rawSize[0], rawSize[1], len(ind)))
            keys = rawStackDict.keys()
            keys.sort()
            i = 0
            for k in keys:
                rawstack[:, :, i] = rawStackDict[k]
                if showLayersOnScan:
                    y, x = np.where(rawstack[:, :, i] > 0)
                    self.scans[y, x, i] = 255
                i += 1

            self.GTlayers = np.copy(rawstack)
            self.compute_distances()

    def get_distances(self):
        return self.overallDistance, self.distances

    def get_distances_drusen(self):
        return self.drusenOverallDistance, self.drusenDistances

    def get_layer(self, sliceNumZ):
        return self.layers[:, :, sliceNumZ]

    def get_layers(self):
        """
        Find the layer segmentation for the RPE and BM layers.
        """
        if self.layers is None:
            try:
                self.layers = self.load_layers()
            except:
                self.layers, self.probmaps = self.compute_layers()

        if self.layerSegmenter is None:
            self.layerSegmenter = deeplearning.DeepLearningLayerSeg(self)

        if self.rpeCSPs is None:
            self.rpeCSPs = [None] * self.layers.shape[2]
        if self.bmProbable is None:
            self.bmProbable = [None] * self.layers.shape[2]
        for i in range(self.layers.shape[2]):
            e = dict()
            e['RPE'] = False
            e['BM'] = False
            self.editedLayers.append(e)
        if self.evaluateLayers:
            self.get_GT_layers()

        return self.layers

    def get_current_path(self):
        return self.scanPath

    def get_drusen(self, hfilter=0):
        """
        Read the drusen segmentation maps from the disk if exists, otherwise,
        compute them automatically from the retinal layer segmentations.
        """
        if self.drusen is None:
            try:
                logger.debug('Load drusen')
                self.drusen = self.load_drusen(hfilter)
            except:
                logger.debug('Compute drusen')
                self.drusen = self.compute_drusen()

        return self.drusen

        """
        if self.drusen is not None:
            # TODO: Prompt the user if he wants to recompute / load drusen or cancel
            pass



        if self.scanPath != '' and self.scanPath is not None:
            drusenPath=os.path.join(self.scanPath,'drusen')
            if os.path.exists(drusenPath):
                logger.debug('Read drusen from disk...')
                self.drusen = self.get_drusen_from_path(drusenPath, hfilter)
                return self.drusen

        logger.debug('Start to compute drusen...')
        self.drusen = self.compute_drusen()

        return self.drusen
        """

    def get_enface(self):
        """
        Read or create the enface projection image.
        """
        try:
            self.enface = io.imread(os.path.join(self.scanPath, "enface.png"))
            self.enfaceBBox = self.get_bbox_from_file_reticular(
                os.path.join(self.scanPath, 'reticular-drusen', 'reticular-drusen-bounding-box.txt'))

        except:
            if self.enface is None:
                self.controller.show_progress_bar()
                self.get_layers()
                projection, masks = self.produce_drusen_projection_image(useWarping=True)
                projection /= np.max(projection) if np.max(projection) != 0.0 else 1.0
                self.enface = (projection * 255).astype(int)  #
                self.enfaceBBox = dict()
                # misc.imsave(os.path.join(self.scanPath,"enface.png"),self.enface)
                self.controller.set_progress_bar_value(100)
                QtGui.QApplication.processEvents()
                self.controller.hide_progress_bar()

        return self.enface, self.enfaceBBox

    def get_enface_drusen(self, recompute=True):
        """
        Compute the enface projection for the drusen segmentation maps.
        """
        if (not self.drusen is None) and recompute:
            self.enfaceDrusen = (np.sum(self.drusen, axis=0) > 0).astype(int).T * 255
        return self.enfaceDrusen

    def get_RPE_layer(self, seg_img):
        """ Return indices of the RPE layer

        The RPE layer is encoded by either 170 or 255
        :param seg_img: Array holding the layer segmentation
        :return: Indices of the RPE layer (y, x)

        >>> get_RPE_layer(np.array([[170, 0, 0],[0, 255, 0],[0,0,127]]))
        (array([0, 1]), array([0, 1]))
        """
        y, x = np.where(np.isin(seg_img, [170, 255]))
        return y, x

    def get_BM_layer(self, seg_img):
        """ Return indices of the BM layer

        The BM layer is encoded by 170, 85, or 127
        :param seg_img: Array holding the layer segmentation
        :return: Indices of the BM layer (y, x)

        >>> get_BM_layer(np.array([[255, 0, 85],[0, 170, 0],[127,0,255]]))
        (array([0, 1, 2]), array([2, 1, 0]))
        """
        y, x = np.where(np.isin(seg_img, [170, 85, 127]))
        return y, x

    def get_RPE_location(self, seg_img):
        y = []
        x = []
        tmp = np.copy(seg_img)
        if np.sum(seg_img) == 0.0:
            return y, x
        if len(np.unique(tmp)) == 4:
            tmp2 = np.zeros(tmp.shape)
            tmp2[np.where(tmp == 170)] = 255
            tmp2[np.where(tmp == 255)] = 255
            y, x = np.where(tmp2 == 255)

        else:
            y, x = np.where(tmp == 255)
        return y, x

    def get_BM_location(self, seg_img):
        y = []
        x = []
        tmp = np.copy(seg_img)
        if np.sum(seg_img) == 0.0:
            return y, x
        tmp2 = np.zeros(tmp.shape)
        tmp2[np.where(tmp == 170)] = 255
        tmp2[np.where(tmp == 85)] = 255
        tmp2[np.where(tmp == 127)] = 255
        y, x = np.where(tmp2 == 255)
        return y, x

    def get_label_of_largest_component(self, labels):
        size = np.bincount(labels.ravel())
        largest_comp_ind = size.argmax()
        return largest_comp_ind

    def get_druse_info(self):
        return self.cx, self.cy, self.area, self.height, self.volume, \
               self.largeR, self.smallR, self.theta

    ## Setters

    def set_slice_edited(self, sliceNum, layerName, status):
        if layerName == 'RPE':
            self.editedLayers[sliceNum - 1]['RPE'] = status
        elif layerName == 'BM':
            self.editedLayers[sliceNum - 1]['BM'] = status

    def set_evaluation_schemes(self, evaluateLayers, evaluateDrusen):
        self.evaluateLayers = evaluateLayers
        self.evaluateDrusen = evaluateDrusen



    def compute_layers(self):
        """

        :return:
        """
        # Use Caffe to create initial layer segmentation
        self.layerSegmenter = deeplearning.DeepLearningLayerSeg(self)
        # TODO: Function returns layers and sets probMaps. Make this explicit
        layers = self.layerSegmenter.get_layer_seg_from_deepnet(self.scans)
        layers[:, :, :] = self.convert_indices(layers[:, :, :])
        probmaps = np.transpose(self.probmaps, (1, 0, 2, 3)).astype('float16')
        self.probmaps = probmaps

        progressVal = self.get_progress_val()
        self.set_progress_val(progressVal + 2)
        self.update_progress_bar()

        # try:
        #    self.certainSlices = list(np.where(np.loadtxt(os.path.join(probPath, \
        #                                                               'prob-entropy.txt')) == 0.05)[0])
        # except:
        #    self.certainSlices = list()

        self.progressBarValue = 100
        self.controller.set_progress_bar_value(self.progressBarValue)
        self.controller.hide_progress_bar()
        QtGui.QApplication.processEvents()
        return layers, probmaps

    def compute_drusen(self):
        if self.drusenSegmenter is None:
            logger.debug('Setting drusenSegmenter')
            self.drusenSegmenter = drusenextractor.DrusenSeg(self.controller)

        self.drusen = self.drusenSegmenter.get_drusen_seg_polyfit(self.layers)*255

        return self.drusen



    def hrf_exist_in_slice(self, sliceNum):
        s = np.sum(self.hrfs[:, :, sliceNum - 1])
        if s > 0:
            return True
        else:
            return False

    def change_layers_format_for_GUI(self):
        for s in range(self.layers.shape[2]):
            if 170 in self.layers[:, :, s]:
                # Join point
                y, x = np.where(self.layers[:, :, s] == 255)
                # RPE
                yrpe, xrpe = np.where(self.layers[:, :, s] == 170)
                # BM
                ybm, xbm = np.where(self.layers[:, :, s] == 85)
                self.layers[yrpe, xrpe, s] = 255
                self.layers[y, x, s] = 170
                self.layers[ybm, xbm, s] = 127

    def change_layers_format_for_saving(self):
        layers = np.copy(self.layers)
        for s in range(layers.shape[2]):
            if 170 in layers[:, :, s]:
                # Join point
                y, x = np.where(layers[:, :, s] == 170)
                # RPE
                yrpe, xrpe = np.where(layers[:, :, s] == 255)
                # BM
                ybm, xbm = np.where(layers[:, :, s] == 127)
                layers[yrpe, xrpe, s] = 170
                layers[ybm, xbm, s] = 85
                layers[y, x, s] = 255
        return layers

    def interpolate_layer_in_region(self, reg, reg2, by, ty, bx, tx, topLeftX, \
                                    topLeftY, polyDegree, layerName, sliceNum):
        if self.drusenSegmenter is None:
            self.drusenSegmenter = drusenextractor.DrusenSeg(self.controller)

        info = dict()
        info['layers'] = np.copy(reg)
        if not self.probmaps is None:
            info['probMaps'] = np.copy(self.probmaps[:, :, :, sliceNum - 1])
            info['uncertainties'] = self.layerSegmenter.get_uncertainties(sliceNum - 1)
        else:
            info['probMaps'] = None
            info['uncertainties'] = None
        info['topLeftX'] = topLeftX
        info['topLeftY'] = topLeftY
        info['prevStatus'] = self.editedLayers[sliceNum - 1][layerName]
        interpolated = self.drusenSegmenter.interpolate_layer_in_region(reg, reg2, \
                                                                        by, ty, bx, tx, polyDegree, layerName)
        interpolated[np.where(interpolated == 85)] = 127.

        eps = 1.e-10
        if not self.probmaps is None:
            if layerName == 'RPE':
                y, x = self.drusenSegmenter.get_RPE_location(interpolated)

                self.probmaps[:, x + topLeftX, 3, sliceNum - 1] = eps
                self.probmaps[:, x + topLeftX, 2, sliceNum - 1] = eps
                self.probmaps[y + topLeftY, x + topLeftX, 2, sliceNum - 1] = 1.
            elif layerName == 'BM':
                y, x = self.drusenSegmenter.get_BM_location(interpolated)
                self.probmaps[:, x + topLeftX, 3, sliceNum - 1] = eps
                self.probmaps[:, x + topLeftX, 1, sliceNum - 1] = eps
                self.probmaps[y + topLeftY, x + topLeftX, 1, sliceNum - 1] = 1.

        info['reg'] = np.copy(interpolated)
        self.editedLayers[sliceNum - 1][layerName] = True
        return info

    def interpolate_layer_in_region_using_info(self, info, sliceNumZ, layerName):
        reg = info['layers']

        self.layers[info['topLeftY']:info['topLeftY'] + reg.shape[0], \
        info['topLeftX']:info['topLeftX'] + reg.shape[1], sliceNumZ] = info['layers']
        self.editedLayers[sliceNumZ][layerName] = info['prevStatus']
        if not info['probMaps'] is None:
            self.probmaps[:, :, :, sliceNumZ] = info['probMaps']
            self.layerSegmenter.set_uncertainties(info['uncertainties'], sliceNumZ)
            self.controller.set_uncertainties(info['uncertainties'], sliceNumZ)

    def convert_indices(self, s):
        ids = np.unique(s)
        if max(ids) < 5:
            if len(ids) == 4:
                s[np.where(s == 3)] = 255.
                s[np.where(s == 2)] = 170.
                s[np.where(s == 1)] = 127.
            else:
                s[np.where(s == 2)] = 255.
                s[np.where(s == 1)] = 127.
        return s

    def update_progress_bar(self):
        self.controller.set_progress_bar_value(self.progressBarValue)
        QtGui.QApplication.processEvents()

    def compute_distance(self, gt, pr):
        distMeasure = 'RPEBM'
        gtRpe = np.sum((np.cumsum((gt > 128).astype(float), axis=0) > 0).astype(float), axis=0)
        gtBM = np.sum((np.cumsum((np.logical_and(gt < 128, gt != 0)).astype(float). \
                                 astype(float), axis=0) > 0).astype(float), axis=0)

        prRpe = np.sum((np.cumsum((pr > 128).astype(float).astype(float), axis=0) > 0).astype(float), axis=0)
        prBM = np.sum((np.cumsum((np.logical_and(pr < 128, pr != 0)).astype(float). \
                                 astype(float), axis=0) > 0).astype(float), axis=0)

        distRpe = np.abs(gtRpe - prRpe)
        distBM = np.abs(gtBM - prBM)

        distRpe[np.where(distRpe > 100)] = 0.
        distBM[np.where(distBM > 100)] = 0.

        rpeMax10 = distRpe[np.where(distRpe > np.percentile(distRpe, 90))]
        bmMax10 = distBM[np.where(distBM > np.percentile(distBM, 90))]

        if distMeasure == 'onlyBM':
            nom = np.sum(bmMax10)
            denom = np.sum((bmMax10 > 0).astype(float))
        elif distMeasure == 'onlyRPE':
            nom = np.sum(rpeMax10)
            denom = np.sum((rpeMax10 > 0).astype(float))
        else:
            nom = np.sum(rpeMax10) + np.sum(bmMax10)
            denom = np.sum((rpeMax10 > 0).astype(float)) + np.sum((bmMax10 > 0).astype(float))
        if denom == 0:
            return 0
        return nom / denom

    def compute_distance_OR(self, gt, pr):

        gtRpe = (np.cumsum((gt > 128).astype(float), axis=0) > 0).astype(float)
        gtBM = (np.cumsum((np.logical_and(gt < 128, gt != 0)).astype(float). \
                          astype(float), axis=0) > 0).astype(float)
        falsePos = np.sum(gtBM, axis=0)
        falsePos = np.where(falsePos == 0)
        gtarea = gtRpe - gtBM
        gtarea[np.where(gtarea < 0)] = 0.
        for fp in falsePos:
            gtarea[:, fp] = 0.
        prRpe = (np.cumsum((pr > 128).astype(float).astype(float), axis=0) > 0).astype(float)
        prBM = (np.cumsum((np.logical_and(pr < 128, pr != 0)).astype(float). \
                          astype(float), axis=0) > 0).astype(float)
        falsePos = np.sum(prBM, axis=0)
        falsePos = np.where(falsePos == 0)
        prarea = prRpe - prBM
        prarea[np.where(prarea < 0)] = 0.
        for fp in falsePos:
            prarea[:, fp] = 0.
        denom = float(np.sum((np.logical_or(gtarea > 0, prarea > 0) > 0).astype('float')))
        nom = float(np.sum((np.logical_and(gtarea > 0, prarea > 0) > 0).astype('float')))
        OR = nom / denom if denom > 0 else 1.0
        return OR

    def c(self, sliceNumZ):
        gt = self.GTlayers[:, :, sliceNumZ]
        pr = self.layers[:, :, sliceNumZ]
        dist = self.compute_distance_OR(gt, pr)
        self.distances[sliceNumZ] = dist
        self.overallDistance = sum(self.distances) / float(len(self.distances))
        return self.overallDistance, self.distances

    def compute_distances(self):
        sumD = 0
        for s in range(self.layers.shape[2]):
            d = self.compute_distance(self.GTlayers[:, :, s], self.layers[:, :, s])
            self.distances.append(d)
            sumD += d
        self.overallDistance = sumD / float(len(self.distances))

    def compute_distance_drusen(self, gt, pr):
        prr = np.sum((pr > 0).astype(int))
        gtt = np.sum((gt > 0).astype(int))
        dist = np.abs(gtt - prr)
        return dist

    def compute_distance_drusen_OR(self, gt, pr):
        gt = gt.astype('float')
        pr = pr.astype('float')
        gt[gt > 0] = 1.0
        gt[gt <= 0] = 0.0
        pr[pr > 0] = 1.0
        pr[pr <= 0] = 0.0

        denom = float(np.sum((np.logical_or(gt > 0, pr > 0) > 0).astype('float')))
        nom = float(np.sum((np.logical_and(gt > 0, pr > 0) > 0).astype('float')))
        OR = nom / denom if denom > 0 else 1.0
        return OR

    def update_distance(self, sliceNumZ):
        gt = self.GTlayers[:, :, sliceNumZ]
        pr = self.layers[:, :, sliceNumZ]
        dist = self.compute_distance(gt, pr)
        self.distances[sliceNumZ] = dist
        self.overallDistance = sum(self.distances) / float(len(self.distances))
        return self.overallDistance, self.distances

    def update_distance_drusen(self, sliceNumZ):
        gt = self.GTdrusen[:, :, sliceNumZ]
        pr = self.drusen[:, :, sliceNumZ]
        dist = self.compute_distance_drusen_OR(gt, pr)
        self.drusenDistances[sliceNumZ] = dist
        self.drusenOverallDistance = sum(self.drusenDistances) / float(len(self.drusenDistances))
        return self.drusenOverallDistance, self.drusenDistances

    def compute_distances_drusen(self):
        sumD = 0
        for s in range(self.drusen.shape[2]):
            d = self.compute_distance_drusen_OR(self.GTdrusen[:, :, s], self.drusen[:, :, s])
            self.drusenDistances.append(d)
            sumD += d
        self.drusenOverallDistance = sumD / float(len(self.drusenDistances))

    def probmaps_does_exist(self):
        if self.probmaps is not None:
            return True
        else:
            return False



    def compute_uncertainties(self):
        self.layerSegmenter.compute_segmentation_uncertainty(self.certainSlices)

    def accept_suggested_segmentation(self, layerName, sliceNumZ, smoothness, \
                                      uncertaintyType, extent, csps):
        info = dict()
        info['layer'] = np.copy(self.layers[:, :, sliceNumZ])

        if not self.probmaps is None:
            info['probMaps'] = np.copy(self.probmaps[:, :, :, sliceNumZ])
            info['uncertainties'] = self.layerSegmenter.get_uncertainties(sliceNumZ)
        else:
            info['probMaps'] = None
            info['uncertainties'] = None
        info['prevStatus'] = self.editedLayers[sliceNumZ][layerName]
        if layerName == 'RPE':
            self.rpeCSPs[sliceNumZ] = csps
            self.rpeSuggestExtent = extent
            layer = self.get_probable_RPE(sliceNumZ, smoothness)
            info['rpeCSPs'] = self.rpeCSPs[sliceNumZ]
            info['suggestExtent'] = extent
            if not csps is None:
                for p in self.rpeCSPs[sliceNumZ]:
                    i = p[0]
                    j = p[1]
                    self.probmaps[:, j, 3, sliceNumZ] = 0
                    self.probmaps[:, j, 2, sliceNumZ] = 0
                    self.probmaps[i, j, 2, sliceNumZ] = 1.
            rpeImg, bmImg = self.decompose_into_RPE_BM_images(self.layers[:, :, sliceNumZ])
            layer = self.combine_RPE_BM_images(layer.astype(int) * 255, bmImg)
        if layerName == 'BM':
            self.bmSuggestExtent = extent
            layer = self.get_probable_BM(sliceNumZ, uncertaintyType)
            rpeImg, bmImg = self.decompose_into_RPE_BM_images(self.layers[:, :, sliceNumZ])
            layer = self.combine_RPE_BM_images(rpeImg, layer.astype(int) * 255)
            info['suggestExtent'] = self.bmSuggestExtent

        #if (self.splineKnots is not None
        #        and self.splineKnots[sliceNumZ] is not None
        #        and layerName in self.splineKnots[sliceNumZ].keys()
        #        and self.splineKnots[sliceNumZ][layerName] is not None):
        #    self.splineKnots[sliceNumZ][layerName] = None
        self.layers[:, :, sliceNumZ] = np.copy(layer)
        self.editedLayers[sliceNumZ][layerName] = True
        return info

    def accept_suggested_segmentation_using_info(self, info, sliceNumZ, layerName, extent, csps):
        if layerName == 'RPE':
            self.rpeSuggestExtent = info['suggestExtent']
            self.rpeCSPs[sliceNumZ] = info['rpeCSPs']
            self.probmaps[:, :, :, sliceNumZ] = np.copy(info['probMaps'])
        elif layerName == 'BM':
            self.bmSuggestExtent = info['suggestExtent']
        self.layers[:, :, sliceNumZ] = np.copy(info['layer'])
        self.editedLayers[sliceNumZ][layerName] = info['prevStatus']

        if not info['probMaps'] is None:
            self.probmaps[:, :, :, sliceNumZ] = info['probMaps']
            self.layerSegmenter.set_uncertainties(info['uncertainties'], sliceNumZ)
            self.controller.set_uncertainties(info['uncertainties'], sliceNumZ)

    def get_probable_RPE(self, sliceNumZ, smoothness):
        rpelayer = None
        if not self.rpeCSPs[sliceNumZ] is None and self.rpeSuggestExtent > 0:
            rpelayer = self.layerSegmenter. \
                update_probability_image_multi_points(self.rpeCSPs[sliceNumZ], sliceNumZ, smoothness)
        return rpelayer

    def get_probable_BM(self, sliceNumZ, uncertaintyType):
        bmlayer = None
        if self.bmSuggestExtent > 0:
            bmlayer = self.layerSegmenter. \
                estimate_bm_3d(sliceNumZ, self.bmSuggestExtent, uncertaintyType)
        return bmlayer

    def get_probable_layers(self, sliceNumZ, smoothness, uncertaintyType='entropy', showSuggestedSegmentation=None):
        # Check if there are suggested CSPs for the RPE layer
        rpelayer = self.get_probable_RPE(sliceNumZ, smoothness)
        bmlayer = self.get_probable_BM(sliceNumZ, uncertaintyType)

        rpelayer = rpelayer if self.rpeSuggestShow else None
        bmlayer = bmlayer if self.bmSuggestShow else None

        if rpelayer is None and bmlayer is None:
            return None

        if rpelayer is None:
            rpelayer = np.zeros(bmlayer.shape)
            rpelayer = rpelayer * 2 + bmlayer
            return rpelayer
        elif bmlayer is None:
            bmlayer = np.zeros(rpelayer.shape)
            rpelayer = rpelayer * 2 + bmlayer
            return rpelayer
        else:
            rpelayer = rpelayer * 2 + bmlayer
            return rpelayer

        return None

    def update_cost_rpe(self, i, j, sliceNumZ, smoothness):
        info = dict()
        self.get_probmaps()
        info['probMaps'] = np.copy(self.probmaps[:, :, :, sliceNumZ])
        info['layers'] = np.copy(self.layers[:, :, sliceNumZ])
        info['uncertainties'] = self.layerSegmenter.get_uncertainties(sliceNumZ)
        info['smoothness'] = smoothness
        self.layers[:, :, sliceNumZ] = self.layerSegmenter. \
            update_probability_image(i, j, sliceNumZ, 'RPE', smoothness)
        info['prevStatus'] = self.editedLayers[sliceNumZ]['RPE']
        self.editedLayers[sliceNumZ]['RPE'] = True
        info['CSP'] = copy.copy(self.rpeCSPs)
        # ProposeSegmentation for neighbors
        # Find shortest path in A-scan direction
        if self.rpeSuggestExtent > 0:
            shortestP = self.layerSegmenter.compute_shortest_path_in_A_scan_direction(i, j, sliceNumZ, smoothness,
                                                                                      self.rpeSuggestExtent)
            startLayer = max(0, sliceNumZ - self.rpeSuggestExtent)
            endLayer = min(self.layers.shape[2] - 1, sliceNumZ + self.rpeSuggestExtent)
            for i in range(startLayer, endLayer + 1):
                if i == sliceNumZ:
                    continue
                if self.rpeCSPs[i] is None:
                    self.rpeCSPs[i] = set()
                y = np.where(shortestP[:, i] > 0)[0][0]
                self.rpeCSPs[i].add((y, j))
            # If current layer has probable points, delete them, as it is updated
            if not self.rpeCSPs[sliceNumZ] is None:
                self.rpeCSPs[sliceNumZ] = None
        return info

    def update_cost_rpe_using_info(self, info, sliceNumZ):
        self.probmaps[:, :, :, sliceNumZ] = info['probMaps']
        self.layers[:, :, sliceNumZ] = info['layers']
        self.layerSegmenter.set_uncertainties(info['uncertainties'], sliceNumZ)
        self.layerSegmenter.set_yLength(info['smoothness'])
        self.controller.set_uncertainties(info['uncertainties'], sliceNumZ)
        self.editedLayers[sliceNumZ]['RPE'] = info['prevStatus']
        self.rpeCSPs = copy.copy(info['CSP'])

    def update_cost_bm(self, i, j, sliceNumZ, smoothness):
        info = dict()
        self.get_probmaps()
        info['probMaps'] = np.copy(self.probmaps[:, :, :, sliceNumZ])
        info['layers'] = np.copy(self.layers[:, :, sliceNumZ])
        info['uncertainties'] = self.layerSegmenter.get_uncertainties(sliceNumZ)
        info['smoothness'] = smoothness
        self.layers[:, :, sliceNumZ] = self.layerSegmenter. \
            update_probability_image(i, j, sliceNumZ, 'BM', smoothness)

        info['prevStatus'] = self.editedLayers[sliceNumZ]['BM']
        self.editedLayers[sliceNumZ]['BM'] = True
        return info

    def update_cost_bm_using_info(self, info, sliceNumZ):
        self.probmaps[:, :, :, sliceNumZ] = info['probMaps']
        self.layers[:, :, sliceNumZ] = info['layers']
        self.layerSegmenter.set_uncertainties(info['uncertainties'], sliceNumZ)
        self.layerSegmenter.set_yLength(info['smoothness'])
        self.editedLayers[sliceNumZ]['BM'] = info['prevStatus']
        self.controller.set_uncertainties(info['uncertainties'], sliceNumZ)

    def walklevel(self, some_dir, level=1):
        some_dir = some_dir.rstrip(os.path.sep)
        assert os.path.isdir(some_dir)
        num_sep = some_dir.count(os.path.sep)
        for root, dirs, files in os.walk(some_dir):
            yield root, dirs, files
            num_sep_this = root.count(os.path.sep)
            if num_sep + level <= num_sep_this:
                del dirs[:]

    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [self.atoi(c) for c in re.split('(\d+)', text)]

    def get_xml_files(self, directory):
        '''This function returnes a list with all the xml files contained in the directory'''
        result = []
        for subdir, dirs, files in os.walk(directory):
            for file in files:
                filepath = subdir + os.sep + file
                if filepath.endswith(".xml"):
                    result.append(filepath)
        return result

    def convert_heidelberg_eng_format(self, directory, layer):
        '''This function takes a directory containing different folders with OCT scan information as input and
        constructs for each image in the different subfolders a corresponding image displaying only the layer
        specified in the input.'''

        path = str(directory + os.path.sep + 'layers')
        # If layers are already extracted, skip
        if os.path.exists(path) and len([f for f in listdir(path) if isfile(join(path, f))]) > 0:
            return
        # Create a new folder to store the image
        if not os.path.exists(path):
            os.makedirs(path)
        os.chdir(path)

        # Get the name of the subfolders in the original folder
        subfolders = [x[0].split(os.path.sep)[-1] for x in os.walk(directory)][1:]

        # Create the same subfolders in the new directory
        for folder in subfolders:
            p = path + os.path.sep + folder
            if not os.path.exists(p) and '-' in folder:
                os.makedirs(p)

        # Get the path to all the xml_files
        xml_files = self.get_xml_files(directory)
        if len(xml_files) == 0:
            return
        # Get the path where to store the xml_file
        img_paths = []
        for p in xml_files:
            img_paths.append(path + os.path.sep)

        imgScalesInMicroMeter = dict()
        # Construct the images
        for index in range(len(xml_files)):
            ftext = open(xml_files[index], 'r')
            textToParse = ftext.read()

            textToParse = textToParse.replace('\xe4', 'ae')
            textToParse = textToParse.replace('\xf6', 'oe')
            textToParse = textToParse.replace('\xfc', 'ue')
            textToParse = textToParse.replace('\xdf', 'ss')
            textToParse = textToParse.replace('\xc4', 'Ae')
            textToParse = textToParse.replace('\xd6', 'Oe')
            textToParse = textToParse.replace('\xdc', 'Ue')
            textToParse = textToParse.replace('\xdf', 'Ss')
            textToParse = textToParse.replace('\xe9', 'Ss')
            tmp = open(path + os.path.sep + 'tmp.xml', 'w')
            tmp.write(textToParse)
            tmp.close()

            # Read in the XML file and store the root in root
            tree = ET.parse(path + os.path.sep + 'tmp.xml')
            root = tree.getroot()

            os.remove(path + os.path.sep + 'tmp.xml')

            # Search for the image information in the file
            for img in root.iter('Image'):

                width = int(img.find('OphthalmicAcquisitionContext').find('Width').text)
                height = int(img.find('OphthalmicAcquisitionContext').find('Height').text)

                # Find the image name and ID and store those variables
                file_path = img.find('ImageData').find('ExamURL').text
                file_name = file_path.split("\\")[-1].split('.')[0]
                img_id = int(img.find('ID').text)

                imgInit = False
                layerId = 1
                # Search for the intensity values
                for seg in img.iter('Segmentation'):
                    for seglin in seg.iter('SegLine'):
                        if seglin.find('Name').text in layer:

                            # Store the intensity values as np array
                            y_values = np.asarray(seglin.find('Array').text.split())
                            y_values = y_values.astype(np.float)

                            if not imgInit:
                                # Create the output array
                                img_layer = np.zeros((height, width))
                                imgInit = True

                            layerId = layer.index(seglin.find('Name').text) + 1

                            # Add the intensity values to the output array
                            for i in range(img_layer.shape[1]):
                                j = int(round(y_values[i]))
                                if j < len(y_values):  # Values outside of this range are missing values
                                    img_layer[j, i] = layerId + img_layer[j, i]

                                    # check discontinuity
                                    # get the previous y value
                                    previous_y = int(round(y_values[i - 1]))
                                    # check if it is in the range of the image
                                    if i > 0 and previous_y < len(y_values):
                                        # while there is discontinuity, add pixels
                                        while abs(previous_y - j) > 1:
                                            if previous_y > j:
                                                img_layer[j + 1, i] = layerId + img_layer[j + 1, i]
                                                j += 1
                                            else:
                                                img_layer[j - 1, i] = layerId + img_layer[j - 1, i]
                                                j -= 1

                # Create the image name and save it if it exists
                img_name = str(img_id) + '-layers.png'
                os.chdir(img_paths[index])
                if img_id != 0:  # If image is B-scan
                    # Find scale information that are in milimeter
                    scaleX = float(img.find('OphthalmicAcquisitionContext').find('ScaleX').text)
                    scaleY = float(img.find('OphthalmicAcquisitionContext').find('ScaleY').text)
                    posY = float(img.find('OphthalmicAcquisitionContext').find('Start').find('Coord').find('Y').text)
                    imgScalesInMicroMeter[img_id] = [scaleX, scaleY, posY]
                try:

                    img_layer = self.convert_indices(img_layer)
                    #                    cv2.imwrite(img_name,img_layer)
                    misc.imsave(img_name, img_layer)
                except:
                    # In case of fundus image, skip the rest
                    if img_id == 0:
                        continue

                # Rename the input image:
                os.chdir(directory + os.path.sep + img_paths[index].split(os.path.sep)[-1])
                new_name = str(img_id) + '-' + file_name + '-Input.tif'
                old_name = file_name + '.tif'
                try:
                    os.rename(old_name, new_name)
                except:
                    continue
        scalesX = list()
        scalesY = list()
        posY = list()
        for k in sorted(imgScalesInMicroMeter.keys()):
            a, b, c = imgScalesInMicroMeter[k]
            scalesX.append(a)
            scalesY.append(b)
            posY.append(c)
        posY = np.asarray(posY)
        scalesZ = np.abs(posY[1:] - posY[:-1])

        self.hx = np.mean(np.asarray(scalesX)) * 1000.
        self.hy = np.mean(np.asarray(scalesY)) * 1000.
        self.hz = np.mean(np.asarray(scalesZ)) * 1000.

        fnames = [f for f in listdir(directory) if isfile(join(directory, f))]

        for f in fnames:
            ftype = f.split('-')[-1]
            if ftype != "Input.tif":
                fFormat = f.split('.')[-1]
                if fFormat == 'tif':
                    #                    print f
                    os.rename(f, "fundus.tif")

    def extract_scale(self, directory):
        '''This function takes a directory containing different folders with OCT scan information as input and
        constructs for each image in the different subfolders a corresponding image displaying only the layer
        specified in the input.'''

        # Get the path to all the xml_files
        xml_files = self.get_xml_files(directory)
        if len(xml_files) == 0:
            return

        imgScalesInMicroMeter = dict()
        # Construct the images
        for index in range(len(xml_files)):
            ftext = open(xml_files[index], 'r')
            textToParse = ftext.read()

            textToParse = textToParse.replace('\xe4', 'ae')
            textToParse = textToParse.replace('\xf6', 'oe')
            textToParse = textToParse.replace('\xfc', 'ue')
            textToParse = textToParse.replace('\xdf', 'ss')
            textToParse = textToParse.replace('\xc4', 'Ae')
            textToParse = textToParse.replace('\xd6', 'Oe')
            textToParse = textToParse.replace('\xdc', 'Ue')
            textToParse = textToParse.replace('\xdf', 'Ss')
            textToParse = textToParse.replace('\xe9', 'Ss')
            tmp = open(directory + os.path.sep + 'tmp.xml', 'w')
            tmp.write(textToParse)
            tmp.close()

            # Read in the XML file and store the root in root
            tree = ET.parse(directory + os.path.sep + 'tmp.xml')
            root = tree.getroot()

            os.remove(directory + os.path.sep + 'tmp.xml')

            # Search for the image information in the file
            for img in root.iter('Image'):

                img_id = int(img.find('ID').text)

                if img_id != 0:  # If image is B-scan
                    # Find scale information that are in milimeter
                    scaleX = float(img.find('OphthalmicAcquisitionContext').find('ScaleX').text)
                    scaleY = float(img.find('OphthalmicAcquisitionContext').find('ScaleY').text)
                    posY = float(img.find('OphthalmicAcquisitionContext').find('Start').find('Coord').find('Y').text)
                    imgScalesInMicroMeter[img_id] = [scaleX, scaleY, posY]

        scalesX = list()
        scalesY = list()
        posY = list()
        for k in sorted(imgScalesInMicroMeter.keys()):
            a, b, c = imgScalesInMicroMeter[k]
            scalesX.append(a)
            scalesY.append(b)
            posY.append(c)
        posY = np.asarray(posY)
        scalesZ = np.abs(posY[1:] - posY[:-1])

        self.hx = np.mean(np.asarray(scalesX)) * 1000.
        self.hy = np.mean(np.asarray(scalesY)) * 1000.
        self.hz = np.mean(np.asarray(scalesZ)) * 1000.

    def rgb_to_gray(self, img):
        if len(img.shape) == 3:
            return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        if len(img.shape) == 2:
            return img

    def import_vol_from(self, filepath):
        """ Import Heidelberg Engineering OCT raw files (.vol ending)

        :param filepath:
        :return:
        """

        file_header = iovol.get_vol_header(filepath)
        # SLO is Scanning Laser Ophthalmoskopie -> NIR
        slo = iovol.get_slo_image(filepath, file_header)
        b_hdrs, b_seglines, b_scans = iovol.get_bscan_images(filepath, file_header, improve_constrast='hist_match')

        self.slo = slo

        self.scanIDs = list(range(1, file_header['NumBScans'] + 1))
        self.scans = b_scans[:, :, :]
        self.numSlices = self.scans.shape[2]
        self.width = self.scans.shape[1]
        self.height = self.scans.shape[0]

        if self.numSlices > 50:
            self.zRate = 2
            self.bResolution = 'high'
        else:
            self.zRate = 13
            self.bResolution = 'low'

        # Invert if too bright
        if np.mean(self.scans) > 180:
            self.scans = 255 - self.scans

    def read_scan_from(self, scanPath):
        rawstack = list()
        ind = list()
        rawStackDict = dict()
        rawSize = ()
        idCounter = 1
        genDepth = 1

        # If heidelberg format, make necessary changes
        self.convert_heidelberg_eng_format(scanPath, ['BM', 'RPE'])
        self.extract_scale(scanPath)

        for root, dirs, files in os.walk(scanPath):
            depth = len(scanPath.split(os.path.sep))
            curDepth = len(root.split(os.path.sep))
            if curDepth > depth or genDepth > 1:
                continue
            if len(files) <= 0:
                return

            files.sort(key=self.natural_keys)
            for fname in files:

                if fname == 'enface.png' or fname == 'fundus.tif':
                    continue
                try:
                    ftype = fname.split('-')[-1]
                except:
                    ftype = ""
                if ftype == 'Input.tif':
                    ind.append(int(fname.split('-')[0]))

                    raw = self.rgb_to_gray(imageio.imread(os.path.join(root, fname)))
                    rawSize = raw.shape
                    rawStackDict[ind[-1]] = raw
                else:
                    try:
                        raw = self.rgb_to_gray(imageio.imread(os.path.join(root, fname)))
                    except:
                        print "Warning reading file:" + fname + " is not image."
                        continue
                    ind.append(idCounter)
                    idCounter += 1

                    rawSize = raw.shape
                    rawStackDict[ind[-1]] = raw
            genDepth += 1

        rawstack = np.empty((rawSize[0], rawSize[1], len(ind)))
        keys = rawStackDict.keys()
        keys.sort()
        self.scanIDs = keys
        i = 0
        for k in keys:
            rawstack[:, :, i] = rawStackDict[k]
            i += 1

        self.scans = np.copy(rawstack)
        self.numSlices = self.scans.shape[2]
        self.width = self.scans.shape[1]
        self.height = self.scans.shape[0]

        if self.numSlices > 50:
            self.zRate = 2
            self.bResolution = 'high'
        else:
            self.zRate = 13
            self.bResolution = 'low'
        meanScanIntensity = np.mean(self.scans)
        # Invert if too bright
        if meanScanIntensity > 180:
            self.scans = 255 - self.scans

    def insert_druse_at_with_normal_thickness(self, slices, posY, thickness):
        if len(slices) > 0:
            prevValues = np.copy(self.drusen[:, posY, slices])
            for s in slices:
                layer = self.layers[:, :, s]
                a = np.zeros(layer.shape)
                a[:, posY] = 1.0
                allArea = self.find_area_between_seg_lines(layer) * 255.0
                # Find BM layer, shift it up by thickness, set everything as background under
                bmy, bmx = self.get_BM_location(layer)
                filterMask = np.zeros(allArea.shape)
                bmy = bmy - thickness
                filterMask[bmy, bmx] = 1.0
                filterMask = np.cumsum(filterMask, axis=0)
                filterMask[filterMask > 0] = 1.0
                allArea[filterMask > 0] = 0
                allArea = allArea * a
                y, x = np.where(allArea > 0)
                self.drusen[y, x, s] = 255
            return prevValues

    def set_values_on_enface(self, posY, s, value, thickness):

        values = np.copy(self.drusen[:, posY, s])
        if value == 0:
            self.drusen[:, posY, s] = 0
        else:
            layer = self.layers[:, :, s]
            a = np.zeros(layer.shape)
            a[:, posY] = 1.0
            allArea = self.find_area_between_seg_lines(layer) * 255.0
            # Find BM layer, shift it up by thickness, set everything as background under
            bmy, bmx = self.get_BM_location(layer)
            filterMask = np.zeros(allArea.shape)
            bmy = bmy - thickness
            filterMask[bmy, bmx] = 1.0
            filterMask = np.cumsum(filterMask, axis=0)
            filterMask[filterMask > 0] = 1.0
            allArea[filterMask > 0] = 0
            allArea = allArea * a
            y, x = np.where(allArea > 0)
            self.drusen[y, x, s] = 255
        return values

    def set_values_on_enface_using_vals(self, posY, slices, values):
        self.drusen[:, posY, slices] = np.copy(values)

    def set_values_on_enface_line(self, layers, posY, value, thickness):
        values = np.copy(self.drusen[:, posY, layers])
        if value == 0:
            self.drusen[:, posY, layers] = 0
        else:
            uniqueLayers = np.unique(layers)
            for s in uniqueLayers:
                layer = self.layers[:, :, s]
                a = np.zeros(layer.shape)
                # find corresponding Ys for the current slice
                ind = np.where(layers == s)[0]
                for ii in ind:
                    a[:, posY[ii]] = 1.0
                allArea = self.find_area_between_seg_lines(layer) * 255.0
                # Find BM layer, shift it up by thickness, set everything as background under
                bmy, bmx = self.get_BM_location(layer)
                filterMask = np.zeros(allArea.shape)
                bmy = bmy - thickness
                filterMask[bmy, bmx] = 1.0
                filterMask = np.cumsum(filterMask, axis=0)
                filterMask[filterMask > 0] = 1.0
                allArea[filterMask > 0] = 0
                allArea = allArea * a
                y, x = np.where(allArea > 0)
                self.drusen[y, x, s] = 255
        return values

    def set_values_on_enface_using_vals_line(self, slices, posY, values):
        self.drusen[:, posY, slices] = np.copy(values)

    def remove_druse_at(self, slices, posY):
        if len(slices) > 0:
            prevValues = np.copy(self.drusen[:, posY, slices])
            self.drusen[:, posY, slices] = 0.
            return prevValues

    def insert_druse_at_pos(self, xs, ys, zs):
        self.drusen[xs, ys, zs] = 255

    def insert_druse_at(self, slices, posY, posX):
        if len(slices) > 0:
            self.drusen[:, posY, slices] = posX

    def insert_hrf_at_slice(self, sliceNum, x, y, value):
        if len(x) > 0:
            self.hrfs[x, y, sliceNum - 1] = value

    def insert_ga_at_slice(self, sliceNum, x, y, value):
        if len(x) > 0:
            self.gas[x, y, sliceNum - 1] = value

    def insert_nga_at_slice(self, sliceNum, x, y, value):
        if len(x) > 0:
            self.ngas[x, y, sliceNum - 1] = value

    def instert_druse_at_slice(self, sliceNum, x, y, value):
        if len(x) > 0:
            self.drusen[x, y, sliceNum - 1] = value

    def instert_layer_at_slice(self, sliceNum, x, y, value):
        if len(x) > 0:
            for i in range(len(x)):
                self.layers[int(x[i]), int(y[i]), sliceNum - 1] = int(value[i])

    def switch_BM_format(self, sliceNum):
        l = self.layers[:, :, sliceNum - 1]
        if 170 in l:
            try:
                self.layers[np.where(l == 127), sliceNum - 1] = 85
            except:
                pointsx, pointsy = np.where(l == 127)
                for i in range(len(pointsx)):
                    x = pointsx[i]
                    y = pointsy[i]
                    h, w = self.layers[:, :, sliceNum - 1].shape
                    if x >= 0 and x < h and y >= 0 and y < w:
                        self.layers[x, y, sliceNum - 1] = 85

    def insert_value_at(self, x, y, z, value):
        if len(x) > 0:
            self.drusen[x, y, z] = value

    def delete_drusen_in_region(self, topLeftS, bottomRightS, topLeftY, bottomRightY):
        tmp = np.copy(self.drusen[:, topLeftY:bottomRightY, topLeftS:bottomRightS])
        self.drusen[:, topLeftY:bottomRightY, topLeftS:bottomRightS] = 0.
        return np.where(tmp != self.drusen[:, topLeftY:bottomRightY, topLeftS:bottomRightS])

    def produce_drusen_projection_image(self, useWarping=False):
        height, width, depth = self.scans.shape
        masks = np.zeros(self.scans.shape)
        xys = dict()
        xysn = dict()
        b_scans = self.scans.astype('float')
        projection = np.zeros((b_scans.shape[2], b_scans.shape[1]))
        total_y_max = 0
        progressStep = ((100. / float(b_scans.shape[2])) / 3.) * 2.
        progressValue = 0.
        img_max = np.zeros(b_scans[:, :, 0].shape)
        for i in range(b_scans.shape[2]):
            progressValue += progressStep
            self.controller.set_progress_bar_value(int(progressValue))
            QtGui.QApplication.processEvents()

            b_scan = self.layers[:, :, i]
            y, x = self.get_RPE_layer(b_scan)
            y_n, x_n = self.normal_RPE_estimation(b_scan, useWarping=useWarping)
            xys[i] = [y, x]
            xysn[i] = [y_n, x_n]
            vr = np.zeros((b_scans.shape[1]))
            vr[x] = y
            vn = np.zeros((b_scans.shape[1]))
            vn[x_n] = y_n
            y_diff = np.abs(y - y_n)
            y_max = np.max(y_diff)
            if total_y_max < y_max:
                h, w = b_scan.shape
                y[np.where(y >= h)] = h - 1
                x[np.where(x >= w)] = w - 1
                y_n[np.where(y_n >= h)] = h - 1
                x_n[np.where(x_n >= w)] = w - 1
                img_max.fill(0)
                img_max[y, x] = 255
                img_max[y_n, x_n] = 127
                total_y_max = y_max
        progressStep = ((100. / float(b_scans.shape[2])) / 3.) * 1.
        for i in range(b_scans.shape[2]):
            progressValue += progressStep
            self.controller.set_progress_bar_value(int(progressValue))
            QtGui.QApplication.processEvents()
            b_scan = b_scans[:, :, i]
            b_scan = (b_scan - np.min(b_scan)) / (np.max(b_scan) - np.min(b_scan)) if \
                len(np.unique(b_scan)) > 1 else np.ones(b_scan.shape)
            label = self.layers[:, :, i]
            n_bscan = np.copy(b_scan)
            y, x = xys[i]
            y_n, x_n = xysn[i]
            y_b, x_b = self.get_BM_layer(label)
            y_max = total_y_max
            upper_y = (y_n - y_max)
            c = 0
            upper_y[np.where(upper_y < 0)] = 0
            upper_y[np.where(upper_y >= height)] = height - 1
            for ix in x:
                if b_scan[upper_y[c]:y_n[c], ix].shape[0] > 0:
                    n_bscan[y[c]:y_n[c], ix] = np.max(b_scan[upper_y[c]:y_n[c], ix])
                    projection[i, ix] = np.sum(n_bscan[upper_y[c]:y_n[c] + 1, ix])
                c += 1
        return projection.astype('float'), masks
        # return y, x

    def decompose_into_RPE_BM_images(self, image):
        rpeImg = ((image == 255) + (image == 170)).astype(int) * 255.
        bmImg = ((image == 127) + (image == 170) + (image == 85)).astype(int) * 255.
        return rpeImg, bmImg

    def combine_RPE_BM_images(self, rpeImg, bmImg):
        join = rpeImg * bmImg
        if np.sum(join) > 0:
            rpeImg[bmImg == 255] = 85.
        else:
            rpeImg[bmImg == 255] = 127.
        rpeImg[join > 0] = 170.
        return rpeImg

    def combine_GA_NGA_images(self, ga, nga):
        res = np.zeros(ga.shape)
        join = ga * nga
        res[ga > 0] = 1.
        res[nga > 0] = 2.
        res[join > 0] = 3.
        return res

    def decompose_into_GA_NGA_images(self, img):
        ga = img == 1.
        nga = img == 2.
        join = img == 3.
        ga[np.where(join == True)] = True
        nga[np.where(join == True)] = True
        ga = ga.astype(float) * 255.
        nga = nga.astype(float) * 255.
        return ga, nga

    def normal_RPE_estimation(self, b_scan, degree=3, it=3, s_ratio=1, \
                              farDiff=5, ignoreFarPoints=True, returnImg=False, \
                              useBM=False, useWarping=True, xloc=[], yloc=[]):

        if useWarping:
            y, x = self.get_RPE_location(b_scan)

            yn, xn = self.warp_BM(b_scan)

            return yn, xn
        if useBM:
            y_b, x_b = self.get_BM_location(b_scan)
            y_r, x_r = self.get_RPE_location(b_scan)

            z = np.polyfit(x_b, y_b, deg=degree)
            p = np.poly1d(z)
            y_b = p(x_r).astype('int')

            prev_dist = np.inf
            offset = 0
            for i in range(50):
                newyb = y_b - i
                diff = np.sum(np.abs(newyb - y_r))
                if diff < prev_dist:
                    prev_dist = diff
                    continue
                offset = i
                break
            if returnImg:
                img = np.zeros(b_scan.shape)
                img[y_b - offset, x_r] = 255.0
                return y_b - offset, x_r, img
            return y_b - offset, x_r

        tmp = np.copy(b_scan)
        y = []
        x = []
        if xloc == [] or yloc == []:
            if np.sum(b_scan) == 0.0:
                return y, x
            if len(np.unique(tmp)) == 4:
                tmp2 = np.zeros(tmp.shape)
                tmp2[np.where(tmp == 170)] = 255
                tmp2[np.where(tmp == 255)] = 255
                y, x = np.where(tmp2 == 255)

            else:
                y, x = np.where(tmp == 255)
        else:
            y = yloc
            x = xloc
        tmpx = np.copy(x)
        tmpy = np.copy(y)
        origy = np.copy(y)
        origx = np.copy(x)
        finalx = np.copy(tmpx)
        finaly = tmpy
        for i in range(it):
            if s_ratio > 1:
                s_rate = len(tmpx) / s_ratio
                rand = np.random.rand(s_rate) * len(tmpx)
                rand = rand.astype('int')

                sx = tmpx[rand]
                sy = tmpy[rand]

                z = np.polyfit(sx, sy, deg=degree)
            else:
                z = np.polyfit(tmpx, tmpy, deg=degree)
            p = np.poly1d(z)
            new_y = p(finalx).astype('int')
            if ignoreFarPoints:
                tmpx = []
                tmpy = []
                for i in range(0, len(origx)):
                    diff = new_y[i] - origy[i]
                    if diff < farDiff:
                        tmpx.append(origx[i])
                        tmpy.append(origy[i])
            else:
                tmpy = np.maximum(new_y, tmpy)
            finaly = new_y
        if returnImg:
            return finaly, finalx, tmp
        return finaly, finalx

    def find_area_btw_RPE_normal_RPE(self, mask):
        area_mask = np.zeros(mask.shape)
        for i in range(mask.shape[1]):
            col = mask[:, i]
            v1 = np.where(col == 1.0)
            v2 = np.where(col == 2.0)
            v3 = np.where(col == 3.0)

            v1 = np.min(v1[0]) if len(v1[0]) > 0 else -1
            v2 = np.max(v2[0]) if len(v2[0]) > 0 else -1
            v3 = np.min(v3[0]) if len(v3[0]) > 0 else -1

            if v1 >= 0 and v2 >= 0:
                area_mask[v1:v2, i] = 1
        return area_mask

    def filter_drusen_by_size(self, dmask, slice_num=-1):
        h_threshold = 0
        max_h_t = 0
        w_over_h_ratio_threshold = 0.0

        hv = (np.sum(dmask, axis=0)).astype(int)
        dmask[:, hv < h_threshold] = 0
        return dmask
        drusen_mask = np.copy(dmask)
        if (h_threshold == 0.0 and max_h_t == 0.0 and \
                w_over_h_ratio_threshold == 10000.0):
            return drusen_mask

        cca, num_drusen = sc.ndimage.measurements.label(drusen_mask)
        filtered_mask = np.ones(drusen_mask.shape)
        h = self.compute_heights(cca)
        filtered_mask[np.where(h <= h_threshold)] = 0.0

        h = self.compute_component_max_height(cca)
        filtered_mask[np.where(h <= max_h_t)] = 0.0
        cca, num_drusen = sc.ndimage.measurements.label(filtered_mask)
        w_o_h, height = self.compute_width_height_ratio_height_local_max(cca)
        filtered_mask = np.ones(drusen_mask.shape).astype('float')
        filtered_mask[np.where(w_o_h > w_over_h_ratio_threshold)] = 0.0
        filtered_mask[np.where(w_o_h == 0.0)] = 0.0

        return filtered_mask

    def filter_druse_by_max_height(self, drusenImg, maxHeight):
        if maxHeight == 0:
            return drusenImg
        if len(drusenImg.shape) < 3:
            cca, num_drusen = sc.ndimage.measurements.label(drusenImg)
            h = self.compute_component_max_height(cca)
            drusenImg[np.where(h <= maxHeight)] = 0.0
        else:
            heightProjection = np.sum((drusenImg > 0).astype(int), axis=0)

            cca, num_drusen = sc.ndimage.measurements. \
                label((heightProjection > 0).astype('int'))
            h = self.compute_component_max_height(cca, heightProjection)
            heightProjection[np.where(h <= maxHeight)] = 0.0

            y, s = np.where(heightProjection == 0)
            drusenImg[:, y, s] = 0.
        return drusenImg

    def warp_BM(self, seg_img, returnWarpedImg=False):
        h, w = seg_img.shape
        yr, xr = self.get_RPE_location(seg_img)
        yb, xb = self.get_BM_location(seg_img)
        rmask = np.zeros((h, w), dtype='int')
        bmask = np.zeros((h, w), dtype='int')
        rmask[yr, xr] = 255
        bmask[yb, xb] = 255
        vis_img = np.copy(seg_img)
        shifted = np.zeros(vis_img.shape)
        wvector = np.empty(w, dtype='int')
        wvector.fill(h - (h / 2))
        nrmask = np.zeros((h, w), dtype='int')
        nbmask = np.zeros((h, w), dtype='int')

        zero_x = []
        zero_part = False
        last_nonzero_diff = 0
        for i in range(w):
            bcol = np.where(bmask[:, i] > 0)[0]
            wvector[i] = wvector[i] - np.max(bcol) if len(bcol) > 0 else 0
            if len(bcol) == 0:
                zero_part = True
                zero_x.append(i)
            if len(bcol) > 0 and zero_part:
                diff = wvector[i]
                zero_part = False
                wvector[zero_x] = diff
                zero_x = []
            if len(bcol) > 0:
                last_nonzero_diff = wvector[i]
            if i == w - 1 and zero_part:
                wvector[zero_x] = last_nonzero_diff
        for i in range(w):
            nrmask[:, i] = np.roll(rmask[:, i], wvector[i])
            nbmask[:, i] = np.roll(bmask[:, i], wvector[i])
            shifted[:, i] = np.roll(vis_img[:, i], wvector[i])
        shifted_yr = []
        for i in range(len(xr)):
            shifted_yr.append(yr[i] + wvector[xr[i]])
        yn, xn = self.normal_RPE_estimation(rmask, it=5, useWarping=False, \
                                            xloc=xr, yloc=shifted_yr)
        for i in range(len(xn)):
            yn[i] = yn[i] - wvector[xn[i]]
        if returnWarpedImg:
            return shifted

        return yn, xn

    def compute_heights(self, cca):
        bg_lbl = self.get_label_of_largest_component(cca)
        mask = cca != bg_lbl
        mask = mask.astype('int')
        cvr_h = np.sum(mask, axis=0)
        hghts = np.tile(cvr_h, cca.shape[0]).reshape(cca.shape)
        mask = mask * hghts
        return mask

    def compute_component_max_height(self, cca, heights=[]):
        labels = np.unique(cca)
        max_hs = np.zeros(cca.shape)
        if heights == []:
            heights = self.compute_heights(cca)
        for l in labels:
            region = cca == l
            max_hs[region] = np.max(region * heights)
        return max_hs

    def compute_width_height_ratio_height_local_max(self, cca):
        mx_h = self.compute_component_sum_local_max_height(cca)
        mx_w = self.compute_component_width(cca)
        mx_h[mx_h == 0] = 1
        return mx_w.astype('float') / (mx_h.astype('float')), mx_h

    def compute_component_sum_local_max_height(self, cca):
        labels = np.unique(cca)
        max_hs = np.zeros(cca.shape)
        bg_lbl = self.get_label_of_largest_component(cca)
        heights = self.compute_heights(cca)
        for l in labels:
            if l != bg_lbl:
                region = cca == l
                masked_heights = region * heights
                col_h = np.max(masked_heights, axis=0)
                local_maxima = self.find_rel_maxima(col_h)
                if len(local_maxima) == 0:
                    local_maxima = np.asarray([np.max(masked_heights)])
                max_hs[region] = np.sum(local_maxima)
        return max_hs

    def compute_component_width(self, cca):
        labels = np.unique(cca)
        max_ws = np.zeros(cca.shape)
        bg_lbl = self.get_label_of_largest_component(cca)
        for l in labels:
            if l != bg_lbl:
                y, x = np.where(cca == l)
                w = np.max(x) - np.min(x)
                max_ws[cca == l] = w
        return max_ws

    def find_rel_maxima(self, arr):
        val = []
        pre = -1
        for a in arr:
            if a != pre:
                val.append(a)
            pre = a
        val = np.asarray(val)
        return val[sc.signal.argrelextrema(val, np.greater)]

    def is_rpe(self, value):
        if value == 170 or value == 255:
            return True
        return False

    def is_above_rpe_and_white(self, image, x, y, c):
        if image[x, y] == 0 and c == 255:
            return True
        return False

    def read_pickle_data(self, dataPath):
        with open(dataPath, 'rb') as input:
            return pickle.load(input)

    def write_pickle_data(self, dataPath, d):
        with open(dataPath, 'w') as output:
            pickle.dump(d, output, pickle.HIGHEST_PROTOCOL)

    def find_area_between_seg_lines(self, label):
        h, w = label.shape
        label_area = np.copy(label)
        ls = np.sort(np.unique(label_area))
        if False:
            if len(ls) == 3:

                for j in range(w):
                    col = label[:, j]
                    l_1 = np.where(col == ls[1])
                    l_2 = np.where(col == ls[2])
                    if len(l_1[0]) != 0 and len(l_2[0]) != 0:
                        label_area[l_1[0][0]:l_2[0][0], j] = 1
                        label_area[l_2[0][0]:l_1[0][0], j] = 1

                # Replace all the labels with 1
                label_area[label_area > 0] = 1

                return label_area
            if len(ls) == 4:

                for j in range(w):
                    col = label[:, j]
                    l_1 = np.where(col == ls[1])
                    l_2 = np.where(col == ls[3])
                    if len(l_1[0]) != 0 and len(l_2[0]) != 0:
                        label_area[l_1[0][0]:l_2[0][0], j] = 1
                        label_area[l_2[0][0]:l_1[0][0], j] = 1

                # Replace all the labels with 1
                label_area[label_area > 0] = 1

                return label_area
        else:
            for j in range(w):
                col = label[:, j]
                y = np.where(col > 0)[0]
                if len(y) > 0:
                    minInd = np.min(y)
                    maxInd = np.max(y)
                    label_area[minInd:maxInd, j] = 1
                    label_area[maxInd:minInd, j] = 1
            label_area[label_area > 0] = 1
            return label_area
        return label

    def increase_resolution(self, b_scans, factor, interp='nearest'):

        self.controller.set_progress_bar_value(int(3))
        QtGui.QApplication.processEvents()

        new_size = (b_scans.shape[0], b_scans.shape[1], b_scans.shape[2] * factor)
        res = np.zeros(new_size)
        progressValue = 5.0

        self.controller.set_progress_bar_value(int(progressValue))
        QtGui.QApplication.processEvents()

        progressStep = ((100. / new_size[1]) / 3.) * 2.

        for i in range(new_size[1]):
            progressValue += progressStep
            self.controller.set_progress_bar_value(int(progressValue))
            QtGui.QApplication.processEvents()
            slice_i = b_scans[:, i, :]

            res[:, i, :] = (misc.imresize(slice_i, (new_size[0], new_size[2]), \
                                          interp=interp).astype('float')) / 255.0
            mask = np.copy(np.cumsum(res[:, i, :], axis=0) * ((res[:, i, :] > 0).astype('int')))
            res[:, i, :] = (mask >= 1.0).astype('float')
        return res, progressValue

    def convert_from_pixel_size_to_meter(self):
        voxelSize = self.hx * self.hy * self.hz
        volumeM = self.volume * voxelSize
        areaM = self.area * self.hx * self.hz
        heightM = self.height * self.hy
        xM = self.largeR * np.cos(self.theta) * self.hx
        zM = self.largeR * np.sin(self.theta) * self.hz
        largeM = np.sqrt(xM ** 2 + zM ** 2)
        thetaVer = self.theta + (np.pi / 2.0)
        xM = self.smallR * np.cos(thetaVer) * self.hx
        zM = self.smallR * np.sin(thetaVer) * self.hz
        smallM = np.sqrt(xM ** 2 + zM ** 2)
        return self.cx * self.hx, self.cy * self.hz, areaM, heightM, volumeM, largeM, smallM, self.theta

    def curve_to_spline(self, layerName, sliceNum, method="estimate", sampling_rate=10):
        """ Compute spline knots for layer curve

        :param layerName:
        :param sliceNum:
        :param method:
        :param sampling_rate:
        :return:
        """
        sliceZ = sliceNum - 1

        # Return knots if already computed
        if sliceZ in self.splineKnots[layerName].keys():
            return self.layers[:, :, sliceZ], self.splineKnots[layerName][sliceZ]

        layer = self.layers[:, :, sliceZ]

        # Get layer indices
        if layerName == 'RPE':
            y, x = self.get_RPE_layer(layer)
        elif layerName == 'BM':
            y, x = self.get_BM_layer(layer)


        if method == "const":
            # Choose every nth point of the layer, depends on sampling rate
            xs = x[::sampling_rate].copy()
            ys = y[::sampling_rate].copy()

        elif method == "estimate":
            # Sort the indices
            argx = np.argsort(x)
            x = x[argx]
            y = y[argx]

            # Remove duplicates
            x, unique_indices = np.unique(x, return_index=True)
            y = y[unique_indices]

            # Fit spline
            spl = UnivariateSpline(x, y)
            wsize = len(x)

            # Smooth the spline
            for s in np.arange(0, wsize)[::-1]:
                if spl.get_residual() < 50:
                    break
                spl.set_smoothing_factor(s)
            xs = np.asarray(spl.get_knots()).astype(int)
            ys = y[xs]

            if len(xs) > 100:
                return self.curve_to_spline(layerName, sliceNum, method="const")
        else:
            msg = 'Method "{}" not supported'.format(method)
            logger.debug(msg)
            raise ValueError(msg)

        self.splineKnots[layerName][sliceZ] = [xs, ys]

        return self.layers[:, :, sliceZ], self.splineKnots[layerName][sliceZ]

    def spline_to_curve(self, layerName, sliceNum):
        sliceZ = sliceNum - 1

        logger.debug('{}, {}: {}'.format(layerName, sliceNum, self.splineKnots))
        if sliceZ not in self.splineKnots[layerName].keys():
            logger.debug("Warning: No spline info for layer " + layerName + " at slice " + str(sliceZ + 1) + " given!")
            logger.debug('After Warning: {}, {}: {}'.format(layerName, sliceNum, self.splineKnots))
            return None, None
        xs, ys = self.splineKnots[layerName][sliceZ]
        for deg in range(1, 4)[::-1]:
            try:
                tck = sc.interpolate.splrep(xs, ys, k=deg)
                break
            except:
                logger.debug("No spline of degree {} can be produced!".format(deg))

        ys = sc.interpolate.splev(np.arange(0, self.layers.shape[1]), tck)
        rpeImg, bmImg = self.decompose_into_RPE_BM_images(self.layers[:, :, sliceZ])
        newCurve = np.zeros(rpeImg.shape)
        h, w = self.layers[:, :, sliceZ].shape
        newCurve[np.rint(ys).astype(int), np.arange(0, w)] = 255.
        prevLoc = -1
        # Fill Gaps
        for i in range(newCurve.shape[1]):
            yloc = np.where(newCurve[:, i])

            if i == 0:
                if len(yloc) > 0:
                    prevLoc = yloc[0][0]
                continue
            if prevLoc != -1:
                if prevLoc < yloc[0][0]:
                    newCurve[prevLoc + 1:yloc[0][0], i] = 255.
                else:
                    newCurve[yloc[0][0]:prevLoc, i] = 255.
            if len(yloc) > 0:
                prevLoc = yloc[0][0]
            elif len(yloc) == 0:
                prevLoc = -1
        if layerName == 'RPE':
            layer = self.combine_RPE_BM_images(newCurve, bmImg)
        elif layerName == 'BM':
            layer = self.combine_RPE_BM_images(rpeImg, newCurve)
        self.layers[:, :, sliceZ] = np.copy(layer)
        return layer, self.splineKnots[layerName][sliceZ]

    def spline_update_using_info(self, info):
        sliceNumZ = info['sliceNum'] - 1
        layerName = info['layerName']
        if info['layer'] is not None:
            self.layers[:, :, sliceNumZ] = info['layer']
            self.splineKnots[layerName][sliceNumZ] = info['knots']
            self.layerSegmenter.set_uncertainties(info['uncertainties'], sliceNumZ)
            self.controller.set_uncertainties(info['uncertainties'], sliceNumZ)

    def spline_update_using_data(self, layer, knots, info):
        sliceNumZ = info['sliceNum'] - 1
        layerName = info['layerName']
        if not layer is None:
            self.layers[:, :, sliceNumZ] = layer
            self.splineKnots[layerName][sliceNumZ] = knots
            self.editedLayers[sliceNumZ][layerName] = info['prevStatus']
            self.layerSegmenter.set_uncertainties(info['uncertainties'], sliceNumZ)
            self.controller.set_uncertainties(info['uncertainties'], sliceNumZ)

    def get_spline_knots(self, layerName, sliceZ):
        if sliceZ not in self.splineKnots[layerName].keys():
            self.curve_to_spline(layerName, sliceZ+1)
        return self.splineKnots[layerName][sliceZ]

    def add_spline_knot(self, y, x, layerName, sliceZ):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return
        knots = self.get_spline_knots(layerName, sliceZ)
        if knots is None:
            return
        knotsx, knotsy = knots
        # Check if point is already exists
        if x in knotsx:
            knotsy[np.where(knotsx == x)] = y  # Update location
        # Otherwise add as new knot
        else:
            knotsx = np.append(knotsx, x)
            knotsy = np.append(knotsy, y)
            sortInd = np.argsort(knotsx)
            sortedx = knotsx[sortInd]
            sortedy = knotsy[sortInd]
            knotsx = sortedx
            knotsy = sortedy
        self.splineKnots[layerName][sliceZ] = [knotsx, knotsy]

    def delete_spline_knot(self, y, x, layerName, sliceZ):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return
        knots = self.get_spline_knots(layerName, sliceZ)
        if knots is None:
            return
        knotsx, knotsy = knots

        mindist = 1000
        minx = x
        for i in [-1, 0, 1]:
            if x + i in knotsx:
                ind = np.where(knotsx == (x + i))
                for j in [-1, 0, 1]:
                    if (y + j) == knotsy[ind]:
                        dist = np.sqrt(i ** 2 + j ** 2)
                        if dist < mindist:
                            minx = x + i
                            mindist = dist

        # Check if point exists
        ind = np.where(knotsx == minx)
        knotsx = np.delete(knotsx, ind)
        knotsy = np.delete(knotsy, ind)
        self.splineKnots[layerName][sliceZ] = [knotsx, knotsy]

    def is_knot(self, y, x, layerName, sliceZ):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        knots = self.get_spline_knots(layerName, sliceZ)
        if knots is None:
            return False
        knotsx, knotsy = knots
        # Check 1 pixel neighborhood
        for i in [-1, 0, 1]:
            if x + i in knotsx:
                ind = np.where(knotsx == (x + i))
                for j in [-1, 0, 1]:
                    if (y + j) == knotsy[ind]:
                        return True
        return False

    def get_closest_knot(self, y, x, layerName, sliceZ):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        knots = self.get_spline_knots(layerName, sliceZ)
        if knots is None:
            return False
        knotsx, knotsy = knots
        # Check neighborhood for candidates
        mindist = 1000
        minx = x
        miny = y
        for i in [-1, 0, 1]:
            if x + i in knotsx:
                ind = np.where(knotsx == (x + i))
                for j in [-1, 0, 1]:
                    if (y + j) == knotsy[ind]:
                        dist = np.sqrt(i ** 2 + j ** 2)
                        if dist < mindist:
                            minx = x + i
                            miny = y + j
                            mindist = dist
        return minx, miny

    """ Not used """
    def show_images(self, images, r, c, titles=[], d=0, save_path="", block=True):
        i = 1
        for img in images:
            ax = plt.subplot(r, c, i)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            if len(titles) != 0:
                ax.set_title(titles[i - 1])
            if len(img.shape) > 2:
                plt.imshow(img)
            else:
                plt.imshow(img, cmap=plt.get_cmap('gray'))
            i += 1
        if save_path != "":
            plt.savefig(save_path + ".png")
            plt.close()
        else:
            plt.show(block)

    def extract_drusen_using_normal_thickness(self, thickness, sliceZ):
        layer = self.layers[:, :, sliceZ]
        allArea = self.find_area_between_seg_lines(layer) * 255.0
        # Find BM layer, shift it up by thickness, set everything as background under
        bmy, bmx = self.get_BM_location(layer)
        filterMask = np.zeros(allArea.shape)
        bmy = bmy - thickness
        filterMask[bmy, bmx] = 1.0
        filterMask = np.cumsum(filterMask, axis=0)
        filterMask[filterMask > 0] = 1.0
        allArea[filterMask > 0] = 0
        self.drusen[:, :, sliceZ] = allArea

    def extract_drusen_using_normal_thickness_in_volume(self, thickness):
        for s in range(self.drusen.shape[2]):
            self.extract_drusen_using_normal_thickness(thickness, s)

    def update_knot_position(self, y, x, oldy, oldx, layerName, sliceZ):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return
        knots = self.get_spline_knots(layerName, sliceZ)
        if knots is None:
            return
        knotsx, knotsy = knots
        if x in knotsx:
            # If new pos concides with another knot, keep old xpos, only update
            # ypos
            knotsy[np.where(knotsx == oldx)] = y  # Update location

        else:
            # Delete the old knot
            self.delete_spline_knot(oldy, oldx, layerName, sliceZ)
            # Add new knot
            self.add_spline_knot(y, x, layerName, sliceZ)

    def quantify_drusen(self):
        self.controller.show_progress_bar()

        projected_labels = self.enfaceDrusen
        labels = self.drusen

        progressValue = 0.

        self.controller.set_progress_bar_value(int(1))
        QtGui.QApplication.processEvents()
        realResProjLbl = sc.misc.imresize(projected_labels, size= \
            (projected_labels.shape[0] * self.zRate, \
             projected_labels.shape[1]), interp='nearest')
        self.controller.set_progress_bar_value(int(2))
        QtGui.QApplication.processEvents()
        realResLbl, progressValue = self.increase_resolution(((labels > 0). \
                                                              astype('float')), factor=self.zRate, interp='bilinear')

        heightImg = np.sum(realResLbl, axis=0).T
        realResProjLbl = ((realResProjLbl * heightImg) > 0.0).astype('float')
        self.controller.set_progress_bar_value(int(4))
        QtGui.QApplication.processEvents()
        cca, numL = sc.ndimage.measurements.label(realResProjLbl)
        self.controller.set_progress_bar_value(int(5))
        QtGui.QApplication.processEvents()

        area = list()
        volume = list()
        largeR = list()
        smallR = list()
        theta = list()
        height = list()
        cx = list()
        cy = list()
        bgL = self.get_label_of_largest_component(cca)
        labels = np.unique(cca)

        self.controller.set_progress_bar_value(int(progressValue))
        QtGui.QApplication.processEvents()

        progressStep = ((100. / len(labels)) / 3.) * 1.

        for l in labels:

            if l != bgL:
                progressValue += progressStep
                self.controller.set_progress_bar_value(int(progressValue))
                QtGui.QApplication.processEvents()

                componentL = (cca == l).astype('float')
                componentL = ((heightImg * componentL) > 0.0).astype('float')
                cyL, cxL = sc.ndimage.measurements.center_of_mass(componentL)
                areaL = np.sum(((heightImg * componentL) > 0.0).astype('float'))
                volumeL = np.sum(heightImg * componentL)
                heightL = np.max(heightImg * componentL)
                largeL = 0.0
                smallL = 0.0
                thetaL = 0.0
                props = skm.regionprops(componentL.astype('int'))
                for p in props:
                    if p.label == 1:
                        areaL = p.area
                        largeL = max(1., p.major_axis_length)
                        smallL = max(1., p.minor_axis_length)
                        thetaL = p.orientation

                area.append(areaL)
                volume.append(volumeL)
                theta.append(thetaL)
                smallR.append(smallL)
                largeR.append(largeL)
                height.append(heightL)
                cx.append(cxL)
                cy.append(cyL)

        area = np.asarray(area)
        volume = np.asarray(volume)
        largeR = np.asarray(largeR)
        smallR = np.asarray(smallR)
        theta = np.asarray(theta)
        height = np.asarray(height)
        cy = np.asarray(cy)
        cx = np.asarray(cx)
        total = (area + height + volume + largeR + smallR) / 5.0
        indx = np.argsort(total)
        indx = indx[::-1]
        area = area[indx]
        volume = volume[indx]
        largeR = largeR[indx]
        smallR = smallR[indx]
        theta = theta[indx]
        height = height[indx]
        cy = cy[indx]
        cx = cx[indx]

        self.controller.set_progress_bar_value(100)
        self.controller.hide_progress_bar()
        self.cx = cx
        self.cy = cy
        self.area = area
        self.height = height
        self.volume = volume
        self.largeR = largeR
        self.smallR = smallR
        self.theta = theta
        return cx, cy, area, height, volume, largeR, smallR, theta

    def pen_or_line_undo(self, sliceNum, info, layerName):

        sliceNumZ = sliceNum - 1

        if sliceNumZ in self.certainSlices:
            self.certainSlices.remove(sliceNumZ)
        self.editedLayers[sliceNumZ][layerName] = info['prevStatus']

        self.layerSegmenter.set_uncertainties(info['uncertainties'], sliceNumZ)
        self.controller.set_uncertainties(info['uncertainties'], sliceNumZ)

    def pen_or_line_redo(self, sliceNum, layerName):

        info = dict()
        sliceNumZ = sliceNum - 1
        info['prevStatus'] = None
        info['prevStatus'] = self.editedLayers[sliceNumZ][layerName]
        self.certainSlices.append(sliceNumZ)
        self.editedLayers[sliceNumZ][layerName] = True

        info['uncertainties'] = self.layerSegmenter.get_uncertainties(sliceNumZ)
        return info

    def create_directory(self, path):
        """
        Check if the directory exists. If not, create it.
        Input:
            path: the directory to create
        Output:
            None.
        """
        if not os.path.exists(path):
            os.makedirs(path)

    def compute_drusen_load_in_px_and_um(self):
        drusenLoad = np.sum((self.drusen > 0).astype(int))
        umCoeff = self.hx * self.hy * self.hz
        drusenLoadInUM = drusenLoad * umCoeff
        return drusenLoad, drusenLoadInUM
