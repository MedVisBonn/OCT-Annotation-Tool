# -*- coding: utf-8 -*-

"""
Created in 2018

@author: Shekoufeh Gorgi Zadeh
"""

import os, sys, inspect
from PyQt4 import QtCore, QtGui

global sfwPath
sfwPath=os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))[:-4]
controllerPath=sfwPath+"controller"
if controllerPath not in sys.path:
     sys.path.insert(0, controllerPath)

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

#==============================================================================
# Toolbox viewer class
#==============================================================================
class Ui_toolBox(object):
    
    def __init__(self,controller=None):
        self.controller=controller
        
    def setupUi(self, toolBox,undoAction,redoAction):
        toolBox.setObjectName(_fromUtf8("toolBox"))
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding,\
            QtGui.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(toolBox.sizePolicy().hasHeightForWidth())
        toolBox.setMinimumSize(QtCore.QSize(249, 500))
        toolBox.setSizePolicy(sizePolicy)
        self.toolBox=toolBox
        self.verticalLayout_7 = QtGui.QVBoxLayout(toolBox)
        self.verticalLayout_7.setObjectName(_fromUtf8("verticalLayout_7"))
        self.groupBox_edit = QtGui.QGroupBox(toolBox)
        self.groupBox_edit.setObjectName(_fromUtf8("groupBox_edit"))
        self.gridLayout = QtGui.QGridLayout(self.groupBox_edit)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.radioButtonRPE = QtGui.QRadioButton(self.groupBox_edit)
        self.radioButtonRPE.setObjectName(_fromUtf8("radioButtonRPE"))
        self.gridLayout.addWidget(self.radioButtonRPE, 0, 0, 1, 1)
        self.radioButtonEnface = QtGui.QRadioButton(self.groupBox_edit)
        self.radioButtonEnface.setObjectName(_fromUtf8("radioButtonEnface"))
        self.gridLayout.addWidget(self.radioButtonEnface, 0, 1, 1, 1)
        self.radioButtonBM = QtGui.QRadioButton(self.groupBox_edit)
        self.radioButtonBM.setObjectName(_fromUtf8("radioButtonBM"))
        self.gridLayout.addWidget(self.radioButtonBM, 1, 0, 1, 1)
        self.radioButtonHRF = QtGui.QRadioButton(self.groupBox_edit)
        self.radioButtonHRF.setObjectName(_fromUtf8("radioButtonHRF"))
        self.gridLayout.addWidget(self.radioButtonHRF, 1, 1, 1, 1)
        self.radioButtonDrusen = QtGui.QRadioButton(self.groupBox_edit)
        self.radioButtonDrusen.setObjectName(_fromUtf8("radioButtonDrusen"))
        self.gridLayout.addWidget(self.radioButtonDrusen, 2, 0, 1, 1)
        self.radioButtonGA = QtGui.QRadioButton(self.groupBox_edit)
        self.radioButtonGA.setObjectName(_fromUtf8("radioButtonGA"))
        self.gridLayout.addWidget(self.radioButtonGA, 2, 1, 1, 1)
        self.verticalLayout_7.addWidget(self.groupBox_edit)
        spacerItem = QtGui.QSpacerItem(20, 4, QtGui.QSizePolicy.Minimum,\
            QtGui.QSizePolicy.Fixed)
        self.verticalLayout_7.addItem(spacerItem)
        self.groupBox_opacity = QtGui.QGroupBox(toolBox)
        self.groupBox_opacity.setObjectName(_fromUtf8("groupBox_opacity"))
        self.verticalLayout_5 = QtGui.QVBoxLayout(self.groupBox_opacity)
        self.verticalLayout_5.setObjectName(_fromUtf8("verticalLayout_5"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.verticalLayout_3 = QtGui.QVBoxLayout()
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.label = QtGui.QLabel(self.groupBox_opacity)
        self.label.setObjectName(_fromUtf8("label"))
        self.verticalLayout_3.addWidget(self.label)
        self.label_2 = QtGui.QLabel(self.groupBox_opacity)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.verticalLayout_3.addWidget(self.label_2)
        self.label_3 = QtGui.QLabel(self.groupBox_opacity)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.verticalLayout_3.addWidget(self.label_3)
        self.horizontalLayout_2.addLayout(self.verticalLayout_3)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalSliderBscan = QtGui.QSlider(self.groupBox_opacity)
        self.horizontalSliderBscan.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSliderBscan.setObjectName(_fromUtf8("horizontalSliderBscan"))
        self.horizontalSliderBscan.setProperty("value", 60)
        self.verticalLayout.addWidget(self.horizontalSliderBscan)
        self.horizontalSliderLayerMap = QtGui.QSlider(self.groupBox_opacity)
        self.horizontalSliderLayerMap.setProperty("value", 0)
        self.horizontalSliderLayerMap.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSliderLayerMap.setObjectName(\
            _fromUtf8("horizontalSliderLayerMap"))
        self.verticalLayout.addWidget(self.horizontalSliderLayerMap)
        self.horizontalSliderDrusenMap = QtGui.QSlider(self.groupBox_opacity)
        self.horizontalSliderDrusenMap.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSliderDrusenMap.setObjectName(\
            _fromUtf8("horizontalSliderDrusenMap"))
        self.verticalLayout.addWidget(self.horizontalSliderDrusenMap)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.verticalLayout_5.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.verticalLayout_4 = QtGui.QVBoxLayout()
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.label_4 = QtGui.QLabel(self.groupBox_opacity)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.verticalLayout_4.addWidget(self.label_4)
        self.label_5 = QtGui.QLabel(self.groupBox_opacity)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.verticalLayout_4.addWidget(self.label_5)
        self.horizontalLayout.addLayout(self.verticalLayout_4)
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.horizontalSliderEnface = QtGui.QSlider(self.groupBox_opacity)
        self.horizontalSliderEnface.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSliderEnface.setProperty("value", 50)
        self.horizontalSliderEnface.setObjectName(\
            _fromUtf8("horizontalSliderEnface"))
        self.verticalLayout_2.addWidget(self.horizontalSliderEnface)
        self.horizontalSliderEnfaceDrusen = QtGui.QSlider(self.groupBox_opacity)
        self.horizontalSliderEnfaceDrusen.setProperty("value", 99)
        self.horizontalSliderEnfaceDrusen.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSliderEnfaceDrusen.setObjectName(\
            _fromUtf8("horizontalSliderEnfaceDrusen"))
        self.verticalLayout_2.addWidget(self.horizontalSliderEnfaceDrusen)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout_5.addLayout(self.horizontalLayout)
        self.verticalLayout_7.addWidget(self.groupBox_opacity)
        spacerItem1 = QtGui.QSpacerItem(20, 4, QtGui.QSizePolicy.Minimum,\
            QtGui.QSizePolicy.Fixed)
        self.verticalLayout_7.addItem(spacerItem1)
        self.groupBox_toolBox = QtGui.QGroupBox(toolBox)
        self.groupBox_toolBox.setObjectName(_fromUtf8("groupBox_toolBox"))
        self.gridLayout_2 = QtGui.QGridLayout(self.groupBox_toolBox)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.toolButtonPen = QtGui.QToolButton(self.groupBox_toolBox)
        self.toolButtonPen.setMinimumSize(QtCore.QSize(34, 34))
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","pen.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButtonPen.setIcon(icon5)
        self.toolButtonPen.setObjectName(_fromUtf8("toolButtonPen"))
        self.gridLayout_2.addWidget(self.toolButtonPen, 0, 0, 1, 1)
        self.toolButtonLine = QtGui.QToolButton(self.groupBox_toolBox)
        self.toolButtonLine.setMinimumSize(QtCore.QSize(34, 34))
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","line.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButtonLine.setIcon(icon6)
        self.toolButtonLine.setObjectName(_fromUtf8("toolButtonLine"))
        self.gridLayout_2.addWidget(self.toolButtonLine, 0, 1, 1, 1)
        self.toolButtonFill = QtGui.QToolButton(self.groupBox_toolBox)
        self.toolButtonFill.setMinimumSize(QtCore.QSize(34, 34))
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","paint.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButtonFill.setIcon(icon7)
        self.toolButtonFill.setObjectName(_fromUtf8("toolButtonFill"))
        self.gridLayout_2.addWidget(self.toolButtonFill, 0, 2, 1, 1)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","chain.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButtonMorphology = QtGui.QToolButton(self.groupBox_toolBox)
        self.toolButtonMorphology.setMinimumSize(QtCore.QSize(34, 34))
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","scale.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButtonMorphology.setIcon(icon10)
        self.toolButtonMorphology.setObjectName(_fromUtf8("toolButtonMorphology"))
        self.gridLayout_2.addWidget(self.toolButtonMorphology, 1, 0, 1, 1)
        self.toolButtonDrawDru = QtGui.QToolButton(self.groupBox_toolBox)
        self.toolButtonDrawDru.setMinimumSize(QtCore.QSize(34, 34))
        icon11 = QtGui.QIcon()
        icon11.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","drawDru.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButtonDrawDru.setIcon(icon11)
        self.toolButtonDrawDru.setObjectName(_fromUtf8("toolButtonDrawDru"))
        self.gridLayout_2.addWidget(self.toolButtonDrawDru, 1, 1, 1, 1)
        self.toolButtonFilterDru = QtGui.QToolButton(self.groupBox_toolBox)
        self.toolButtonFilterDru.setMinimumSize(QtCore.QSize(34, 34))
        icon12 = QtGui.QIcon()
        icon12.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","heightThreshold.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButtonFilterDru.setIcon(icon12)
        self.toolButtonFilterDru.setObjectName(_fromUtf8("toolButtonFilterDru"))
        self.gridLayout_2.addWidget(self.toolButtonFilterDru, 1, 2, 1, 1)
        self.toolButtonGrab = QtGui.QToolButton(self.groupBox_toolBox)
        self.toolButtonGrab.setMinimumSize(QtCore.QSize(34, 34))
        self.toolButtonGrab.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        icon13 = QtGui.QIcon()
        icon13.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","grab.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButtonGrab.setIcon(icon13)
        self.toolButtonGrab.setCheckable(False)
        self.toolButtonGrab.setChecked(False)
        self.toolButtonGrab.setObjectName(_fromUtf8("toolButtonGrab"))
        self.gridLayout_2.addWidget(self.toolButtonGrab, 0, 3, 1, 1)
        
        self.toolButtonCCA = QtGui.QToolButton(self.groupBox_toolBox)
        self.toolButtonCCA.setMinimumSize(QtCore.QSize(34, 34))
        self.toolButtonCCA.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        icon14 = QtGui.QIcon()
        icon14.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","CCA.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButtonCCA.setIcon(icon14)
        self.toolButtonCCA.setObjectName(_fromUtf8("toolButtonCCA"))
        self.gridLayout_2.addWidget(self.toolButtonCCA, 2, 4, 1, 1)
        
        self.toolButtonNextComp = QtGui.QToolButton(self.groupBox_toolBox)
        self.toolButtonNextComp.setMinimumSize(QtCore.QSize(34, 34))
        self.toolButtonNextComp.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        icon15 = QtGui.QIcon()
        icon15.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","next.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButtonNextComp.setIcon(icon15)
        self.toolButtonNextComp.setObjectName(_fromUtf8("toolButtonNextComp"))
        self.gridLayout_2.addWidget(self.toolButtonNextComp, 2, 1, 1, 1)
        
        self.toolButtonPrevComp = QtGui.QToolButton(self.groupBox_toolBox)
        self.toolButtonPrevComp.setMinimumSize(QtCore.QSize(34, 34))
        self.toolButtonPrevComp.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        icon16 = QtGui.QIcon()
        icon16.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","prev.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButtonPrevComp.setIcon(icon16)
        self.toolButtonPrevComp.setObjectName(_fromUtf8("toolButtonPrevComp"))
        self.gridLayout_2.addWidget(self.toolButtonPrevComp, 2, 0, 1, 1)        
        
        
        self.toolButtonCheckComp = QtGui.QToolButton(self.groupBox_toolBox)
        self.toolButtonCheckComp.setMinimumSize(QtCore.QSize(34, 34))
        self.toolButtonCheckComp.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        icon17 = QtGui.QIcon()
        icon17.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","check.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButtonCheckComp.setIcon(icon17)
        self.toolButtonCheckComp.setObjectName(_fromUtf8("toolButtonCheckComp"))
        self.gridLayout_2.addWidget(self.toolButtonCheckComp, 2, 3, 1, 1)   
        
        self.toolButtonUnCheckComp = QtGui.QToolButton(self.groupBox_toolBox)
        self.toolButtonUnCheckComp.setMinimumSize(QtCore.QSize(34, 34))
        self.toolButtonUnCheckComp.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        icon18 = QtGui.QIcon()
        icon18.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","cancel.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButtonUnCheckComp.setIcon(icon18)
        self.toolButtonUnCheckComp.setObjectName(_fromUtf8("toolButtonUnCheckComp"))
        self.gridLayout_2.addWidget(self.toolButtonUnCheckComp, 2, 2, 1, 1)
        
        
        self.toolButtonBBox = QtGui.QToolButton(self.groupBox_toolBox)
        self.toolButtonBBox.setMinimumSize(QtCore.QSize(34, 34))
        self.toolButtonBBox.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        icon27 = QtGui.QIcon()
        icon27.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","bbox.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButtonBBox.setIcon(icon27)
        self.toolButtonBBox.setObjectName(_fromUtf8("toolButtonBBox"))
        self.gridLayout_2.addWidget(self.toolButtonBBox, 1, 3, 1, 1)
        
        self.toolButtonCostPnt = QtGui.QToolButton(self.groupBox_toolBox)
        self.toolButtonCostPnt.setMinimumSize(QtCore.QSize(34, 34))
        self.toolButtonCostPnt.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        icon28 = QtGui.QIcon()
        icon28.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","CSP.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButtonCostPnt.setIcon(icon28)
        self.toolButtonCostPnt.setObjectName(_fromUtf8("toolButtonCostPnt"))
        self.gridLayout_2.addWidget(self.toolButtonCostPnt, 0, 4, 1, 1)
        
        self.toolButtonPolyFit = QtGui.QToolButton(self.groupBox_toolBox)
        self.toolButtonPolyFit.setMinimumSize(QtCore.QSize(34, 34))
        self.toolButtonPolyFit.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        icon29 = QtGui.QIcon()
        icon29.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","polyfit.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButtonPolyFit.setIcon(icon29)
        self.toolButtonPolyFit.setObjectName(_fromUtf8("toolButtonPolyFit"))
        self.gridLayout_2.addWidget(self.toolButtonPolyFit, 1, 4, 1, 1)
                
        self.toolButtonSplitDrusen = QtGui.QToolButton(self.groupBox_toolBox)
        self.toolButtonSplitDrusen.setMinimumSize(QtCore.QSize(34, 34))
        self.toolButtonSplitDrusen.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        icon30 = QtGui.QIcon()
        icon30.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","split.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButtonSplitDrusen.setIcon(icon30)
        self.toolButtonSplitDrusen.setObjectName(_fromUtf8("toolButtonSplitDrusen"))
        self.gridLayout_2.addWidget(self.toolButtonSplitDrusen, 3, 0, 1, 1)
        
        self.verticalLayout_6 = QtGui.QVBoxLayout(self.toolBox)
        self.verticalLayout_6.setObjectName(_fromUtf8("verticalLayout_6"))
        spacerItem2 = QtGui.QSpacerItem(20, 10, QtGui.QSizePolicy.Minimum,\
            QtGui.QSizePolicy.Fixed)
        self.verticalLayout_6.addItem(spacerItem2)
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.label_size1 = QtGui.QLabel(self.groupBox_toolBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed,\
            QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_size1.sizePolicy().\
            hasHeightForWidth())
        self.label_size1.setSizePolicy(sizePolicy)
        self.label_size1.setMinimumSize(QtCore.QSize(35, 10))
        self.label_size1.setObjectName(_fromUtf8("label_size1"))
        self.horizontalLayout_4.addWidget(self.label_size1)
        self.horizontalSlider_size1 = QtGui.QSlider(self.groupBox_toolBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding,\
            QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.horizontalSlider_size1.sizePolicy().\
            hasHeightForWidth())
        self.horizontalSlider_size1.setSizePolicy(sizePolicy)
        self.horizontalSlider_size1.setMinimumSize(QtCore.QSize(60, 5))
        self.horizontalSlider_size1.setMaximum(19)
        self.horizontalSlider_size1.setProperty("value", 1)
        self.horizontalSlider_size1.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_size1.setObjectName(_fromUtf8("horizontalSlider_size1"))
        self.horizontalLayout_4.addWidget(self.horizontalSlider_size1)
        self.spinBox_size1 = QtGui.QSpinBox(self.groupBox_toolBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed,\
            QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.spinBox_size1.sizePolicy().\
            hasHeightForWidth())
        self.spinBox_size1.setSizePolicy(sizePolicy)
        self.spinBox_size1.setMinimumSize(QtCore.QSize(20, 27))
        self.spinBox_size1.setMaximum(19)
        self.spinBox_size1.setProperty("value", 1)
        self.spinBox_size1.setObjectName(_fromUtf8("spinBox_size1"))
        self.horizontalLayout_4.addWidget(self.spinBox_size1)
        self.label_7 = QtGui.QLabel(self.groupBox_toolBox)
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.horizontalLayout_4.addWidget(self.label_7)
        self.verticalLayout_6.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.label_size2 = QtGui.QLabel(self.groupBox_toolBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed,\
            QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_size2.sizePolicy().\
            hasHeightForWidth())
        self.label_size2.setSizePolicy(sizePolicy)
        self.label_size2.setMinimumSize(QtCore.QSize(35, 10))
        self.label_size2.setObjectName(_fromUtf8("label_size2"))
        self.horizontalLayout_3.addWidget(self.label_size2)
        self.horizontalSlider_size2 = QtGui.QSlider(self.groupBox_toolBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding,\
            QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.horizontalSlider_size2.sizePolicy().\
            hasHeightForWidth())
        self.horizontalSlider_size2.setSizePolicy(sizePolicy)
        self.horizontalSlider_size2.setMinimumSize(QtCore.QSize(60, 5))
        self.horizontalSlider_size2.setMaximum(19)
        self.horizontalSlider_size2.setProperty("value", 2)
        self.horizontalSlider_size2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_size2.setObjectName(_fromUtf8("horizontalSlider_size2"))
        self.horizontalLayout_3.addWidget(self.horizontalSlider_size2)
        self.spinBox_size2 = QtGui.QSpinBox(self.groupBox_toolBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.\
            QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.spinBox_size2.sizePolicy().\
            hasHeightForWidth())
        self.spinBox_size2.setSizePolicy(sizePolicy)
        self.spinBox_size2.setMinimumSize(QtCore.QSize(20, 27))
        self.spinBox_size2.setMaximum(19)
        self.spinBox_size2.setProperty("value", 2)
        self.spinBox_size2.setObjectName(_fromUtf8("spinBox_size2"))
        self.horizontalLayout_3.addWidget(self.spinBox_size2)
        self.label_9 = QtGui.QLabel(self.groupBox_toolBox)
        self.label_9.setObjectName(_fromUtf8("label_9"))
        self.horizontalLayout_3.addWidget(self.label_9)
        self.verticalLayout_6.addLayout(self.horizontalLayout_3)
        
        self.horizontalLayout_13 = QtGui.QHBoxLayout()
        self.horizontalLayout_13.setObjectName(_fromUtf8("horizontalLayout_13"))
        
        self.toolButtonFilterDruImm = QtGui.QToolButton(self.groupBox_toolBox)
        sizePolicy2 = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding,\
            QtGui.QSizePolicy.Fixed)
        self.toolButtonFilterDruImm.setSizePolicy(sizePolicy2)
        icon18 = QtGui.QIcon()
        icon18.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","heightThreshold.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButtonFilterDruImm.setText('Apply')
        self.toolButtonFilterDruImm.setObjectName(\
            _fromUtf8("toolButtonFilterDruImm"))
        self.horizontalLayout_13.addWidget(self.toolButtonFilterDruImm)
        
        self.verticalLayout_6.addLayout(self.horizontalLayout_13)
        
        self.verticalLayoutDruseSplitTools = QtGui.QVBoxLayout()
        self.verticalLayoutDruseSplitTools.setObjectName(\
            _fromUtf8("verticalLayoutDruseSplitTools"))
        self.radioButtonUseMaximaAsMarker = QtGui.QRadioButton()
        self.radioButtonUseMaximaAsMarker.setObjectName(\
            _fromUtf8("radioButtonUseMaximaAsMarker"))
        self.verticalLayoutDruseSplitTools.addWidget(self.radioButtonUseMaximaAsMarker)
        self.horizontalLayoutSplitNeighborhood = QtGui.QHBoxLayout()
        self.horizontalLayoutSplitNeighborhood.setObjectName(\
            _fromUtf8("horizontalLayoutSplitNeighborhood"))
        spacerItem = QtGui.QSpacerItem(13, 20, QtGui.QSizePolicy.Fixed,\
            QtGui.QSizePolicy.Minimum)
        self.horizontalLayoutSplitNeighborhood.addItem(spacerItem)
        self.labelNeighborhoodSize = QtGui.QLabel()
        self.labelNeighborhoodSize.setObjectName(\
            _fromUtf8("labelNeighborhoodSize"))
        self.horizontalLayoutSplitNeighborhood.addWidget(\
            self.labelNeighborhoodSize)
        self.spinBoxNeighborhoodSize = QtGui.QSpinBox()
        self.spinBoxNeighborhoodSize.setMinimum(2)
        self.spinBoxNeighborhoodSize.setMaximum(19)
        self.spinBoxNeighborhoodSize.setObjectName(\
            _fromUtf8("spinBoxNeighborhoodSize"))
        self.horizontalLayoutSplitNeighborhood.addWidget(\
            self.spinBoxNeighborhoodSize)
        self.verticalLayoutDruseSplitTools.addLayout(\
            self.horizontalLayoutSplitNeighborhood)
        self.radioButtonManualMarker = QtGui.QRadioButton()
        self.radioButtonManualMarker.setObjectName(\
            _fromUtf8("radioButtonManualMarker"))
        self.verticalLayoutDruseSplitTools.addWidget(\
            self.radioButtonManualMarker)
        self.horizontalLayoutSplitButtons = QtGui.QHBoxLayout()
        self.horizontalLayoutSplitButtons.setObjectName(\
            _fromUtf8("horizontalLayoutSplitButtons"))
        self.toolButtonRunWatershed = QtGui.QToolButton()
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding,\
            QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.toolButtonRunWatershed.sizePolicy().\
            hasHeightForWidth())
        self.toolButtonRunWatershed.setSizePolicy(sizePolicy)
        self.toolButtonRunWatershed.setMinimumSize(QtCore.QSize(34, 34))
        self.toolButtonRunWatershed.setObjectName(\
            _fromUtf8("toolButtonRunWatershed"))
        self.horizontalLayoutSplitButtons.addWidget(self.toolButtonRunWatershed)
        self.toolButtonMerge = QtGui.QToolButton()
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding,\
            QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.toolButtonMerge.sizePolicy().\
            hasHeightForWidth())
        self.toolButtonMerge.setSizePolicy(sizePolicy)
        self.toolButtonMerge.setMinimumSize(QtCore.QSize(34, 34))
        self.toolButtonMerge.setObjectName(_fromUtf8("toolButtonMerge"))
        self.horizontalLayoutSplitButtons.addWidget(self.toolButtonMerge)
        self.verticalLayoutDruseSplitTools.addLayout(\
            self.horizontalLayoutSplitButtons)
        
        self.verticalLayoutSeparationThreshold = QtGui.QVBoxLayout()
        self.verticalLayoutSeparationThreshold.setObjectName(\
            _fromUtf8("verticalLayoutSeparationThreshold"))
        self.labelSeparationThreshold = QtGui.QLabel()
        self.labelSeparationThreshold.setObjectName(\
            _fromUtf8("labelSeparationThreshold"))
        self.verticalLayoutSeparationThreshold.addWidget(\
            self.labelSeparationThreshold)
        self.horizontalLayoutSeparationThreshold = QtGui.QHBoxLayout()
        self.horizontalLayoutSeparationThreshold.setObjectName(\
            _fromUtf8("horizontalLayoutSeparationThreshold"))
        self.horizontalSliderSeparationThreshold = QtGui.QSlider()
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding,\
            QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.horizontalSliderSeparationThreshold.\
            sizePolicy().hasHeightForWidth())
        self.horizontalSliderSeparationThreshold.setSizePolicy(sizePolicy)
        self.horizontalSliderSeparationThreshold.setMinimumSize(QtCore.QSize(150, 0))
        self.horizontalSliderSeparationThreshold.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSliderSeparationThreshold.setObjectName(\
            _fromUtf8("horizontalSliderSeparationThreshold"))
        self.horizontalLayoutSeparationThreshold.addWidget(\
            self.horizontalSliderSeparationThreshold)
        self.spinBoxSeparationThreshold = QtGui.QSpinBox()
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed,\
            QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.spinBoxSeparationThreshold.\
            sizePolicy().hasHeightForWidth())
        self.spinBoxSeparationThreshold.setSizePolicy(sizePolicy)
        self.spinBoxSeparationThreshold.setMinimumSize(QtCore.QSize(51, 0))
        self.spinBoxSeparationThreshold.setObjectName(\
            _fromUtf8("spinBoxSeparationThreshold"))
        self.horizontalLayoutSeparationThreshold.addWidget(\
            self.spinBoxSeparationThreshold)
        self.verticalLayoutSeparationThreshold.addLayout(\
            self.horizontalLayoutSeparationThreshold)
        self.verticalLayoutDruseSplitTools.addLayout(\
            self.verticalLayoutSeparationThreshold)
        
        self.toolButtonApplySplit = QtGui.QToolButton()
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding,\
            QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.toolButtonApplySplit.sizePolicy().\
            hasHeightForWidth())
        self.toolButtonApplySplit.setSizePolicy(sizePolicy)
        self.toolButtonApplySplit.setMinimumSize(QtCore.QSize(34, 34))
        self.toolButtonApplySplit.setObjectName(_fromUtf8("toolButtonApplySplit"))
        self.verticalLayoutDruseSplitTools.addWidget(self.toolButtonApplySplit)
        
        self.toolButtonApplySplit.setText('Apply split')
        self.toolButtonMerge.setText('Merge drusen')
        self.toolButtonRunWatershed.setText('Segment')
        
        self.groupBoxDruseInfo = QtGui.QGroupBox()
        self.groupBoxDruseInfo.setGeometry(QtCore.QRect(50, 70, 161, 101))
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding,\
            QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBoxDruseInfo.sizePolicy().\
            hasHeightForWidth())
        self.groupBoxDruseInfo.setSizePolicy(sizePolicy)
        self.groupBoxDruseInfo.setMinimumSize(QtCore.QSize(161, 0))
        self.groupBoxDruseInfo.setObjectName(_fromUtf8("groupBoxDruseInfo"))
        self.verticalLayoutDruInfo = QtGui.QVBoxLayout(self.groupBoxDruseInfo)
        self.verticalLayoutDruInfo.setObjectName(_fromUtf8("verticalLayoutDruInfo"))
        self.labelInfoHeight = QtGui.QLabel(self.groupBoxDruseInfo)
        self.labelInfoHeight.setObjectName(_fromUtf8("labelInfoHeight"))
        self.verticalLayoutDruInfo.addWidget(self.labelInfoHeight)
        self.labelInfoVolume = QtGui.QLabel(self.groupBoxDruseInfo)
        self.labelInfoVolume.setObjectName(_fromUtf8("labelInfoVolume"))
        self.verticalLayoutDruInfo.addWidget(self.labelInfoVolume)
        self.labelInfoBrightness = QtGui.QLabel(self.groupBoxDruseInfo)
        self.labelInfoBrightness.setObjectName(_fromUtf8("labelInfoBrightness"))
        self.verticalLayoutDruInfo.addWidget(self.labelInfoBrightness)
        
        self.verticalLayout_7.addWidget(self.groupBox_toolBox)
        self.verticalLayout_7.addLayout(self.verticalLayout_6)
        self.verticalLayout_7.addLayout(self.verticalLayoutDruseSplitTools)
        self.verticalLayout_7.addWidget(self.groupBoxDruseInfo)
        if(not self.controller is None):
            self.controller.set_editing_layer(0)# RPE layer
        
#==============================================================================
# Connect events
#==============================================================================
        self.toolButtonPen.setCheckable(True)
        self.toolButtonLine.setCheckable(True)
        self.toolButtonFill.setCheckable(True)
        self.toolButtonMorphology.setCheckable(True)
        self.toolButtonDrawDru.setCheckable(True)
        self.toolButtonFilterDru.setCheckable(True)
        self.toolButtonGrab.setCheckable(True)
        self.toolButtonBBox.setCheckable(True)
        self.toolButtonCostPnt.setCheckable(True)
        self.toolButtonPolyFit.setCheckable(True)
        self.toolButtonSplitDrusen.setCheckable(True)
        self.toolButtonMerge.setCheckable(True)
        self.toolButtonCCA.setCheckable(True)
        
        self.toolButtonCCAClicked=False
        self.toolButtonMergeClicked=False
        self.toolButtonSplitDrusenClicked=False
        
        self.sizingToolsEnabled=False
        self.lastClickedButton=None
        self.morghologyValue=1
        self.thresholdAllValue=1
        self.thresholdMaxValue=2
        self.polyFitDegreeValue=2
        self.smoothnessValue=2
        
        self.druseSplitingNeighborhood=2
        self.druseSplittingMethod='localMax' #manual
        self.separationThreshold=100
        self.synchronize=0

        self.toolButtonPen.clicked.connect(self.pen_action)
        self.toolButtonLine.clicked.connect(self.line_action)
        self.toolButtonFill.clicked.connect(self.fill_action)
        self.toolButtonMorphology.clicked.connect(self.morphology_action)
        self.toolButtonDrawDru.clicked.connect(self.draw_dru_action)
        self.toolButtonFilterDru.clicked.connect(self.filter_dru_action)
        self.toolButtonGrab.clicked.connect(self.grab_action)
        self.toolButtonCCA.clicked.connect(self.cca_action)
        self.toolButtonCheckComp.clicked.connect(self.check_component_action)
        self.toolButtonUnCheckComp.clicked.connect(self.uncheck_component_action)
        self.toolButtonBBox.clicked.connect(self.bounding_box_action)
        self.toolButtonCostPnt.clicked.connect(self.cost_point_action)
        self.toolButtonPolyFit.clicked.connect(self.poly_fit_action)
        self.toolButtonNextComp.clicked.connect(self.next_component_action)
        self.toolButtonPrevComp.clicked.connect(self.prev_component_action)
        self.toolButtonFilterDruImm.clicked.connect(\
            self.apply_threshold_immediately_action)
        self.toolButtonSplitDrusen.clicked.connect(self.split_drusen_action)
        self.toolButtonRunWatershed.clicked.connect(self.run_watershed)
        self.toolButtonMerge.clicked.connect(self.merge_drusen)
        self.toolButtonApplySplit.clicked.connect(self.apply_split)
        self.radioButtonUseMaximaAsMarker.clicked.connect(\
            self.drusen_splitting_maxima_as_marker_selected)
        self.radioButtonManualMarker.clicked.connect(\
            self.drusen_splitting_manual_marker_selected)
        
        self.horizontalSliderBscan.valueChanged.connect(\
            self.b_scan_slider_value_changed)
        self.horizontalSliderLayerMap.valueChanged.connect(\
            self.layer_slider_value_changed)
        self.horizontalSliderEnface.valueChanged.connect(\
            self.enface_slider_value_changed)
        self.horizontalSliderSeparationThreshold.valueChanged.connect(\
            self.separation_slider_value_changed)
        
        self.horizontalSlider_size1.valueChanged.connect(\
            self.slider_1_value_changed)
        self.horizontalSlider_size2.valueChanged.connect(\
            self.slider_2_value_changed)
        self.spinBox_size1.valueChanged.connect(self.spinbox_1_value_changed)
        self.spinBox_size2.valueChanged.connect(self.spinbox_2_value_changed)
        self.spinBoxNeighborhoodSize.valueChanged.connect(\
            self.spinbox_neighborhood_size_changed)
        self.spinBoxSeparationThreshold.valueChanged.connect(\
            self.spinbox_separation_threshold_changed)
        
        self.radioButtonRPE.clicked.connect(self.rpe_editing_selected)
        self.radioButtonBM.clicked.connect(self.bm_editing_selected)
        self.radioButtonDrusen.clicked.connect(self.drusen_editing_selected)
        self.radioButtonEnface.clicked.connect(self.enface_editing_selected)
        self.radioButtonHRF.clicked.connect(self.hrf_editing_selected)
        self.radioButtonGA.clicked.connect(self.ga_editing_selected)
        
        self.disable_all()
        self.enable_layer_editing_tools()

        self.retranslateUi(toolBox)
        QtCore.QMetaObject.connectSlotsByName(toolBox)

        self.b_scan_slider_value_changed()
        self.layer_slider_value_changed()
        self.enface_slider_value_changed()
        
        if(self.druseSplittingMethod=='localMax'):
            self.radioButtonUseMaximaAsMarker.setChecked(True)
        elif(self.druseSplittingMethod=='manual'):
            self.radioButtonManualMarker.setChecked(True)
            
    def retranslateUi(self, toolBox):
        toolBox.setWindowTitle(_translate("toolBox", "Toolbox", None))
        self.groupBox_edit.setTitle(_translate("toolBox", "Edit", None))
        self.radioButtonRPE.setText(_translate("toolBox", "RPE layer", None))
        self.radioButtonEnface.setText(_translate("toolBox", "Enface", None))
        self.radioButtonBM.setText(_translate("toolBox", "BM layer", None))
        self.radioButtonHRF.setText(_translate("toolBox", "HRF", None))
        self.radioButtonDrusen.setText(_translate("toolBox", "Drusen", None))
        self.radioButtonGA.setText(_translate("toolBox", "GA", None))
        
        self.groupBox_opacity.setTitle(_translate("toolBox", "Opacity", None))
        self.label.setText(_translate("toolBox", "B-Scan", None))
        self.label_2.setText(_translate("toolBox", "Layer map", None))
        self.label_3.setText(_translate("toolBox", "Drusen map", None))
        self.label_4.setText(_translate("toolBox", "Enface", None))
        self.label_5.setText(_translate("toolBox", "Enface drusen", None))
        self.groupBox_toolBox.setTitle(_translate("toolBox", "Toolbox", None))
        self.toolButtonPen.setText(_translate("toolBox", "...", None))
        self.toolButtonLine.setText(_translate("toolBox", "...", None))
        self.toolButtonFill.setText(_translate("toolBox", "...", None))
        self.toolButtonCCA.setText(_translate("toolBox", "...", None))
        self.toolButtonNextComp.setText(_translate("toolBox", "...", None))
        self.toolButtonPrevComp.setText(_translate("toolBox", "...", None))
        self.toolButtonUnCheckComp.setText(_translate("toolBox", "...", None))
        self.toolButtonBBox.setText(_translate("toolBox", "...", None))
        self.toolButtonCostPnt.setText(_translate("toolBox", "...", None))
        self.toolButtonPolyFit.setText(_translate("toolBox", "...", None))
        self.toolButtonCheckComp.setText(_translate("toolBox", "...", None))
        self.toolButtonMorphology.setText(_translate("toolBox", "...", None))
        self.toolButtonDrawDru.setText(_translate("toolBox", "...", None))
        self.toolButtonFilterDru.setText(_translate("toolBox", "...", None))
        self.toolButtonSplitDrusen.setText(_translate("toolBox","...",None))
        self.toolButtonGrab.setText(_translate("toolBox", "...", None))
        self.label_size1.setText(_translate("toolBox", "t-all", None))
        self.label_7.setText(_translate("toolBox", "px", None))
        self.label_size2.setText(_translate("toolBox", "t-max", None))
        self.label_9.setText(_translate("toolBox", "px", None))
        
        self.radioButtonUseMaximaAsMarker.setText(\
            _translate("Form", "Use local maximum as druse peak", None))
        self.labelNeighborhoodSize.setText(\
            _translate("Form", "Neighborhood size: ", None))
        self.radioButtonManualMarker.setText(\
            _translate("Form", "Mark drusen peaks", None))
        self.toolButtonRunWatershed.setText(_translate("Form", "Segment", None))
        self.toolButtonMerge.setText(_translate("Form", "Merge drusen", None))
        self.toolButtonApplySplit.setText(_translate("Form", "Apply split", None))

        self.groupBoxDruseInfo.setTitle(_translate("Form", "Druse Info", None))
        self.labelInfoHeight.setText(_translate("Form", "Height:", None))
        self.labelInfoVolume.setText(_translate("Form", "Volume:", None))
        self.labelInfoBrightness.setText(_translate("Form", "Brightness:", None))
        
        self.labelSeparationThreshold.setText(\
            _translate("Form", "Remove separation lines with height >", None))
        
        self.toolButtonPen.setToolTip("Pencil Tool: Hard edge marking with"+\
            " left click.\n Erase marking with right click.")
        self.toolButtonLine.setToolTip("Line Tool: Draw line with left click"+\
            ".\n Erase line with right click.\nMark regions with geographic"+\
            " atrophy.")
        self.toolButtonFill.setToolTip("Bucket Fill Tool: Fill selected area "+\
            "with left click. (In drusen map editor, it fills only up to RPE "+\
            "layer). \nErase selected area with right click.")
        self.toolButtonCCA.setToolTip("Druse Trace Tool: Color drusen in "+\
            "enface drusen editor.\nGreen: Checked drusen. \nYellow: Current"+\
            " druse being edited.\nMagenta: Not yet checked drusen.")
        self.toolButtonNextComp.setToolTip("Next Druse: Select next largest "+\
            "druse in drusen annotation procedure.")
        self.toolButtonPrevComp.setToolTip("Previous Druse: Select previous "+\
            "largest druse in drusen annotation procedure.")
        
        self.toolButtonBBox.setToolTip("Bounding Box Tool: Draw bounding box"+\
            " with left click for hyperreflective foci or reticular drusen "+\
            "annotation.\nRemove bounding boxes with right click.")
        self.toolButtonCostPnt.setToolTip("Probability Modifier Tool: Enforce"+\
            " shortest path to run through the selected point.")
        self.toolButtonPolyFit.setToolTip("Polynomial Fitting Tool: Fit a "+\
            "polynomial to layers. ")
        self.toolButtonSplitDrusen.setToolTip("Druse Splitting Tool: Split "+\
            "the druse into smaller ones.")
        self.toolButtonCheckComp.setToolTip("Check Druse: Check druse "+\
            "annotation as correct.")
        self.toolButtonUnCheckComp.setToolTip("Uncheck Druse: Uncheck druse"+\
            " annotation as incorrect.")
        
        self.toolButtonMorphology.setToolTip("Morphological Filter Tool: "+\
            "Enlarge selected area with left click.\nErode selected area "+\
            "with right click.\nUse size slider to set tool strength.")
                                             
        self.toolButtonDrawDru.setToolTip("Auto Annotate Tool: Automatically"+\
            " draws druse/HRF in selected area with left click.\nErase "+\
            "druse/HRF in selected area.")
        self.toolButtonFilterDru.setToolTip("Filter Small Drusen Tool: "+\
            "Filter drusen with respect to their height in the selected area.\n"+\
            "t-all: Filter out columns that have overall number of pixels "+\
            "below t-all.\nt-max: Filter out any druse that has a height "+\
            "below t-max.\nApply Immediately: Apply the filter all over the"+\
            " B-scan for drusen map editor, \nand all over OCT scan for "+\
            "enface drusen editor.")
                                        
        self.toolButtonGrab.setToolTip("Select Tool: Select a druse for "+\
            "editing in the enface drusen editor.\nSelect corresponding "+\
            "locations between drusen map and enface drusen editors.")

    def check_component_action(self):
        if(self.toolButtonCCA.isChecked()):
            self.controller.write_in_log(self.controller.get_time()+','+\
                self.toolButtonCheckComp.objectName()+','+\
                self.get_current_active_window()+'\n')
            self.controller.check_component()
            
    def uncheck_component_action(self):
        if(self.toolButtonCCA.isChecked()):
            self.controller.write_in_log(self.controller.get_time()+','+\
                self.toolButtonUnCheckComp.objectName()+','+\
                self.get_current_active_window()+'\n')
            self.controller.uncheck_component()
            
    def next_component_action(self):
        if(self.toolButtonCCA.isChecked()):
            self.controller.write_in_log(self.controller.get_time()+','+\
                self.toolButtonNextComp.objectName()+','+\
                self.get_current_active_window()+'\n')
            self.controller.next_component()
            
    def prev_component_action(self):
        if(self.toolButtonCCA.isChecked()):
            self.controller.write_in_log(self.controller.get_time()+','+\
                self.toolButtonPrevComp.objectName()+','+\
                self.get_current_active_window()+'\n')
            self.controller.prev_component()
        
    def cca_action(self):
        self.toolButtonCCAClicked=True if not self.toolButtonCCAClicked else False
        self.toolButtonCCA.setChecked(self.toolButtonCCAClicked)
        if(self.toolButtonCCA.isChecked()):
            self.enable_cca_tools()
            self.controller.write_in_log(self.controller.get_time()+','+\
                self.toolButtonCCA.objectName()+','+\
                self.get_current_active_window()+'\n')
            self.controller.show_CCA_enface()
        else:
            self.disable_cca_tools()
            self.controller.hide_CCA_enface()
            
    def apply_threshold_immediately_action(self):
        if(self.lastClickedButton is self.toolButtonFilterDru):
            self.controller.write_in_log(self.controller.get_time()+','+\
            self.toolButtonFilterDruImm.objectName()+','+\
            self.get_current_active_window()+'\n')
            self.controller.apply_threshold_immediately()
        elif(self.lastClickedButton is self.toolButtonSplitDrusen):
            self.controller.apply_splitting_threshold()

    def pen_action(self):
        if(self.sizingToolsEnabled):
            self.disable_size_tools()
        if(self.lastClickedButton is not None and self.lastClickedButton is\
                not self.toolButtonPen):
            self.lastClickedButton.setChecked(False)
        self.toolButtonPen.setChecked(True)
        self.lastClickedButton=self.toolButtonPen
        self.controller.write_in_log(self.controller.get_time()+','+\
            self.toolButtonPen.objectName()+','+\
            self.get_current_active_window()+'\n')
        self.controller.activate_pen()
        
    def line_action(self):
        if(self.sizingToolsEnabled):
            self.disable_size_tools()
        if(self.lastClickedButton is not None and self.lastClickedButton is\
                not self.toolButtonLine):
            self.lastClickedButton.setChecked(False)
        self.toolButtonLine.setChecked(True)
        self.lastClickedButton=self.toolButtonLine
        self.controller.write_in_log(self.controller.get_time()+','+\
            self.toolButtonLine.objectName()+','+\
            self.get_current_active_window()+'\n')
        self.controller.activate_line()
        
    def fill_action(self):
        if(self.sizingToolsEnabled):
            self.disable_size_tools()
        if(self.lastClickedButton is not None and self.lastClickedButton is\
                not self.toolButtonFill):
            self.lastClickedButton.setChecked(False)
        self.toolButtonFill.setChecked(True)
        self.lastClickedButton=self.toolButtonFill
        self.controller.write_in_log(self.controller.get_time()+','+\
            self.toolButtonFill.objectName()+','+\
            self.get_current_active_window()+'\n')
        self.controller.activate_fill()
        
    def morphology_action(self):
        self.hide_druse_split_tools()
        if(self.sizingToolsEnabled):
            self.disable_size_tools()
        if(self.lastClickedButton is not None and self.lastClickedButton is\
                not self.toolButtonMorphology):
            self.lastClickedButton.setChecked(False)
        self.toolButtonMorphology.setChecked(True)
        self.lastClickedButton=self.toolButtonMorphology
        self.controller.write_in_log(self.controller.get_time()+','+\
            self.toolButtonMorphology.objectName()+','+\
            self.get_current_active_window()+'\n')
        self.enable_morphology_tools()
        self.controller.activate_morphology(self.morghologyValue)
    
    def draw_dru_action(self):
        if(self.sizingToolsEnabled):
            self.disable_size_tools()
        if(self.lastClickedButton is not None and self.lastClickedButton is\
                not self.toolButtonDrawDru):
            self.lastClickedButton.setChecked(False)
        self.toolButtonDrawDru.setChecked(True)
        self.lastClickedButton=self.toolButtonDrawDru
        self.controller.write_in_log(self.controller.get_time()+','+\
            self.toolButtonDrawDru.objectName()+','+\
            self.get_current_active_window()+'\n')
        self.controller.activate_draw_dru()
        
    def filter_dru_action(self):
        self.hide_druse_split_tools()
        if(self.sizingToolsEnabled):
            self.disable_size_tools()
        if(self.lastClickedButton is not None and self.lastClickedButton is\
                not self.toolButtonFilterDru):
            self.lastClickedButton.setChecked(False)
        self.toolButtonFilterDru.setChecked(True)
        self.lastClickedButton=self.toolButtonFilterDru  
        self.enable_dru_filter_tools()
        self.controller.write_in_log(self.controller.get_time()+','+\
            self.toolButtonFilterDru.objectName()+','+\
            self.get_current_active_window()+'\n')
        self.controller.activate_filter_dru(self.thresholdAllValue,\
            self.thresholdMaxValue)
        
    def split_drusen_action(self):
        if(self.sizingToolsEnabled):
            self.disable_size_tools()
        if(self.lastClickedButton is not None and self.lastClickedButton is\
                not self.toolButtonSplitDrusen):
            self.lastClickedButton.setChecked(False)
        self.toolButtonSplitDrusenClicked=True if not\
            self.toolButtonSplitDrusenClicked else False
        self.toolButtonSplitDrusen.setChecked(self.toolButtonSplitDrusenClicked)
        if(self.toolButtonSplitDrusenClicked):
            self.controller.write_in_log(self.controller.get_time()+','+\
                self.toolButtonSplitDrusen.objectName()+','+\
                self.get_current_active_window()+'\n')
            self.enable_dru_spliting_tools()
        else:
            self.controller.done_spliting()
            self.hide_druse_split_tools()
        self.lastClickedButton=self.toolButtonSplitDrusen
       
    def grab_action(self):
        if(self.sizingToolsEnabled):
            self.disable_size_tools()
        if(self.lastClickedButton is not None and self.lastClickedButton is\
                not self.toolButtonGrab):
            self.lastClickedButton.setChecked(False)
        self.toolButtonGrab.setChecked(True)
        self.lastClickedButton=self.toolButtonGrab
        self.controller.write_in_log(self.controller.get_time()+','+\
            self.toolButtonGrab.objectName()+','+\
            self.get_current_active_window()+'\n')
        self.controller.activate_grab()
        
    def bounding_box_action(self):
        if(self.sizingToolsEnabled):
            self.disable_size_tools()
        if(self.lastClickedButton is not None and self.lastClickedButton is\
                not self.toolButtonBBox):
            self.lastClickedButton.setChecked(False)
        self.toolButtonBBox.setChecked(True)
        self.lastClickedButton=self.toolButtonBBox
        self.controller.write_in_log(self.controller.get_time()+','+\
            self.toolButtonBBox.objectName()+','+\
            self.get_current_active_window()+'\n')
        self.controller.activate_bounding_box()
        
    def cost_point_action(self):
        if(self.sizingToolsEnabled):
            self.disable_size_tools()
        if(self.lastClickedButton is not None and self.lastClickedButton is\
                not self.toolButtonCostPnt):
            self.lastClickedButton.setChecked(False)
        self.toolButtonCostPnt.setChecked(True)
        self.enable_cost_point_tools()
        self.lastClickedButton=self.toolButtonCostPnt
        self.controller.write_in_log(self.controller.get_time()+','+\
            self.toolButtonCostPnt.objectName()+','+\
            self.get_current_active_window()+'\n')
        self.controller.activate_cost_point(self.smoothnessValue)
        
    def poly_fit_action(self):
        if(self.sizingToolsEnabled):
            self.disable_size_tools()
        if(self.lastClickedButton is not None and self.lastClickedButton is\
                not self.toolButtonPolyFit):
            self.lastClickedButton.setChecked(False)
        self.toolButtonPolyFit.setChecked(True)
        self.enable_polyfit_tools()
        self.lastClickedButton=self.toolButtonPolyFit
        self.controller.write_in_log(self.controller.get_time()+','+\
            self.toolButtonPolyFit.objectName()+','+\
            self.get_current_active_window()+'\n')
        self.controller.activate_poly_fit(self.polyFitDegreeValue)
        
    def run_watershed(self):
        self.toolButtonMerge.setChecked(False)
        self.controller.run_watershed()
        
    def merge_drusen(self):
        self.toolButtonMergeClicked=True if not self.toolButtonMergeClicked else False
        self.toolButtonMerge.setChecked(self.toolButtonMergeClicked)
        if(self.toolButtonMergeClicked):
            self.controller.set_merge_drusen()
            self.controller.write_in_log(self.controller.get_time()+','+\
                self.toolButtonMerge.objectName()+','+\
                self.get_current_active_window()+'\n')
        else:
            self.controller.unset_merge_drusen()
            
    def get_current_active_watershed_method(self):
        if(self.radioButtonUseMaximaAsMarker.isChecked()):
            return 'Maxima'
        if(self.radioButtonManualMarker.isChecked()):
            return 'Manual'
        return 'None'
        
    def apply_split(self):
        self.controller.write_in_log(self.controller.get_time()+','+\
            self.toolButtonApplySplit.objectName()+','+\
            self.get_current_active_window()+','+\
            self.get_current_active_watershed_method()+'\n')
        self.toolButtonMerge.setChecked(False)
        self.toolButtonMergeClicked=False
        self.controller.apply_split()
        
    def set_druse_info(self,height,volume,brightness):
        self.labelInfoHeight.setText('Height: '+str(height)+'   px')
        self.labelInfoVolume.setText('Volume: '+str(volume)+'   px^3')
        self.labelInfoBrightness.setText('Brightness: '+str(round(brightness,1)))
        
    def clear_druse_info(self):
        self.labelInfoHeight.setText('Height:')
        self.labelInfoVolume.setText('Volume:')
        self.labelInfoBrightness.setText('Brightness:')
        
    def hide_druse_info(self):
        self.groupBoxDruseInfo.hide()
        
        if(self.toolButtonSplitDrusen.isChecked()):
            self.toolBox.setMinimumSize(QtCore.QSize(249, 650))
            self.toolBox.resize(249, 650)
        else:
            self.toolBox.setMinimumSize(QtCore.QSize(249, 450))
            self.toolBox.resize(249,450)
        self.toolBox.update()
        self.update_further()
        
    def show_druse_info(self):
        self.groupBoxDruseInfo.show()
        
        if(self.toolButtonSplitDrusen.isChecked()):
            self.toolBox.setMinimumSize(QtCore.QSize(249, 750))
            self.toolBox.resize(249, 750)
        else:
            self.toolBox.setMinimumSize(QtCore.QSize(249, 550))
            self.toolBox.resize(249, 550)
            
        self.toolBox.update()
        self.update_further()
        
    def hide_druse_split_tools(self):
        self.radioButtonUseMaximaAsMarker.hide()
        self.labelNeighborhoodSize.hide()
        self.radioButtonManualMarker.hide()
        self.toolButtonRunWatershed.hide()
        self.toolButtonMerge.hide()
        self.toolButtonApplySplit.hide()
        self.spinBoxNeighborhoodSize.hide()
        self.labelSeparationThreshold.hide()
        self.spinBoxSeparationThreshold.hide()
        self.horizontalSliderSeparationThreshold.hide()

        self.toolBox.setMinimumSize(QtCore.QSize(249, 550))
        self.toolBox.resize(249, 550)
        self.toolBox.update()
        self.update_further()
        
    def show_druse_split_tools(self):
        self.radioButtonUseMaximaAsMarker.show()
        self.labelNeighborhoodSize.show()
        self.radioButtonManualMarker.show()
        self.toolButtonRunWatershed.show()
        self.toolButtonMerge.show()
        self.toolButtonApplySplit.show() 
        self.spinBoxNeighborhoodSize.show()
        self.labelSeparationThreshold.show()
        self.spinBoxSeparationThreshold.show()
        self.horizontalSliderSeparationThreshold.show()
        self.toolBox.setMinimumSize(QtCore.QSize(249, 750))
        self.toolBox.resize(249, 750)
        self.toolBox.update()
        self.update_further()
        
    def disable_all(self):
        self.toolButtonPen.setDisabled(True)
        self.toolButtonLine.setDisabled(True)
        self.toolButtonFill.setDisabled(True)
        self.toolButtonCCA.setDisabled(True)
        self.toolButtonNextComp.setDisabled(True)
        self.toolButtonPrevComp.setDisabled(True)
        self.toolButtonUnCheckComp.setDisabled(True)
        self.toolButtonCheckComp.setDisabled(True)
        self.toolButtonMorphology.setDisabled(True)
        self.toolButtonDrawDru.setDisabled(True)
        self.toolButtonFilterDru.setDisabled(True)
        self.toolButtonSplitDrusen.setDisabled(True)
        self.toolButtonGrab.setDisabled(True)
        self.toolButtonBBox.setDisabled(True)
        self.toolButtonCostPnt.setDisabled(True)
        self.toolButtonPolyFit.setDisabled(True)
        
        self.hide_druse_split_tools()
        self.hide_druse_info()
        
        self.label.hide()
        self.label_2.hide()
        self.label_3.hide()
        self.label_4.hide()
        self.label_5.hide()
        
        self.horizontalSliderBscan.hide()
        self.horizontalSliderLayerMap.hide()
        self.horizontalSliderDrusenMap.hide()
        self.horizontalSliderEnface.hide()
        self.horizontalSliderEnfaceDrusen.hide()
        
        self.label_size1.hide()
        self.label_7.hide()
        self.label_size2.hide()
        self.label_9.hide()
    
        self.horizontalSlider_size1.hide()
        self.spinBox_size1.hide()
        self.horizontalSlider_size2.hide()
        self.spinBox_size2.hide()

        self.toolButtonFilterDruImm.hide()
        
    def enable_layer_editing_tools(self):

        self.toolButtonPen.setDisabled(False)
        self.toolButtonLine.setDisabled(False)
        if(self.enable_probability_related_tools):
            self.toolButtonCostPnt.setDisabled(False)
            self.toolButtonPolyFit.setDisabled(False)
        
        self.label.show()   
        self.horizontalSliderBscan.show()
        
        self.toolBox.setMinimumSize(QtCore.QSize(249, 450))
        self.toolBox.resize(249, 450)        
        self.toolBox.update()
        self.update_further()
    
    def enable_cca_tools(self):
        self.toolButtonNextComp.setDisabled(False)
        self.toolButtonPrevComp.setDisabled(False)
        self.toolButtonUnCheckComp.setDisabled(False)
        self.toolButtonCheckComp.setDisabled(False)
        self.toolButtonSplitDrusen.setDisabled(False)
        self.show_druse_info()
        
    def disable_cca_tools(self):
        self.toolButtonNextComp.setDisabled(True)
        self.toolButtonPrevComp.setDisabled(True)
        self.toolButtonUnCheckComp.setDisabled(True)
        self.toolButtonCheckComp.setDisabled(True)
        self.toolButtonSplitDrusen.setDisabled(True)
        self.hide_druse_info()
        
    def enable_drusen_editing_tools(self):
        self.disable_all()
        self.toolButtonPen.setDisabled(False)
        self.toolButtonLine.setDisabled(False)       

        self.toolButtonDrawDru.setDisabled(False)
        self.toolButtonFill.setDisabled(False)
        self.toolButtonMorphology.setDisabled(False)
        self.toolButtonFilterDru.setDisabled(False)
        
        self.toolButtonGrab.setDisabled(False)
        
        self.label.show()
        self.label_2.show()        
        self.horizontalSliderBscan.show()
        self.horizontalSliderLayerMap.show()
        
        self.toolBox.setMinimumSize(QtCore.QSize(249, 450))
        self.toolBox.resize(249, 450)
        self.toolBox.update()
        self.update_further()
        
    def update_further(self):
        self.controller.update_tool_box()
        
    def disable_size_tools(self):
        
        self.label_size1.hide()
        self.label_7.hide()
        self.label_size2.hide()
        self.label_9.hide()
    
        self.horizontalSlider_size1.hide()
        self.spinBox_size1.hide()
        self.horizontalSlider_size2.hide()
        self.spinBox_size2.hide()
        self.toolButtonFilterDruImm.hide()
        self.sizingToolsEnabled=False
        
        self.toolBox.setMinimumSize(QtCore.QSize(249, 450))
        self.toolBox.resize(249, 450)
        self.toolBox.update()
        self.update_further()
        
    def enable_morphology_tools(self):
        
        self.horizontalSlider_size1.setValue(self.morghologyValue)
        self.spinBox_size1.setValue(self.morghologyValue)
        self.label_size1.setText(_translate("toolBox", "Size", None))
        self.label_size1.show()
        self.horizontalSlider_size1.show()
        self.label_7.show()
        self.spinBox_size1.show()
        self.sizingToolsEnabled=True
        
        self.toolBox.setMinimumSize(QtCore.QSize(249, 500))
        self.toolBox.resize(249, 500)
        self.toolBox.update()
        self.update_further()
    
    def enable_dru_spliting_tools(self):
        self.show_druse_split_tools()
        
    def enable_cost_point_tools(self):
        self.horizontalSlider_size1.setValue(self.smoothnessValue)
        self.spinBox_size1.setValue(self.smoothnessValue)
        self.label_size1.setText(_translate("toolBox", "Smoothness", None))
        self.label_size1.show()
        self.horizontalSlider_size1.show()
        self.label_7.show()
        self.spinBox_size1.show()
        self.sizingToolsEnabled=True
        
        self.toolBox.setMinimumSize(QtCore.QSize(249, 500))
        self.toolBox.resize(249, 500)
        self.toolBox.update()
        self.update_further()
        
    def enable_polyfit_tools(self):
        self.horizontalSlider_size1.setValue(self.polyFitDegreeValue)
        self.spinBox_size1.setValue(self.polyFitDegreeValue)
        self.label_size1.setText(_translate("toolBox", "Degree", None))
        self.label_size1.show()
        self.horizontalSlider_size1.show()
        self.label_7.show()
        self.spinBox_size1.show()
        self.sizingToolsEnabled=True
        
        self.toolBox.setMinimumSize(QtCore.QSize(249, 500))
        self.toolBox.resize(249, 500)
        self.toolBox.update()
        self.update_further()
        
    def enable_dru_filter_tools(self):        
        self.horizontalSlider_size1.setValue(self.thresholdAllValue)
        self.spinBox_size1.setValue(self.thresholdAllValue)
        self.label_size1.setText(_translate("toolBox", "t-all", None))
        self.label_size1.show()
        self.horizontalSlider_size1.show()
        self.label_7.show()
        self.spinBox_size1.show()
        
        self.label_size2.show()
        self.horizontalSlider_size2.show()
        self.label_9.show()
        self.spinBox_size2.show()
        self.toolButtonFilterDruImm.show()
        self.sizingToolsEnabled=True
        
        self.toolBox.setMinimumSize(QtCore.QSize(249, 600))
        self.toolBox.resize(249, 600)
        self.toolBox.update()
        self.update_further()
     
    def enable_enface_editing_tools(self):
        self.disable_all()
        self.toolButtonBBox.setDisabled(False)
        self.toolButtonGrab.setDisabled(False)       
        self.toolButtonBBox.setDisabled(False)  
        self.label_4.show()
        self.horizontalSliderEnface.show()    
        
    def enable_enface_drusen_editing_tools(self):
        self.disable_all()
        
        self.toolButtonPen.setDisabled(False)
        self.toolButtonLine.setDisabled(False)
        self.toolButtonCCA.setDisabled(False)
        if(self.toolButtonCCA.isChecked()):
            self.enable_cca_tools()
        
        self.toolButtonDrawDru.setDisabled(False)
        self.toolButtonFill.setDisabled(False)
        self.toolButtonMorphology.setDisabled(False)
        self.toolButtonFilterDru.setDisabled(False)
        self.toolButtonGrab.setDisabled(False)
        self.label_4.show()
        self.horizontalSliderEnface.show()    
        
    def enable_HRF_editing_tools(self):
        self.toolButtonPen.setDisabled(False)
        self.toolButtonLine.setDisabled(False)
        self.toolButtonBBox.setDisabled(False)
        
        self.toolButtonDrawDru.setDisabled(False)
        self.toolButtonFill.setDisabled(False)
        self.toolButtonMorphology.setDisabled(False)
        self.toolButtonBBox.setDisabled(False)     
        self.label.show()   
        self.horizontalSliderBscan.show()
        
        self.toolBox.setMinimumSize(QtCore.QSize(249, 450))
        self.toolBox.resize(249, 450)        
        self.toolBox.update()
        self.update_further()
        
    def enable_GA_editing_tools(self):
        self.toolButtonLine.setDisabled(False)
        self.label.show()   
        self.horizontalSliderBscan.show()
        
        self.toolBox.setMinimumSize(QtCore.QSize(249, 450))
        self.toolBox.resize(249, 450)        
        self.toolBox.update()
        self.update_further()

    def get_current_active_window(self):
        if(self.radioButtonRPE.isChecked()):
            return 'LayersRPE'
        if(self.radioButtonBM.isChecked()):
            return 'LayersBM'
        if(self.radioButtonDrusen.isChecked()):
            return 'Drusen'    
        if(self.radioButtonEnface.isChecked()):
            return 'Enface'
        if(self.radioButtonGA.isChecked()):
            return 'GA'
        if(self.radioButtonHRF.isChecked()):
            return 'HRF'
        return 'None'
        
    def rpe_editing_selected(self):
        self.disable_all()
        if(not self.radioButtonRPE.isChecked()):
            self.radioButtonRPE.setChecked(True)
        self.controller.unset_editing_layer()
        self.enable_layer_editing_tools()
        self.controller.set_editing_layer(0)
        if(self.toolButtonPolyFit.isChecked()):
            self.enable_polyfit_tools()
        if(self.toolButtonCostPnt.isChecked()):
            self.enable_cost_point_tools()
            
    def bm_editing_selected(self):
        self.disable_all()
        if(not self.radioButtonBM.isChecked()):
            self.radioButtonBM.setChecked(True)
        self.controller.unset_editing_layer()
        self.enable_layer_editing_tools()
        self.controller.set_editing_layer(1)
        if(self.toolButtonPolyFit.isChecked()):
            self.enable_polyfit_tools()
        if(self.toolButtonCostPnt.isChecked()):
            self.enable_cost_point_tools()
            
    def drusen_editing_selected(self):
        self.disable_all()
        if(not self.radioButtonDrusen.isChecked()):
            self.radioButtonDrusen.setChecked(True)
        self.controller.unset_editing_layer()
        self.enable_drusen_editing_tools()
        if(self.toolButtonFilterDru.isChecked()):
            self.enable_dru_filter_tools()
        if(self.toolButtonMorphology.isChecked()):
            self.enable_morphology_tools()
            
    def enface_editing_selected(self):
        self.disable_all()
        if(not self.radioButtonEnface.isChecked()):
            self.radioButtonEnface.setChecked(True)
        self.controller.unset_editing_layer()
        self.enable_enface_editing_tools()
        
    def enface_drusen_editing_selected(self):
        self.disable_all()
        if(not self.radioButtonEnface.isChecked()):
            self.radioButtonEnface.setChecked(True)
        self.controller.unset_editing_layer()
        self.enable_enface_drusen_editing_tools()
        if(self.toolButtonCCA.isChecked()):
            self.enable_cca_tools()
        if(self.toolButtonSplitDrusen.isChecked()):
            self.show_druse_split_tools()
        if(self.toolButtonFilterDru.isChecked()):
            self.enable_dru_filter_tools()
        if(self.toolButtonMorphology.isChecked()):
            self.enable_morphology_tools()
            
    def hrf_editing_selected(self):
        self.disable_all()
        if(not self.radioButtonHRF.isChecked()):
            self.radioButtonHRF.setChecked(True)
        self.controller.unset_editing_layer()
        self.enable_HRF_editing_tools()
    
    def ga_editing_selected(self):
        self.disable_all()
        if(not self.radioButtonGA.isChecked()):
            self.radioButtonGA.setChecked(True)
        self.controller.unset_editing_layer()
        self.enable_GA_editing_tools()
    
    def drusen_splitting_maxima_as_marker_selected(self):
        self.radioButtonManualMarker.setChecked(False)
        self.radioButtonUseMaximaAsMarker.setChecked(True)
        self.druseSplittingMethod='localMax'
        
    def drusen_splitting_manual_marker_selected(self):
        
        self.radioButtonUseMaximaAsMarker.setChecked(False)
        self.radioButtonManualMarker.setChecked(True)
        self.druseSplittingMethod='manual'
        self.controller.set_up_manual_marker_selection()
            
    def slider_1_value_changed(self):
        value=self.horizontalSlider_size1.value()
        self.spinBox_size1.setValue(value)
        if(self.lastClickedButton is self.toolButtonFilterDru):
            self.thresholdAllValue=value
            self.controller.all_threshold_value_changed(self.thresholdAllValue)
        elif(self.lastClickedButton is self.toolButtonMorphology):
            self.morghologyValue=value
            self.controller.morphology_value_changed(self.morghologyValue)
        elif(self.lastClickedButton is self.toolButtonPolyFit):
            self.polyFitDegreeValue=value
            self.controller.poly_fit_degree_value_changed(self.polyFitDegreeValue)
        elif(self.lastClickedButton is self.toolButtonCostPnt):
            self.smoothnessValue=value
            self.controller.smoothness_value_changed(self.smoothnessValue)
            
    def slider_2_value_changed(self):
        value=self.horizontalSlider_size2.value()
        self.spinBox_size2.setValue(value)
        if(self.lastClickedButton is self.toolButtonFilterDru):
            self.thresholdMaxValue=value
            self.controller.max_threshold_value_changed(self.thresholdMaxValue)
            
    def spinbox_1_value_changed(self):
        value=self.spinBox_size1.value()
        self.horizontalSlider_size1.setValue(value)
        if(self.lastClickedButton is self.toolButtonFilterDru):
            self.thresholdAllValue=value
            self.controller.all_threshold_value_changed(self.thresholdAllValue)
        elif(self.lastClickedButton is self.toolButtonMorphology):
            self.morghologyValue=value
            self.controller.morphology_value_changed(self.morghologyValue)
        elif(self.lastClickedButton is self.toolButtonPolyFit):
            self.polyFitDegreeValue=value
            self.controller.poly_fit_degree_value_changed(self.polyFitDegreeValue)
        elif(self.lastClickedButton is self.toolButtonCostPnt):
            self.smoothnessValue=value
            self.controller.smoothness_value_changed(self.smoothnessValue)
            
    def spinbox_neighborhood_size_changed(self):
        value=self.spinBoxNeighborhoodSize.value()
        self.druseSplitingNeighborhood=value
    
    def spinbox_separation_threshold_changed(self):
        self.synchronize=self.synchronize+1
        value=self.spinBoxSeparationThreshold.value()
        if(self.synchronize==1):
            self.horizontalSliderSeparationThreshold.setValue(value)
            self.controller.separation_theshold_changed(value)
            self.separationThreshold=value
        elif(self.synchronize==2):
            self.synchronize=0
    def get_neighborhood_size(self):
        return self.druseSplitingNeighborhood
        
    def get_splitting_method(self):
        return self.druseSplittingMethod
        
    def spinbox_2_value_changed(self):
        value=self.spinBox_size2.value()
        self.horizontalSlider_size2.setValue(value)
        if(self.lastClickedButton is self.toolButtonFilterDru):
            self.thresholdMaxValue=value
            self.controller.max_threshold_value_changed(self.thresholdMaxValue)
            
    def get_active_edit_index(self):
        if(self.radioButtonRPE.isChecked()):
            return 0
        elif(self.radioButtonBM.isChecked()):
            return 1
        elif(self.radioButtonDrusen.isChecked()):
            return 2
        elif(self.radioButtonEnface.isChecked()):
            return 3
        elif(self.radioButtonHRF.isChecked()):
            return 4
        elif(self.radioButtonGA.isChecked()):
            return 5
        else:
            return -1
            
    def b_scan_slider_value_changed(self):
        index=self.get_active_edit_index()
        self.controller.slider_value_changed(\
            self.horizontalSliderBscan.value(),'scan',index)
        
    def layer_slider_value_changed(self):
        index=self.get_active_edit_index()
        self.controller.slider_value_changed(\
            self.horizontalSliderLayerMap.value(),'layer',index)
        
    def enface_slider_value_changed(self):
        index=self.get_active_edit_index()
        self.controller.slider_value_changed(\
            self.horizontalSliderEnface.value(),'enface',index)
    
    def separation_slider_value_changed(self):
        self.synchronize=self.synchronize+1
        self.separationThreshold=self.horizontalSliderSeparationThreshold.value()
        if(self.synchronize==1):
            self.spinBoxSeparationThreshold.setValue(\
                self.horizontalSliderSeparationThreshold.value())
            self.controller.separation_theshold_changed(self.separationThreshold)
        elif(self.synchronize==2):
            self.synchronize=0
        
    def set_separation_threshold_range(self,minVal,maxVal):
        self.spinBoxSeparationThreshold.setMinimum(minVal-1)
        self.horizontalSliderSeparationThreshold.setMinimum(minVal-1)
        self.spinBoxSeparationThreshold.setMaximum(maxVal+1)
        self.horizontalSliderSeparationThreshold.setMaximum(maxVal+1)
        self.spinBoxSeparationThreshold.setValue(maxVal+1)
        self.horizontalSliderSeparationThreshold.setValue(maxVal+1)
        self.separationThreshold=maxVal+1
        
    def enable_probability_related_tools(self):
        self.enable_probability_related_tools=True
        self.toolButtonCostPnt.setDisabled(False)
        self.toolButtonPolyFit.setDisabled(False)
        
    def disable_probability_related_tools(self):
        self.enable_probability_related_tools=False
        self.toolButtonCostPnt.setDisabled(True)
        self.toolButtonPolyFit.setDisabled(True)
