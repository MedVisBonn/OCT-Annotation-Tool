"""
Created in 2018

@author: Shekoufeh Gorgi Zadeh
"""

import os, sys, inspect

controllerPath=os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))[:-4]+"controller"
if controllerPath not in sys.path:
     sys.path.insert(0, controllerPath)

import matplotlib
import numpy as np
import image_editor as se
import matplotlib.cm as cm
import qimage2ndarray as q2np
from PyQt4 import QtCore, QtGui
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


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
# Label class, used to draw the uncertainty tables. Then used under the viewer.
#==============================================================================
class MyQLabel(QtGui.QLabel):
    
    def __init__(self,layout,parentWindow,name):
        QtGui.QLabel.__init__(self,layout) 
        self.parentWindow=parentWindow
        
        self.colors=list() # Bimodal Entropy and Probability 
        self.colorsEntropy=list()
        self.colorsProbability=list()
        
        self.colorMapToUse='entropyAndProbability'
        
        res=np.ones((10,145*5,3))
        res.fill(255)
        qimg=qimg=q2np.array2qimage(res)
        image=QtGui.QPixmap.fromImage(qimg)
        
        self.setPixmap(image)
        self.image=None
        
        self.mul=1
        self.size=10
        self.numOfEls=(51/self.mul)+(self.mul-1)
        
        self.lastSelected=0
        self.max=0
        
        colors1 = [(254/255.,232/255.,200/255.),(253/255.,187/255.,132/255.),\
            (227/255.,74/255.,51/255.)]  
        colors2 = [(236/255.,231/255.,242/255.),(166/255.,189/255.,219/255.),\
            (43/255.,140/255.,190/255.)]  
        cmap_name1 = 'oranges'
        cmap_name2 = 'blues'
        self.cm1 = LinearSegmentedColormap.from_list(cmap_name1, colors1, N=200)
        self.cm2 = LinearSegmentedColormap.from_list(cmap_name2, colors2, N=200)
        self.name=name
        self.activeMap=None
    
    def set_max(self,m):
        self.max=m
    
    def set_map_name(self,mapName):
        self.colorMapToUse=mapName
    
    def paint(self, painter, option, widget=None):
        painter.drawPixmap(0,0, self.pixmap())          
        
    def set_color_to_perfect_in_slice(self,sliceNum,perfectNum):
        if(self.parentWindow.entropyVals is None):
            self.colors[sliceNum-1]=self.map_number_to_color(1.)
        else:
            self.colors[sliceNum-1]=\
                self.map_number_to_bimodal_color_rgb_diagonal_plane(perfectNum,perfectNum)
            self.colorsEntropy[sliceNum-1]=self.map_number_to_color2(perfectNum)
            self.colorsProbability[sliceNum-1]=self.map_number_to_color(perfectNum)
            
        self.select(sliceNum)
    
    def transform_point_to_bimodal_coordinate(self,x,y):
        eps=1e-10
        luminance=(np.sqrt(x**2+y**2)/np.sqrt(2.))
        saturation=1.- (1./(1.+(y/(x+eps))))
        return luminance, saturation
        
    def transform_point_to_bimodal_coordinate_rgb_diagonal_plane(self,x,y):
        return 0.5*x+0.5*y
        
    def map_number_to_color(self,number):
        minima = 0.
        maxima = 1.
        
        norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=self.cm2)
        
        return mapper.to_rgba(number)
    
    def map_number_to_color2(self,number):
        minima = 0.
        maxima = 1.
        norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=self.cm1)
        return mapper.to_rgba(number)
    
    def map_number_to_bimodal_color(self,n1,n2):
        minima = 0.
        maxima = 1.
        
        l,s=self.transform_point_to_bimodal_coordinate(n1,n2)
        
        norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.Greys)

        c1=mapper.to_rgba(n1)
        
        norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.PRGn)
        
        c2=mapper.to_rgba(n2)
        return ((c1[0]+c2[0])/2.,(c1[1]+c2[1])/2.,(c1[2]+c2[2])/2.,(c1[3]+c2[3])/2.)
        
    def map_number_to_bimodal_color_rgb_diagonal_plane(self,n1,n2):
        n3=self.transform_point_to_bimodal_coordinate_rgb_diagonal_plane(n1,n2)
        return (n1,n3,n2,1.)
        
    def map_numbers_to_bimodal_colors(self,numbers1,numbers2):
        colors=list()
        for i in range(len(numbers1)):
            colors.append(self.map_number_to_bimodal_color(numbers1[i],numbers2[i]))
        return colors

    def map_numbers_to_bimodal_colors_rgb_diagonal_plane(self,numbers1,numbers2):
        colors=list()
        for i in range(len(numbers1)):
            colors.append(self.map_number_to_bimodal_color_rgb_diagonal_plane(\
                numbers1[i],numbers2[i]))
        return colors

    def map_numbers_to_colors(self,numbers):
        minima = 0.
        maxima = 1.
        norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=self.cm2)
        colors=list()
        for v in numbers:
            colors.append(mapper.to_rgba(v))
        return colors
    
    def map_numbers_to_colors2(self,numbers):
        minima = 0.
        maxima = 1.
        norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=self.cm1)
        colors=list()
        for v in numbers:
            colors.append(mapper.to_rgba(v))
        return colors

    def show_color_map(self, block = True ):
        image=np.zeros((100,100,3))
        for i in range(100):
            for j in range(100):
                c=self.map_number_to_bimodal_color_rgb_diagonal_plane(i*0.01,j*0.01)
                image[i,j,0]=c[0]
                image[i,j,1]=c[1]
                image[i,j,2]=c[2]
                
        plt.imshow( image)
        plt.show(block)
        QtGui.QApplication.processEvents()

    def show_image(self, image,block = True ):
        plt.imshow( image)
        plt.show(block)
        QtGui.QApplication.processEvents()
    
    def set_active_map(self,mapName):
        self.activeMap=mapName
    
    def createImage(self):
        
        if(len(self.colors)<20):
            self.mul=3
            hh=30
        else:
            self.mul=1
            hh=30
        mul=self.mul
        self.size=10*mul
        size=self.size
        self.numOfEls=len(self.colorsEntropy)
        numOfEls=self.numOfEls
        width=int(numOfEls*size)
        height=hh
        if(self.image is None):
            self.image=np.empty((height,width,4))
            self.image.fill(0.)
        i=0
        j=0
        counter=0
        colors=list()
        if(self.colorMapToUse=='entropyAndProbability'):
            colors=self.colors
        elif(self.colorMapToUse=='entropy'):
            colors=self.colorsEntropy
        elif(self.colorMapToUse=='probability'):
            colors=self.colorsProbability
        
        for c in colors:
            tmp=np.repeat(c,size*height).reshape(4,size*height).T
            tmp=tmp.reshape((4,height,size))
            tmp=np.ones((height,size,4))
            tmp[:,:,0]=c[0]
            tmp[:,:,1]=c[1]
            tmp[:,:,2]=c[2]
            tmp[:,:,3]=c[3]
            self.image[(j*size):,i*size:i*size+size,:]=tmp
            counter=counter+1
            
            j=counter/numOfEls
            i=counter%numOfEls
        self.image=(self.image*255).astype(int)
        self.image=self.image*1.
        self.image[np.where(self.image>255)]=255
        qimg=qimg=q2np.array2qimage(self.image)
        image=QtGui.QPixmap.fromImage(qimg)
        self.setPixmap(image)
        self.update()
       
    def updateImage(self):
        if(not self.parentWindow.uncertaintyValues is None and\
                len(self.parentWindow.uncertaintyValues)>0):
            if(not self.parentWindow.entropyVals is None and\
                    len(self.parentWindow.entropyVals)>0):
                self.colors=\
                    self.map_numbers_to_bimodal_colors_rgb_diagonal_plane(\
                        self.parentWindow.entropyVals,self.parentWindow.probVals)
                self.colorsEntropy=\
                    self.map_numbers_to_colors2(self.parentWindow.entropyVals)
                self.colorsProbability=\
                    self.map_numbers_to_colors(self.parentWindow.probVals)
            else:
                self.colors=\
                    self.map_numbers_to_colors(self.parentWindow.uncertaintyValues)
            self.createImage()
        else:
            res=np.ones((10,145*5,3))
            res.fill(255)
            qimg=qimg=q2np.array2qimage(res)
            image=QtGui.QPixmap.fromImage(qimg)
            self.image=image
            self.setPixmap(self.image)
        self.update()
    
    def select(self,num):
        if(self.parentWindow.uncertaintyValues is None and\
                len(self.parentWindow.uncertaintyValues)>0):
            return
        hh=self.image.shape[0]
        # Refill last selected item
        col=self.lastSelected%self.numOfEls
        
        colIndex=self.size*col
        
        if(self.colorMapToUse=='entropyAndProbability'):
            c=self.colors[self.lastSelected]
        elif(self.colorMapToUse=='entropy'):
            c=self.colorsEntropy[self.lastSelected]
        elif(self.colorMapToUse=='probability'):
            c=self.colorsProbability[self.lastSelected]
            
        self.image[:,colIndex:colIndex+self.size,0]=int(c[0]*255)
        self.image[:,colIndex:colIndex+self.size,1]=int(c[1]*255)
        self.image[:,colIndex:colIndex+self.size,2]=int(c[2]*255)
        self.image[:,colIndex:colIndex+self.size,3]=int(c[3]*255)
        
        # Draw border around newly selected
        index=num-1
        col=index%self.numOfEls
        colIndex=self.size*col
        for i in range(self.size):
            self.image[hh-1,colIndex+i,:]=[0.,0.,0.,255.]
        if(self.activeMap==self.name):
            for i in range(self.size):
                self.image[hh-2,colIndex+i,:]=[0.,0.,0.,255.]
                self.image[hh-3,colIndex+i,:]=[0.,0.,0.,255.]
        qimg=qimg=q2np.array2qimage(self.image)
        image=QtGui.QPixmap.fromImage(qimg)
        self.setPixmap(image)
        self.lastSelected=index
        self.update()
        
    def mousePressEvent (self, event):
        x=event.pos().x()
        col=x/self.size
        self.activeMap=self.name
        index=col
        if(index!=self.lastSelected):
            self.select(index+1)
            self.parentWindow.set_spinbox_value(index+1)
        self.parentWindow.set_current_colormap(self.name)
        self.update()
        
    def wheelEvent(self, event):
        self.activeMap=self.name
        if event.delta() > 0:
            newIndex=min(self.lastSelected+1,self.max-1)
        else:
            newIndex=max(self.lastSelected-1,0)
        self.select(newIndex+1)
        self.parentWindow.set_spinbox_value(newIndex+1)
        
class MyQSpinBox(QtGui.QSpinBox):
    
    def __init__(self,layout,parentWindow):
       QtGui.QSpinBox.__init__(self,layout) 
       self.parentWindow=parentWindow
    
    def keyPressEvent(self,event):
        if(event.key()==QtCore.Qt.Key_Space):
            self.parentWindow.toggle_check_box()
        elif(event.key()==QtCore.Qt.Key_B):
            self.parentWindow.graphicsViewImageViewer.toggle_annotation_view()
        else:
            return QtGui.QSpinBox.keyPressEvent(self, event)

#==============================================================================
# Image editor viewer that contains graphicsview class.            
#==============================================================================
class Ui_imageEditor(object):
    
    def __init__(self,controller=None,name=""):
        self.controller=controller
        self.name=name
    def setupUi(self, imageEditor):
        imageEditor.setObjectName(_fromUtf8("imageEditor"))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join("icons",\
            "633563-basic-icons","png","233-empty.png"))), QtGui.QIcon.Normal,\
            QtGui.QIcon.Off)
        imageEditor.setWindowIcon(icon)
        self.imageEditor=imageEditor
        self.gridLayout = QtGui.QGridLayout(imageEditor)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        
        self.graphicsViewImageViewer=se.ImageEditor(self.controller,self.name)
        self.graphicsViewImageViewer.setObjectName(\
            _fromUtf8("graphicsViewImageViewer"))
        
        self.gridLayout.addWidget(self.graphicsViewImageViewer, 0, 0, 1, 1)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.spinBoxSliceNum = MyQSpinBox(imageEditor,self)
        self.spinBoxSliceNum.setMinimum(1)
        self.spinBoxSliceNum.setMaximum(1000)
        sizePolicy=QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding,\
            QtGui.QSizePolicy.Fixed)
        self.spinBoxSliceNum.setSizePolicy(sizePolicy)
        self.spinBoxSliceNum.setObjectName(_fromUtf8("spinBoxSliceNum"))
        
        self.spinBoxSliceNum.valueChanged.connect(self.value_change)        
        self.currentSliceNum=self.spinBoxSliceNum.value()
        self.horizontalLayout.addWidget(self.spinBoxSliceNum)
        self.uncertaintyValues=None
        self.entropyVals=None
        self.probVals=None
        self.currentMap='entropyAndProbability'
        self.gridLayout.addLayout(self.horizontalLayout, 1, 0, 1, 1)
        if(self.name=='hrfViewer'):
            self.checkBoxHRF=QtGui.QCheckBox(imageEditor)
            self.checkBoxHRF.setObjectName(_fromUtf8("checkBoxHRF"))
            self.horizontalLayout.addWidget(self.checkBoxHRF)
            self.allHRFStatus=np.empty(1000,dtype=bool)
            self.allHRFStatus.fill(False)
            self.checkBoxHRF.setChecked(self.allHRFStatus[self.currentSliceNum-1])
            self.checkBoxHRF.clicked.connect(self.check_box_clicked)
           
        if(self.name=='layerViewer'):
            self.uncertaintyMapIsVisible=False 
            self.horizontalLayout3 = QtGui.QHBoxLayout()
            self.horizontalLayout3.setObjectName(_fromUtf8("horizontalLayout"))
            self.verticalLayout_UpUe = QtGui.QVBoxLayout()
            self.verticalLayout_UpUe.setObjectName(_fromUtf8("verticalLayout_UpUe"))
            self.label_Up = QtGui.QLabel(imageEditor)
            self.label_Up.setObjectName(_fromUtf8("label_Up"))
            self.verticalLayout_UpUe.addWidget(self.label_Up)
            self.label_Ep = QtGui.QLabel(imageEditor)
            self.label_Ep.setObjectName(_fromUtf8("label_Ep"))
            self.verticalLayout_UpUe.addWidget(self.label_Ep)
            self.horizontalLayout3.addLayout(self.verticalLayout_UpUe)
            self.verticalLayout_labelImages = QtGui.QVBoxLayout()
            self.verticalLayout_labelImages.setObjectName(\
                _fromUtf8("verticalLayout_labelImages"))
            self.labelImage = MyQLabel(imageEditor,self,'probability')
            sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding,\
                QtGui.QSizePolicy.Fixed)
            sizePolicy.setHorizontalStretch(0)
            sizePolicy.setVerticalStretch(0)
            sizePolicy.setHeightForWidth(self.labelImage.sizePolicy().\
                hasHeightForWidth())
            self.labelImage.setSizePolicy(sizePolicy)
            self.labelImage.setObjectName(_fromUtf8("labelImage"))
            self.verticalLayout_labelImages.addWidget(self.labelImage)
            self.labelImage2 = MyQLabel(imageEditor,self,'entropy')
            sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding,\
                QtGui.QSizePolicy.Preferred)
            sizePolicy.setHorizontalStretch(0)
            sizePolicy.setVerticalStretch(0)
            sizePolicy.setHeightForWidth(self.labelImage2.sizePolicy().\
                hasHeightForWidth())
            self.labelImage2.setSizePolicy(sizePolicy)
            self.labelImage2.setObjectName(_fromUtf8("labelImage2"))
            self.verticalLayout_labelImages.addWidget(self.labelImage2)
            self.horizontalLayout3.addLayout(self.verticalLayout_labelImages)

            self.gridLayout.addLayout(self.horizontalLayout3,2,0,1,1)
            self.labelImage.hide()
            self.label_Up.hide()
            self.labelImage2.hide()
            self.label_Ep.hide()
            
        self.retranslateUi(imageEditor)
        QtCore.QMetaObject.connectSlotsByName(imageEditor)
        
    def set_pen(self):
        self.graphicsViewImageViewer.set_pen()
        
    def set_line(self):
        self.graphicsViewImageViewer.set_line()
        
    def set_fill(self):
        self.graphicsViewImageViewer.set_fill()
    
    def set_draw_dru(self):
        self.graphicsViewImageViewer.set_draw_dru()
    
    def set_morphology(self,itLevel):
        self.graphicsViewImageViewer.set_morphology(itLevel)
        
    def set_filter_dru(self,filteringHeight,maxFilteringHeight):
        self.graphicsViewImageViewer.set_filter_dru(filteringHeight,\
            maxFilteringHeight)
       
    def set_grab(self):
        self.graphicsViewImageViewer.set_grab()
        
    def all_threshold_value_changed(self,value):
        self.graphicsViewImageViewer.all_threshold_value_changed(value)        
                
    def morphology_value_changed(self,value):
        self.graphicsViewImageViewer.morphology_value_changed(value)
                 
    def poly_degree_value_changed(self,value):
        self.graphicsViewImageViewer.poly_degree_value_changed(value)
    
    def smoothness_value_changed(self,value):
        self.graphicsViewImageViewer.smoothness_value_changed(value)
    def max_threshold_value_changed(self,value):
        self.graphicsViewImageViewer.max_threshold_value_changed(value)
        
    def grab_value_changed(self,position):
        self.graphicsViewImageViewer.grab_value_changed(position)
        
    def wheelEvent(self, event):
        self.graphicsViewImageViewer.wheelEvent(event)
        
    def retranslateUi(self, imageEditor):
        imageEditor.setWindowTitle(_translate("imageEditor", "View", None))
        if(self.name=='hrfViewer'):
            self.checkBoxHRF.setText(_translate("imageEditor", "HRF Exists", None))
        if(self.name=='layerViewer'):
            self.label_Up.setText(_translate("Form", "Uncertainy P:", None))
            self.label_Ep.setText(_translate("Form", "Uncertainy E:", None))
            self.labelImage.setText(_translate("Form", "TextLabel", None))
            self.labelImage2.setText(_translate("Form", "TextLabel", None))

    def check_box_clicked(self):
        if(self.name=='hrfViewer'):
            self.allHRFStatus[self.currentSliceNum-1]=self.checkBoxHRF.isChecked()
            self.controller.update_HRF_status(self.currentSliceNum-1,\
                self.checkBoxHRF.isChecked())
            
    def toggle_check_box(self):
        if(self.name=='hrfViewer'):
            self.checkBoxHRF.toggle()
            self.allHRFStatus[self.currentSliceNum-1]=self.checkBoxHRF.isChecked()
            self.controller.update_HRF_status(self.currentSliceNum-1,\
                self.checkBoxHRF.isChecked())
            
    def set_check_box(self,status):
        if(self.name=='hrfViewer'):
            self.checkBoxHRF.setChecked(status)
            self.allHRFStatus[self.currentSliceNum-1]=self.checkBoxHRF.isChecked()
            
    def value_change(self):
        
        self.currentSliceNum=self.spinBoxSliceNum.value()
        self.controller.slice_value_changed(self.spinBoxSliceNum.value(),\
            self.name,furtherUpdate=False)
        if(self.name=='hrfViewer'):
            self.checkBoxHRF.setChecked(self.allHRFStatus[self.currentSliceNum-1])
        if(self.name=='layerViewer' and self.uncertaintyMapIsVisible):
            self.labelImage.select(self.currentSliceNum)
            self.labelImage2.select(self.currentSliceNum)
    def set_spinbox_value(self,value):
        self.spinBoxSliceNum.setValue(value)
        
    def set_all_HRF_status(self,hrfStatus):
        self.allHRFStatus=np.copy(hrfStatus)

    def get_all_HRF_status(self):
        return self.allHRFStatus
        
    def set_max_possible_value(self,maxValue):
        self.spinBoxSliceNum.setMaximum(maxValue)
        if(self.name=='layerViewer'):
            self.labelImage.set_max(maxValue)
            self.labelImage2.set_max(maxValue)
        
    def closeEvent(self,event):
        event.accept()
        
    def set_current_colormap(self,mapName):
        self.graphicsViewImageViewer.set_uncertainty_type(mapName)
        self.labelImage.set_active_map(mapName)
        self.labelImage2.set_active_map(mapName)
        self.labelImage.update()
        self.labelImage2.update()
    
    def set_uncertainty_value(self,uncertainties,sliceNumZ):
        if(uncertainties[0] is None):
            return
        if(not self.uncertaintyValues is None and len(self.uncertaintyValues)>0):
            self.uncertaintyValues[sliceNumZ]=uncertainties[0]
            self.entropyVals[sliceNumZ]=uncertainties[1]
            self.probVals[sliceNumZ]=uncertainties[2]
            if(self.uncertaintyMapIsVisible and not self.uncertaintyValues is\
                    None and len(self.uncertaintyValues)>0):
                self.labelImage.updateImage()
                self.labelImage2.updateImage()
                self.labelImage.select(sliceNumZ+1)
                self.labelImage2.select(sliceNumZ+1)
    
    def set_uncertaintyValues(self,uncertaintyValues,entropyVals,probVals):
        self.uncertaintyValues=uncertaintyValues
        self.entropyVals=entropyVals
        self.probVals=probVals
        if(self.uncertaintyMapIsVisible and not self.uncertaintyValues is\
                None and len(self.uncertaintyValues)>0):
            self.labelImage.updateImage()
            self.labelImage.select(self.currentSliceNum)
            
            self.labelImage2.updateImage()
            self.labelImage2.select(self.currentSliceNum)

    def slice_edited(self,sliceNum,layerName):
        if(self.name=='layerViewer' and self.uncertaintyMapIsVisible):
            perfectNum=0.05
            self.uncertaintyValues[sliceNum-1]=perfectNum
            self.entropyVals[sliceNum-1]=perfectNum
            self.probVals[sliceNum-1]=perfectNum
            self.labelImage.set_color_to_perfect_in_slice(sliceNum,perfectNum)
            self.labelImage2.set_color_to_perfect_in_slice(sliceNum,perfectNum)
            self.graphicsViewImageViewer.slice_edited(sliceNum,layerName)

    def triggerUncertaintyMap(self,mapNameP,mapNameE):
        if(self.name=='layerViewer'):
            self.controller.compute_probability_maps()
            if(self.uncertaintyMapIsVisible):
                self.labelImage.hide()
                self.labelImage2.hide()
                self.label_Ep.hide()
                self.label_Up.hide()
                self.uncertaintyMapIsVisible=False
                self.graphicsViewImageViewer.show_uncertainties(False)
            else:
                self.labelImage.show()
                self.labelImage2.show()
                self.label_Ep.show()
                self.label_Up.show()
                self.uncertaintyMapIsVisible=True
                self.graphicsViewImageViewer.show_uncertainties(True)
            self.graphicsViewImageViewer.set_uncertainty_type(mapNameP)
            self.currentMap=mapNameP
            self.labelImage.set_map_name(mapNameP)
            self.labelImage.updateImage()
            self.labelImage2.set_map_name(mapNameE)
            self.labelImage2.updateImage()
            
    def get_uncertainties(self):
        return self.uncertaintyValues,self.entropyVals,self.probVals

