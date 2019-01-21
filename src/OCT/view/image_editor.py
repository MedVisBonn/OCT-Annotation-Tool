"""
Created in 2018

@author: Shekoufeh Gorgi Zadeh
"""

import math
import matplotlib
import scipy as sc
import numpy as np
import matplotlib.cm as cm
import qimage2ndarray as q2np
from PyQt4 import QtCore,QtGui
from bresenham import bresenham
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


#==============================================================================
# Viewer's graphics part redefined to handle different mouse cursor
#==============================================================================
class MyGraphicsView(QtGui.QGraphicsView):
    
    def __init__(self,parent=None):
        QtGui.QGraphicsView.__init__(self)
        self.setTransformationAnchor(QtGui.QGraphicsView.AnchorUnderMouse)
        self.parentWindow=parent
        
    def wheelEvent(self,event):        
        adj = (event.delta()/120) * 0.1
        self.scale(1+adj,1+adj)
        
    def fit(self):
        self.scale(1.1,1.1)
        self.scale(0.9,0.9)
    
    def keyPressEvent(self,event):
        if(event.key()==QtCore.Qt.Key_Control):
            status=self.parentWindow.get_cursor_status()
            if(status=='showHand'):
                viewport=self.viewport()
                viewport.setCursor(QtCore.Qt.PointingHandCursor)
        else:
            return QtGui.QGraphicsView.keyPressEvent(self, event)
            
    def keyReleaseEvent(self,event):
        if(event.key()==QtCore.Qt.Key_Control):
            status=self.parentWindow.get_cursor_status()
            if(status=='showHand'):
                viewport=self.viewport()
                viewport.setCursor(QtCore.Qt.ArrowCursor)
        else:
            return QtGui.QGraphicsView.keyPressEvent(self, event)    
 
#==============================================================================
# Major class that draws annotations on screen and takes in user actions       
#==============================================================================
class ImageDrawPanel(QtGui.QGraphicsPixmapItem):
    
    def __init__(self, pixmap=None, parent=None, scene=None, controller=None,\
            editorType=''):
        super(ImageDrawPanel, self).__init__()
        
        self.showUncertainties=False
        self.entropyValsPerBcan=None
        self.probabilityValsPerBcan=None
        self.entropyColor=None
        self.probabilityColor=None
        self.uncertaintyType='None'
        
        self.editedLayers=list()
        self.redrawLayers=False
        
        self.showSplineKnots=False
        self.hasPickedKnot=False
        self.pickedKnot=None
        
        self.controller=controller
        self.x, self.y = -1, -1      
        self.prevX,self.prevY=self.x,self.y
        
        self.etype=editorType
        self.mainPhoto=np.empty((10,10))
        self.overlayedPhotos=[self.mainPhoto]
        self.suggestionLayerImg=None
        
        self.coeffs=[1.0]        
        self.radius = 10
        self.width=10
        self.height=10
        
        self.showAnnotation=True
        
        self.sliceline=(0,0,0,0)
        self.sliceNum=1
        self.combine_images()
        self.drawPen=False
        self.selectRect=False 
        self.morphology=False
        self.drawDru=False
        self.filterHeight=False
        self.filteringHeight=1
        self.maxFilteringHeight=2
        
        self.rect=QtCore.QRectF()
        self.rectFixPoint=QtCore.QPointF()
        self.itLevel=1
        self.polyDegreeValue=2
        self.smoothness=2
        self.selectLine=False
        self.line=QtCore.QLineF()
        
        self.color=QtGui.QColor(255,255,255)
        
        self.fillArea=False
        self.drawSliceLine=True
        
        self.updateGrab=False
        self.grabPosition=0
        
        self.bboxList=dict()
        self.drawBBox=False
        self.bBoxCount=0
        
        self.ccaMask=None
        
        self.markCostPnt=False
        self.fitPolynomial=False
        
        self.drusenPeaks=set()
        self.markDrusenPeak=False
        
        self.showSeparators=False
        self.separators=None
        self.labels=None
        self.separatorAvgHeight=None
        self.separatorHeightThreshold=100000
        
        self.mergeDrusenMode=False
        self.sourceLabel=-1
     
        self.clickType=1 # 1: left click, 2: right click
        colors1 = [(255/255.,0/255.,0/255.),(227/255.,128/255.,0/255.),\
                    (212/255.,162/255.,40/255.),(219/255.,173/255.,110/255.),\
                    (209/255.,198/255.,179/255.)]
        colors2 = [(0/255.,72/255.,255/255.),(30/255.,113/255.,250/255.),\
                    (116/255.,182/255.,207/255.),(183/255.,208/255.,225/255.),\
                    (207/255.,221/255.,231/255.)]
                    
        cmap_name1 = 'oranges'
        cmap_name2 = 'blues'
        self.cm1 = LinearSegmentedColormap.from_list(cmap_name1, colors1, N=200)
        self.cm2 = LinearSegmentedColormap.from_list(cmap_name2, colors2, N=200)
        
        
    def unset_all(self):
        self.selectRect=False 
        self.morphology=False
        self.drawDru=False
        self.filterHeight=False
        self.selectLine=False
        self.fillArea=False
        self.drawPen=False
        self.updateGrab=False
        self.drawBBox=False
        self.markCostPnt=False
        self.fitPolynomial=False
        self.markDrusenPeak=False
        self.showSeparators=False
        self.mergeDrusenMode=False
        
    def set_uncertainty_type(self,utype):
        self.uncertaintyType=utype
        self.update()
        
    def show_uncertainties(self,status):
        self.showUncertainties=status
        self.redrawLayers=(not status)
        self.update()
        
    def set_cca_mask(self,mask):
        self.ccaMask=mask
        
    def unset_cca_mask(self):
        self.ccaMask=None
        self.combine_images()
        self.update()
        
    def set_pen(self):
        self.unset_all()
        self.drawPen=True
       
    def set_line2(self):
        self.unset_all()
        self.selectLine=True
        
    def set_fill(self):
        self.unset_all()
        self.fillArea=True
    
    def set_draw_dru(self):
        self.unset_all()
        self.selectRect=True
        self.drawDru=True
    
    def set_morphology(self,itLevel):
        self.unset_all()
        self.itLevel=itLevel
        self.selectRect=True
        self.morphology=True
    
    def set_filter_dru(self,filteringHeight,maxFilteringHeight):
        self.unset_all()
        self.filteringHeight=filteringHeight
        self.maxFilteringHeight=maxFilteringHeight
        self.selectRect=True
        self.filterHeight=True
        
    def set_grab(self):
        self.unset_all()
        self.updateGrab=True
    
    def set_bounding_box(self):
        self.unset_all()
        self.selectRect=True
        self.drawBBox=True
    
    def set_cost_point(self):
        self.unset_all()
        self.markCostPnt=True  
    
    def set_poly_fit(self,value):
        self.unset_all()
        self.selectRect=True
        self.fitPolynomial=True 
        self.polyDegreeValue=value
        
    def set_HRF_BBox(self,hrfBBox):
        self.bboxList=hrfBBox
        self.bBoxCount=0
        for s in self.bboxList.keys():
            self.bBoxCount+=len(self.bboxList[s])
     
    def set_nga_BBox(self,ngaBBox):
        self.bboxList=ngaBBox
        self.bBoxCount=0
        for s in self.bboxList.keys():
            self.bBoxCount+=len(self.bboxList[s])
            
    def set_enface_BBox(self,hrfBBox):
        self.bboxList=hrfBBox
        self.bBoxCount=0
        for s in self.bboxList.keys():
            self.bBoxCount+=len(self.bboxList[s])
            
    def set_manual_marker_selection(self):
        self.unset_all()
        self.markDrusenPeak=True
    
    def set_merge_drusen(self):
        self.mergeDrusenMode=True
        
    def unset_merge_drusen(self):
        self.mergeDrusenMode=False
        
    def set_separation_threshold(self,value):
        self.separatorHeightThreshold=value
        self.combine_images()

    def curve_to_spline(self):
        self.unset_all()
        self.showSplineKnots=True
        self.selectRect=True
        self.combine_images()
    
    def spline_to_curve(self):
        self.unset_all()
        self.showSplineKnots=False
        self.combine_images()        
    
    def get_manual_markers(self):
        return self.drusenPeaks
        
    def update_slice_number(self,sliceValue):
        self.sliceNum=sliceValue
        
    def map_numbers_to_colors(self,numbers,colorMap):
        minima = np.min(numbers)
        maxima = np.max(numbers)
        norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=colorMap)
        
        clrs=list()
        for v in numbers:
            c=mapper.to_rgba(v)
            c=(c[0]*255,c[1]*255,c[2]*255,c[3]*255)
            clrs.append(c)
        return clrs
        
    def show_drusen_splitting_separators(self,separators,separatorsAvgHeight,\
                labels):    
        self.showSeparators=True
        self.separators=separators
        self.labels=labels
        self.separatorAvgHeight=separatorsAvgHeight
        self.combine_images()
        
    def all_threshold_value_changed(self,value):
        self.filteringHeight=value   
                
    def morphology_value_changed(self,value):
        self.itLevel=value
        
    def poly_degree_value_changed(self,value):
        self.polyDegreeValue=value
        
    def smoothness_value_changed(self,value):
        self.smoothness=value
        
    def get_smoothness(self):
        return self.smoothness
        
    def max_threshold_value_changed(self,value):
        self.maxFilteringHeight=value
    
    def grab_value_changed(self,position):
        self.updateGrab=True
        self.grabPosition=position
        self.update()
        
    def set_edited_layers(self,editedLayers):
        self.editedLayers=editedLayers
    
    def get_cursor_status(self):
        if(self.mergeDrusenMode and self.etype=='enfaceDrusenViewer'):
            return 'showHand'
        return 'arrowCursor'
    def get_drusen_separators(self):
        if(self.separators is None):
            return None
            
        separators=np.copy(self.separators)
        if(not self.separatorAvgHeight is None and len( np.where(\
                self.separatorAvgHeight>self.separatorHeightThreshold)[0])>0):
            separators[np.where(self.separatorAvgHeight>\
                self.separatorHeightThreshold)]=0.
        return separators
        
    def set_drusen_separators(self,separators):        
        self.separators=separators
        
    def done_splitting(self):
        del self.labels
        del self.separators
        del self.drusenPeaks
        del self.separatorAvgHeight
        
        self.labels=None
        self.separators=None
        self.separatorAvgHeight=None
        self.drusenPeaks=set()
        self.markDrusenPeak=False
        self.showSeparators=False
        self.mergeDrusenMode=False
        self.sourceLabel=-1
        self.separatorHeightThreshold=100000
        self.combine_images()
        
    def box_exists(self):
        if(not self.sliceNum-1 in self.bboxList.keys()):
            return False
        else:
            if(len(self.bboxList[self.sliceNum-1])>0):
                return True
            else:
                return False
                
    def delete_box(self,rect,sliceNum):
        if(sliceNum-1 in self.bboxList.keys()):
            bboxes=self.bboxList[sliceNum-1]
            i=0
            for box in bboxes:
                if(box==rect):
                    self.bboxList[sliceNum-1].pop(i)
                    self.bBoxCount-=1
                i=i+1  
        if(self.etype=='hrfViewer'):
            self.controller.update_HRF_status(self.sliceNum-1,self.box_exists(),True)
        elif(self.etype=='enfaceViewer'):
            self.controller.update_enface_status()
        elif(self.etype=='gaViewer'):
            self.controller.update_ga_status()
        self.update()
        
    def add_box(self,rect,sliceNum):
        if(sliceNum-1 in self.bboxList.keys()):
            bboxes=self.bboxList[sliceNum-1]
            i=0
            found=False
            for box in bboxes:
                if(box==rect):
                    found=True
                    break
                i=i+1     
            if(not found):
                self.bboxList[sliceNum-1].append(rect)
                self.bBoxCount+=1
        else:
            self.bboxList[sliceNum-1]=[rect]
            self.bBoxCount+=1
        if(self.etype=='hrfViewer'):
            self.controller.update_HRF_status(self.sliceNum-1,self.box_exists(),True)
        elif(self.etype=='enfaceViewer'):
            self.controller.update_enface_status()
        elif(self.etype=='gaViewer'):
            self.controller.update_ga_status()
        self.update()
        
    def delete_boxes(self,rects,sliceNum):
        if(sliceNum-1 in self.bboxList.keys()):
            bboxes=self.bboxList[sliceNum-1]
            for rect in rects:
                i=0
                for box in bboxes:
                    if(box==rect):
                        self.bboxList[sliceNum-1].pop(i)
                        self.bBoxCount-=1
                    i=i+1  
        if(self.etype=='hrfViewer'):
            self.controller.update_HRF_status(self.sliceNum-1,self.box_exists(),True)
        elif(self.etype=='gaViewer'):
            self.controller.update_ga_status()
        elif(self.etype=='enfaceViewer'):
            self.controller.update_enface_status()
        self.update()
        
    def add_boxes(self,rects,sliceNum):
        if(sliceNum-1 in self.bboxList.keys()):
            bboxes=self.bboxList[sliceNum-1]
            for rect in rects:
                i=0
                found=False
                for box in bboxes:
                    if(box==rect):
                        found=True
                        break
                    i=i+1     
                if(not found):
                    self.bboxList[sliceNum-1].append(rect)
                    self.bBoxCount+=1
        else:
            self.bboxList[sliceNum-1]=list()
            for rect in rects:
                self.bboxList[sliceNum-1].append(rect)
                self.bBoxCount+=1
        if(self.etype=='hrfViewer'):
            self.controller.update_HRF_status(self.sliceNum-1,self.box_exists(),True)
        elif(self.etype=='gaViewer'):
            self.controller.update_ga_status(self.sliceNum-1,self.box_exists(),True)
        elif(self.etype=='enfaceViewer'):
            self.controller.update_enface_status()
        self.update()
    
    def toggle_annotation_view(self):
        if(self.showAnnotation):
            self.showAnnotation=False
        else:
            self.showAnnotation=True
        self.combine_images()
        self.update()
        
    def paint(self, painter, option, widget=None):
        self.controller.image_changed(self.mainPhoto,self.etype)
        self.setOffset(QtCore.QPointF(0.5,0.5))
        painter.drawPixmap(0,0, self.pixmap())
        
        if(self.selectRect):
            self.pen = QtGui.QPen(QtCore.Qt.DashLine)
            self.pen.setColor(QtGui.QColor(127,127,127))
            
            self.pen.setWidth(0.5)
            painter.setPen(self.pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawRect(self.rect)
            
        if(self.selectLine):
            self.pen = QtGui.QPen(QtCore.Qt.DashLine)
            self.pen.setColor(QtGui.QColor(127,127,127))
            self.pen.setWidth(0.5)
            painter.setPen(self.pen)
            painter.drawLine(self.line)

        lineOffset=0.5  
        if(self.etype=='enfaceViewer'):
            for sliceNum in self.bboxList.keys():
                rects=self.bboxList[sliceNum]
                self.pen = QtGui.QPen(QtCore.Qt.SolidLine)
                self.pen.setColor(QtGui.QColor(0,255,255,170))
                self.pen.setWidth(0.5)
                painter.setPen(self.pen)
                for rect in rects:
                    painter.drawRect(rect)
            
            unc=self.controller.get_enface_uncertainty_overlay()
            if(not unc is None):
                layer=unc[0]
                for i in range(len(self.editedLayers)):
                    edited=self.editedLayers[i][layer]
                    if(edited):
                        self.pen = QtGui.QPen(QtCore.Qt.SolidLine)
                        self.pen.setColor(QtGui.QColor(255,255,0,200))
                        self.pen.setWidth(1.0)
                        painter.setPen(self.pen)
                        width=self.width
                        painter.drawLine(QtCore.QLineF(float(width-10),\
                            float(i)+0.5,float(width-1),float(i)+0.5))
        if(self.drawSliceLine and self.etype!='drusenViewer'):
            self.pen = QtGui.QPen(QtCore.Qt.SolidLine)
            self.pen.setColor(QtGui.QColor(0,255,255,170))
            self.pen.setWidth(1.0)
            painter.setPen(self.pen)
            sliceLine=self.sliceline
            if(max(0,sliceLine[1]-5)!=max(0,sliceLine[3]-1)):
                painter.drawLine(QtCore.QLineF(self.grabPosition+lineOffset,\
                    max(0,sliceLine[1]-5)+lineOffset,self.grabPosition+\
                    lineOffset,max(0,sliceLine[3]-1)+0.5))
            if(min(self.height,sliceLine[1]+1)!=min(self.height,sliceLine[3]+5)):
                painter.drawLine(QtCore.QLineF(self.grabPosition+lineOffset,\
                    min(self.height,sliceLine[1]+1)+lineOffset,\
                    self.grabPosition+lineOffset,min(self.height,\
                    sliceLine[3]+5)+lineOffset))
            if(max(0,sliceLine[0])!=max(0,self.grabPosition-1)):
                painter.drawLine(QtCore.QLineF(max(0,sliceLine[0])+lineOffset,\
                sliceLine[1]+lineOffset,max(0,self.grabPosition-1)+lineOffset,\
                sliceLine[3]+lineOffset))
            painter.drawLine(QtCore.QLineF(min(self.width,self.grabPosition+1)+\
                lineOffset,sliceLine[1]+lineOffset,sliceLine[2]+lineOffset,\
                sliceLine[3]+lineOffset))
        
        if(self.drawSliceLine and (self.etype=='drusenViewer' or self.etype=='layerViewer')):
            self.pen = QtGui.QPen(QtCore.Qt.SolidLine)
            self.pen.setColor(QtGui.QColor(0,255,255,170))
            self.pen.setWidth(1.0)
            painter.setPen(self.pen)            
            sliceLine=self.sliceline
            painter.drawLine(QtCore.QLineF(float(self.grabPosition)+lineOffset,\
                0.+lineOffset,float(self.grabPosition)+lineOffset,\
                float(self.height-1)+lineOffset))
        
        if(self.etype=='hrfViewer'):
            if(self.sliceNum-1 in self.bboxList.keys()):
                rects=self.bboxList[self.sliceNum-1]
                self.pen = QtGui.QPen(QtCore.Qt.SolidLine)
                self.pen.setColor(QtGui.QColor(0,255,255,170))
                self.pen.setWidth(0.5)
                painter.setPen(self.pen)
                for rect in rects:
                    painter.drawRect(rect)
                    
        if(self.etype=='gaViewer'):
            if(self.sliceNum-1 in self.bboxList.keys()):
                rects=self.bboxList[self.sliceNum-1]
                self.pen = QtGui.QPen(QtCore.Qt.SolidLine)
                self.pen.setColor(QtGui.QColor(0,255,255,170))
                self.pen.setWidth(0.5)
                painter.setPen(self.pen)
                for rect in rects:
                    painter.drawRect(rect)               
               
        if(self.etype=='enfaceDrusenViewer'):
            for p in self.drusenPeaks:
                self.pen = QtGui.QPen(QtCore.Qt.SolidLine)
                self.pen.setColor(QtGui.QColor(255,0,0,170))
                self.pen.setWidth(1.)
                painter.setPen(self.pen)
                painter.drawPoint(QtCore.QPointF(p[0]+0.5,p[1]+0.5))
                
            if(not self.separators is None):
                separators=np.copy(self.separators)
                if(len( np.where(self.separatorAvgHeight>\
                        self.separatorHeightThreshold)[0])>0):
                    separators[np.where(self.separatorAvgHeight>\
                        self.separatorHeightThreshold)]=0.
                self.pen = QtGui.QPen(QtCore.Qt.SolidLine)
                self.pen.setColor(QtGui.QColor(0,0,0,170))
                self.pen.setWidth(1.)
                painter.setPen(self.pen)
                
                sy,sx=np.where(separators==1)
                for i in range(len(sy)):
                    painter.drawPoint(QtCore.QPointF(sx[i]+0.5,sy[i]+0.5))
                sy,sx=np.where(separators==2)
                for i in range(len(sy)):
                    painter.drawPoint(QtCore.QPointF(sx[i]+0.5,sy[i]+0.5))  
                    
        if(self.etype=='drusenViewer'):
            if(not self.separators is None):
                self.separators.shape
                separators=np.copy(self.separators)
                if(len( np.where(self.separatorAvgHeight>\
                        self.separatorHeightThreshold)[0])>0):
                    separators[np.where(self.separatorAvgHeight>\
                        self.separatorHeightThreshold)]=0.
                    
                self.pen = QtGui.QPen(QtCore.Qt.SolidLine)
                self.pen.setColor(QtGui.QColor(0,0,0,170))
                self.pen.setWidth(1.)
                painter.setPen(self.pen)
                sx=np.where(separators[self.sliceNum-1,:]==1)[0]
                dru=self.mainPhoto
                for x in sx:
                    sy=np.where(dru[:,x]>0)[0]
                    
                    for y in sy:
                        painter.drawPoint(QtCore.QPointF(x+0.5,y+0.5))
                    if(len(sy)>0):
                        minimum=max(0,min(sy)-1)
                        painter.drawPoint(QtCore.QPointF(x+0.5,minimum+0.5))
            
        if(self.redrawLayers and self.etype=='layerViewer'):
            self.combine_images()
            self.redrawLayers=False
        
        if(self.showSplineKnots and self.etype=='layerViewer'):
            knots=self.controller.get_spline_knots(self.sliceNum-1)          
        
            if(knots is None):
                return
            lineThikness=1.
            knotsx,knotsy=knots
            self.pen = QtGui.QPen(QtCore.Qt.SolidLine)
            self.pen.setColor(QtGui.QColor(255,0,0,150))
            self.pen.setWidth(lineThikness)
            painter.setPen(self.pen)
                
            for i in range(len(knotsx)):
                painter.drawPoint(QtCore.QPointF(knotsx[i]+0.5,knotsy[i]+0.5))
                painter.drawPoint(QtCore.QPointF(knotsx[i]+0.5,knotsy[i]+0.5+1))
                painter.drawPoint(QtCore.QPointF(knotsx[i]+0.5,knotsy[i]+0.5-1))
                painter.drawPoint(QtCore.QPointF(knotsx[i]+0.5+1,knotsy[i]+0.5))
                painter.drawPoint(QtCore.QPointF(knotsx[i]+0.5-1,knotsy[i]+0.5))
                
    def set_uncertainties(self,unEnt,unProb,entCol,probCol):
        self.entropyValsPerBcan=unEnt
        self.probabilityValsPerBcan=unProb
        self.entropyColor=entCol
        self.probabilityColor=probCol
        
    def setPaintingColor(self,button): 
        if(button==1):#Left click
            self.color.setRgb(255,255,255)
        elif(button==2):#right click
            self.color.setRgb(0,0,0)
    
    def mousePressEvent (self, event):
        modifiers = QtGui.QApplication.keyboardModifiers()
        self.setPaintingColor(event.button())        
        
        self.x=event.pos().x()
        self.y=event.pos().y()
        self.controller.write_in_log(self.controller.get_time()+',clicked,'+\
            self.etype+','+self.controller.get_current_active_window()+'\n')
        self.clickType=event.button()
        if(self.selectRect):
            isRightClick=event.button()==2
            if((self.showSplineKnots and isRightClick) or (not self.showSplineKnots)):
                self.rectFixPoint.setX(self.x)
                self.rectFixPoint.setY(self.y)
        if(self.selectLine):
            x=int(self.x)+0.5
            y=int(self.y)+0.5
            self.line.setP1(QtCore.QPointF(x,y))
            self.line.setP2(QtCore.QPointF(x,y))
        elif(self.fillArea):
            image=self.mainPhoto
            overImg=self.overlayedPhotos[1]
            image=self.controller.fill_in_area(image,overImg,\
                int(math.floor(self.x)),int(math.floor(self.y)),\
                self.color.getRgb(),self.etype)            
            self.set_main_photo(image)
            self.combine_images()
            self.update()
        elif(self.drawPen):   
            self.draw_point(event)
        elif(self.markCostPnt):
            self.controller.update_cost_image(min(max(0,int(self.x)),\
                self.width),min(max(0,int(self.y)),self.height),\
                self.smoothness,self.etype)
            
        elif(self.markDrusenPeak and not self.mergeDrusenMode):
            x=int(self.x)
            y=int(self.y)
            if(event.button()==1):
                if(x>=0 and x<self.width and y>=0 and y<self.height):
                    self.drusenPeaks.add((x,y))
            elif(event.button()==2):
                if(x>=0 and x<self.width and y>=0 and y<self.height and\
                        (x,y) in self.drusenPeaks):
                    self.drusenPeaks.remove((x,y))
            self.combine_images()
            self.update()   
        elif(self.mergeDrusenMode):
            # If ctrl is hold, select the current druse as merging source
            if(modifiers == QtCore.Qt.ControlModifier):
                self.sourceLabel=self.labels[int(self.y),int(self.x)]
            else: # By click merge the druse with the source
                self.copy_source_label_to_destination(int(self.y),int(self.x))
        
        elif(self.showSplineKnots):
            isRightClick=event.button()==2
            if(isRightClick): # Delete knot
                self.controller.delete_spline_knot(int(self.y),int(self.x),self.sliceNum-1)
            else:       
                knotExists=self.controller.is_knot(int(self.y),int(self.x),self.sliceNum-1)
                if(knotExists):
                    self.hasPickedKnot=True
                    kx,ky=self.controller.get_closest_knot(int(self.y),int(self.x),self.sliceNum-1)
                    self.pickedKnot=[kx,ky]
                if(modifiers == QtCore.Qt.ControlModifier):
                    if(not self.hasPickedKnot):
                        self.controller.add_spline_knot(int(self.y),int(self.x),self.sliceNum-1)
        if(self.updateGrab):
            self.grabPosition=self.x
            self.controller.grap_position_changed(int(self.x),int(self.y),self.etype)
            
    def copy_source_label_to_destination(self,y,x):
        if(self.labels is None):
            return
        destLabel=self.labels[y,x]
        if(destLabel==0):
            return
            
        self.labels[np.where(self.labels==destLabel)]=self.sourceLabel
        self.separators[np.where(self.labels==self.sourceLabel)]=0.
        self.separatorAvgHeight[np.where(self.labels==self.sourceLabel)]=0
        druBoundary=np.empty(self.labels.shape)
        druBoundary.fill(0.)
        druBoundary[np.where(self.labels==self.sourceLabel)]=1.
        dilDruBoundry=sc.ndimage.morphology.binary_dilation(druBoundary,\
            iterations=1)
        dilDruBoundry=dilDruBoundry-druBoundary
        self.separators[np.where(dilDruBoundry==1)]=1.
        self.combine_images()

    def mouseMoveEvent (self, event):
        self.x=event.pos().x()
        self.y=event.pos().y()  
        
        if(self.selectRect):
            isRightClick=self.clickType==2
            if((self.showSplineKnots and isRightClick) or (not self.showSplineKnots)):
                self.draw_rect_area(event)
        elif(self.selectLine):
            x=int(self.x)+0.5
            y=int(self.y)+0.5
            self.line.setP2(QtCore.QPointF(x,y))
            self.update()
        elif(self.drawPen):
            self.controller.app.processEvents()
            self.draw_point_moving_mouse(event)
        if(self.updateGrab):
            self.grabPosition=self.x
            self.controller.grap_position_changed(self.x,self.y,self.etype)
        if(self.hasPickedKnot):
            self.controller.update_knot_position(int(self.y),int(self.x),\
                        self.pickedKnot[1],self.pickedKnot[0],self.sliceNum-1)
            del self.pickedKnot
            self.pickedKnot=[int(self.x),int(self.y)]
            
    def mouseReleaseEvent(self, event):
        if(self.selectLine):
            self.draw_line(event)
            
        if(self.selectRect):
            isRightClick=event.button()==2
            if((self.showSplineKnots and isRightClick) or (not self.showSplineKnots)):
                self.apply_function_in_rect(event) 
            
        if(self.drawPen):
            color=self.color.getRgb()
            self.controller.finished_drawing_with_pen(color,self.etype)
        if(self.hasPickedKnot):
            self.hasPickedKnot=False
            del self.pickedKnot   
            
    def draw_rect_area(self,event):
        tl=self.rectFixPoint.x(),self.rectFixPoint.y()
        width=(self.x-tl[0])
        height=(self.y-tl[1])
        topLeft=QtCore.QPoint()
        bottomRight=QtCore.QPoint()
        if(width>0 and height>0):
            topLeft.setX(math.floor(tl[0]))
            topLeft.setY(math.floor(tl[1]))
            bottomRight.setX(math.ceil(self.x))
            bottomRight.setY(math.ceil(self.y))
        elif(width<0 and height<0):
            topLeft.setX(math.floor(self.x))
            topLeft.setY(math.floor(self.y))
            bottomRight.setX(math.ceil(tl[0]))
            bottomRight.setY(math.ceil(tl[1]))
        elif(width<0):
            topLeft.setX(math.floor(self.x))
            topLeft.setY(math.floor(tl[1]))
            bottomRight.setX(math.ceil(tl[0]))
            bottomRight.setY(math.ceil(self.y))
        elif(height<0):
            topLeft.setX(math.floor(tl[0]))
            topLeft.setY(math.floor(self.y))
            bottomRight.setX(math.ceil(self.x))
            bottomRight.setY(math.ceil(tl[1]))
        if(self.etype=='drusenViewer' and self.drawDru):
            topLeft.setY(0)
            bottomRight.setY(self.height-1)
        self.rect.setTopLeft(topLeft)
        self.rect.setBottomRight(bottomRight)
        self.update()
    
    def remove_bbox(self,x,y,topLeftX,topLeftY,bottomRightX,bottomRightY):
        removedBoxes=list()
        dx=bottomRightX-topLeftX
        dy=bottomRightY-topLeftY
        if(dx==0 and dy==0):
            if(self.sliceNum-1 in self.bboxList.keys()):
                rects=self.bboxList[self.sliceNum-1]
                minSize=np.inf
                minIndx=-1
                i=0
                for rect in rects:
                    if(rect.contains(int(x),int(y))):
                        size=rect.width()*rect.height()
                        if(size<minSize):
                            minSize=size
                            minIndx=i
                    i+=1    
                if(minIndx>=0):
                    removedBoxes.append(self.bboxList[self.sliceNum-1][minIndx])
        else:
            if(self.sliceNum-1 in self.bboxList.keys()):
                rects=self.bboxList[self.sliceNum-1]
                i=0
                delRect=QtCore.QRect(QtCore.QPoint(topLeftX,\
                       topLeftY),QtCore.QPoint(bottomRightX-1,bottomRightY-1))
                delIndx=[]
                for rect in rects:
                    if(delRect.intersects(rect)):
                        delIndx.append(i)
                           
                    i+=1    
                delIndx=delIndx[::-1]
                for i in delIndx:
                    removedBoxes.append(self.bboxList[self.sliceNum-1][i])
        if(len(removedBoxes)>0):
            self.controller.removed_box_command(removedBoxes,self.sliceNum,\
                self.etype)
    
    def apply_function_in_rect(self,event):
        image=self.mainPhoto
        h,w=image.shape[0],image.shape[1]
        tl=self.rect.topLeft()
        br=self.rect.bottomRight()
        if(event.button()==1):
            topLeftX,topLeftY=max(0,int(tl.x())),max(0,int(tl.y()))
            bottomRightX,bottomRightY=min(w,int(br.x())),min(h,int(br.y()))
            if(self.morphology):
                image=self.controller.dilate_in_region(image,topLeftX,topLeftY,\
                    bottomRightX,bottomRightY,self.itLevel,self.etype)
            elif(self.drawDru):
                overImg=self.overlayedPhotos[1]
                image=self.controller.extract_drusen_in_region(image,overImg,\
                    topLeftX,topLeftY,bottomRightX,bottomRightY,self.etype,\
                    self.maxFilteringHeight)
            elif(self.filterHeight):
                image=self.controller.filter_drusen_wrt_height(image,\
                    self.filteringHeight,self.maxFilteringHeight,topLeftX,\
                    topLeftY,bottomRightX,bottomRightY,self.etype)
            elif(self.drawBBox and (self.etype=='hrfViewer' or\
                    self.etype=='enfaceViewer' or self.etype=='gaViewer')):
                bBox=QtCore.QRect(QtCore.QPoint(topLeftX,topLeftY),\
                    QtCore.QPoint(bottomRightX-1,bottomRightY-1))
                self.controller.box_command(bBox,self.sliceNum,self.etype)
            elif(self.fitPolynomial):
                self.controller.polyfit_in_region(image,topLeftX,topLeftY,\
                    bottomRightX,bottomRightY,self.polyDegreeValue,self.etype)
                del self.rect
                self.rect=QtCore.QRectF()
                return
        elif(event.button()==2):
            topLeftX,topLeftY=max(0,int(tl.x())),max(0,int(tl.y()))
            bottomRightX,bottomRightY=min(w,int(br.x())),min(h,int(br.y()))
            if(self.morphology):
                image=self.controller.erosion_in_region(image,topLeftX,topLeftY,\
                    bottomRightX,bottomRightY,self.itLevel,self.etype)
            elif(self.drawDru):
                image=self.controller.delete_drusen_in_region(image,topLeftX,\
                    topLeftY,bottomRightX,bottomRightY,self.etype)
            elif(self.drawBBox and (self.etype=='hrfViewer' or\
                    self.etype=='enfaceViewer' or self.etype=='gaViewer')):
                self.remove_bbox(event.pos().x(),event.pos().y(),topLeftX,\
                    topLeftY,bottomRightX,bottomRightY)
            elif(self.showSplineKnots and self.etype=='layerViewer'):
                self.controller.delete_spline_knots_in_region(topLeftX,\
                    topLeftY,bottomRightX,bottomRightY,self.sliceNum-1)

        self.set_main_photo(image)
        del self.rect
        self.rect=QtCore.QRectF()
        self.combine_images()
        self.update()

    def set_image_in_editor(self,image):
        self.set_main_photo(image)
        self.combine_images()
        self.update()

    def draw_line(self,event):
        prev=self.mainPhoto
        color=self.color.getRgb()
        prev=self.controller.draw_line(prev,int(self.line.x1()),int(self.line.y1()),\
                                int(self.line.x2()),int(self.line.y2()),color,self.etype)
        self.set_main_photo(prev)
        del self.line
        self.line=QtCore.QLineF()
        self.combine_images()
        self.update()
            
    def draw_point_moving_mouse(self,event):
        prev=self.mainPhoto
        self.prevX=self.x if self.prevX<0 else self.prevX
        self.prevY=self.y if self.prevY<0 else self.prevY
        color=self.color.getRgb()
        prev=self.controller.draw_line(prev,int(self.prevX),int(self.prevY),int(self.x),\
            int(self.y),color,self.etype,undoRedo=True,movingMouse=True)
        self.set_main_photo(prev)
        self.combine_images()
        self.update()
        self.prevX=self.x
        self.prevY=self.y
    
    def draw_point_moving_mouse_undo_redo(self,prevX,prevY,x,y,color):
        prev=self.mainPhoto
        prev=self.controller.draw_line(prev,int(prevX),int(prevY),int(x),int(y),color,self.etype,\
            undoRedo=True)
        self.set_main_photo(prev)
        self.combine_images()
        self.update()

    def draw_fill_undo_redo(self,x,y,color):
        if(len(x)>0):
            image=self.mainPhoto
            overImg=self.overlayedPhotos[1]
            image=self.controller.fill_in_area(image,overImg,\
                int(math.floor(x[0])),int(math.floor(y[0])),color,self.etype)

    def draw_point(self,event):
        self.prevX=self.x
        self.prevY=self.y
        prev=self.mainPhoto
        color=self.color.getRgb()
        prev=self.controller.draw_point(prev,int(self.x),int(self.y),color,self.sliceNum,\
            self.etype)
        self.set_main_photo(prev)
        self.combine_images()
        self.update()
        
    def draw_point_undo_redo(self,x,y,color):
        prev=self.mainPhoto
        prev=self.controller.draw_point(prev,int(x),int(y),color,self.sliceNum,self.etype,\
            undoRedo=True)
        self.set_main_photo(prev)
        self.combine_images()
        self.update()  
        
    def get_hrf_bounding_boxes(self):
        return self.bboxList

    def get_nga_bounding_boxes(self):
        return self.bboxList

    def get_enface_bounding_boxes(self):
        return self.bboxList

    def get_rastered_line_points(self,x1,y1,x2,y2):
        return list(bresenham(int(x1),int(y1),int(x2),int(y2)))
        
    def set_line(self,x1,y1,x2,y2):
        self.sliceline=(x1,y1,x2,y2)

    def set_slice(self,sliceNum):
        self.sliceline=(0,sliceNum,self.width-1,sliceNum)
        
    def add_overlay_image(self,image,v=0.0):
        self.overlayedPhotos.append(image)
        self.coeffs.append(v)
        self.combine_images()
        
    def set_overlay_image(self,image,index):
        self.overlayedPhotos[index]=image
        self.combine_images()
     
    def update_suggestion_layers(self,image):
        self.suggestionLayerImg=image
        
    def set_main_photo(self,image):
        self.mainPhoto=image
        self.overlayedPhotos[0]=image
        self.combine_images()

    def set_coeffs(self,coeffs):
        self.coeffs=coeffs
        
    def apply_height_thresholds(self):
        image=self.mainPhoto
        h,w=image.shape[0],image.shape[1]
        image=self.controller.filter_drusen_wrt_height(image,\
            self.filteringHeight,self.maxFilteringHeight,0,0,w-1,h-1,self.etype)
        self.set_main_photo(image)
        self.combine_images()
        self.update()    
    
    def map_label_number_to_color(self,n,minima,maxima):
        norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.Blues)
        c1=mapper.to_rgba(n)
        return [c1[0],c1[1],c1[2]]
        
    def show_labels_on_cca(self):
        labels=np.unique(self.labels)
        ccaMask=np.copy(self.ccaMask)
        for l in labels:
            if(l==0):
                continue
            color=self.map_label_number_to_color(l,min(labels),max(labels))
            color=(np.asarray(color)*255).astype(int)
            y,x=np.where(self.labels==l)
            ccaMask[y,x,0]=color[0]
            ccaMask[y,x,1]=color[1]
            ccaMask[y,x,2]=color[2]
            
        return ccaMask
        
    def combine_images(self):
        lineThikness=1
        mainPhoto=self.overlayedPhotos[0]
        if(mainPhoto is None):
            return
        self.height,self.width=mainPhoto.shape
        res=np.empty((self.height,self.width,3))
        if(not self.showAnnotation and len(self.overlayedPhotos)>1):
            scan=self.overlayedPhotos[1]
            if(self.etype=='drusenViewer'):
                scan=self.overlayedPhotos[2]
            qimg=q2np.array2qimage(scan)
            self.setPixmap(QtGui.QPixmap.fromImage(qimg))
            return    
        if(self.etype=='layerViewer'  ):
            res.fill(0)
            if(self.showUncertainties):
                uncType='Entropy' if self.uncertaintyType=='entropy' else 'Probability'
                photo=self.overlayedPhotos[1]
                mrpe=np.copy(self.controller.oct.get_uncer_map('RPE',uncType)[self.sliceNum-1,:])
                mbm=np.copy(self.controller.oct.get_uncer_map('BM',uncType)[self.sliceNum-1,:])
                
                rpeEdited=self.editedLayers[self.sliceNum-1]['RPE']
                bmEdited=self.editedLayers[self.sliceNum-1]['BM']
                if(rpeEdited):
                    mrpe[:,0]=255
                    mrpe[:,1]=255
                    mrpe[:,2]=0
                if(bmEdited):
                    mbm[:,0]=255
                    mbm[:,1]=255
                    mbm[:,2]=0
                y,x=np.where(mainPhoto>170)
                res[y,x,:]=(np.ones(res.shape)*mrpe[np.newaxis,:])[y,x,:]
                a=(mainPhoto<170).astype(int)
                a[mainPhoto==0]=0
                y,x=np.where(a>0)
                res[y,x,:]=(np.ones(res.shape)*mbm[np.newaxis,:])[y,x,:]
                y,x=np.where(mainPhoto==170)
                res[y,x,:]=(np.ones(res.shape)*mrpe[np.newaxis,:])[y,x,:]
                
            else:
                y,x=np.where(mainPhoto>170)
                res[y,x,0]=0
                res[y,x,1]=255
                res[y,x,2]=255
                a=(mainPhoto<170).astype(int)
                a[mainPhoto==0]=0
                y,x=np.where(a>0)
                res[y,x,0]=255
                res[y,x,1]=166
                res[y,x,2]=0
                y,x=np.where(mainPhoto==170)
                res[y,x,0]=255
                res[y,x,1]=210
                res[y,x,2]=255
            showGT=False
            if(len(self.overlayedPhotos)>1):
                    photo=self.overlayedPhotos[1]
                    if(showGT):
                        gtLayer=self.controller.get_GT_layer()
                        res2=np.empty((self.height,self.width,3))
                        res2.fill(0.)
                        res3=np.empty((self.height,self.width,3))
                        res3.fill(0.)
                        y,x=np.where(gtLayer>127)
                        res2[y,x,0]=0
                        res2[y,x,1]=255
                        res2[y,x,2]=0
                        
                        halfSize=np.floor(lineThikness/2).astype(int)
                        for i in range(halfSize):
                            shiftedUp=np.roll(res2,-(i+1),axis=0)
                            shiftedDown=np.roll(res2,(i+1),axis=0)
                            shiftedLeft=np.roll(res2,-(i+1),axis=1)
                            shiftedRight=np.roll(res2,(i+1),axis=1)
                            res2=np.maximum(res2,shiftedUp)
                            res2=np.maximum(res2,shiftedDown)
                            res2=np.maximum(res2,shiftedLeft)
                            res2=np.maximum(res2,shiftedRight)
                        
                        if(not self.suggestionLayerImg is None):
                            
                            y,x=np.where(self.suggestionLayerImg>1)
                            res3[y,x,0]=255
                            res3[y,x,1]=0
                            res3[y,x,2]=0
                            y,x=np.where(self.suggestionLayerImg==1)
                            res3[y,x,0]=0
                            res3[y,x,1]=255
                            res3[y,x,2]=0
                            halfSize=np.floor(lineThikness/2).astype(int)
                            for i in range(halfSize):
                                shiftedUp=np.roll(res3,-(i+1),axis=0)
                                shiftedDown=np.roll(res3,(i+1),axis=0)
                                shiftedLeft=np.roll(res3,-(i+1),axis=1)
                                shiftedRight=np.roll(res3,(i+1),axis=1)
                                res3=np.maximum(res3,shiftedUp)
                                res3=np.maximum(res3,shiftedDown)
                                res3=np.maximum(res3,shiftedLeft)
                                res3=np.maximum(res3,shiftedRight)
                        halfSize=np.floor(lineThikness/2).astype(int)
                        for i in range(halfSize):
                            shiftedUp=np.roll(res,-(i+1),axis=0)
                            shiftedDown=np.roll(res,(i+1),axis=0)
                            shiftedLeft=np.roll(res,-(i+1),axis=1)
                            shiftedRight=np.roll(res,(i+1),axis=1)
                            res=np.maximum(res,shiftedUp)
                            res=np.maximum(res,shiftedDown)
                            res=np.maximum(res,shiftedLeft)
                            res=np.maximum(res,shiftedRight)
                        x,y,z=np.where(res>0)
                        res2[x,y,0]=res[x,y,0]
                        res2[x,y,1]=res[x,y,1]
                        res2[x,y,2]=res[x,y,2]
                        
                        x,y,z=np.where(res3>0)
                        res2[x,y,0]=res3[x,y,0]
                        res2[x,y,1]=res3[x,y,1]
                        res2[x,y,2]=res3[x,y,2]
                        res=res2
                    else:
                        if(not self.suggestionLayerImg is None):
                            res2=np.empty((self.height,self.width,3))
                            res2.fill(0.)
                            y,x=np.where(self.suggestionLayerImg>1)
                            res2[y,x,0]=255
                            res2[y,x,1]=0
                            res2[y,x,2]=0
                            y,x=np.where(self.suggestionLayerImg==1)
                            res2[y,x,0]=0
                            res2[y,x,1]=255
                            res2[y,x,2]=0
                            res=res+res2
                        halfSize=np.floor(lineThikness/2).astype(int)
                        for i in range(halfSize):
                            shiftedUp=np.roll(res,-(i+1),axis=0)
                            shiftedDown=np.roll(res,(i+1),axis=0)
                            shiftedLeft=np.roll(res,-(i+1),axis=1)
                            shiftedRight=np.roll(res,(i+1),axis=1)
                            res=np.maximum(res,shiftedUp)
                            res=np.maximum(res,shiftedDown)
                            res=np.maximum(res,shiftedLeft)
                            res=np.maximum(res,shiftedRight)
                    res[:,:,0]=photo*self.coeffs[1]+res[:,:,0]*self.coeffs[0]
                    res[:,:,1]=photo*self.coeffs[1]+res[:,:,1]*self.coeffs[0]
                    res[:,:,2]=photo*self.coeffs[1]+res[:,:,2]*self.coeffs[0]
         
            qimg=q2np.array2qimage(res)
            self.setPixmap(QtGui.QPixmap.fromImage(qimg))
            return
            
        if(self.etype=='hrfViewer'):
            res.fill(0)
            if(self.etype=='hrfViewer' and len(self.overlayedPhotos)>1):
                    photo=self.overlayedPhotos[1]

                    y,x=np.where(mainPhoto>0)
                    res[y,x,0]=255
                    res[y,x,1]=0
                    res[y,x,2]=0
                    
                    res[:,:,0]=photo*self.coeffs[1]+res[:,:,0]*self.coeffs[0]
                    res[:,:,1]=photo*self.coeffs[1]+res[:,:,1]*self.coeffs[0]
                    res[:,:,2]=photo*self.coeffs[1]+res[:,:,2]*self.coeffs[0]

            qimg=q2np.array2qimage(res)
            self.setPixmap(QtGui.QPixmap.fromImage(qimg))
            return

        if(self.etype=='gaViewer'):
            res.fill(0)
            if(self.etype=='gaViewer' and len(self.overlayedPhotos)>1):
                    photo=self.overlayedPhotos[1]
                    y,x=np.where(mainPhoto==1.)
                    res[y,x,0]=255
                    res[y,x,1]=117
                    res[y,x,2]=128
                    y,x=np.where(mainPhoto==2.)
                    res[y,x,0]=117
                    res[y,x,1]=213
                    res[y,x,2]=255
                    y,x=np.where(mainPhoto==3.)
                    res[y,x,0]=229
                    res[y,x,1]=135
                    res[y,x,2]=255
                    res[:,:,0]=photo*self.coeffs[1]+res[:,:,0]*self.coeffs[0]
                    res[:,:,1]=photo*self.coeffs[1]+res[:,:,1]*self.coeffs[0]
                    res[:,:,2]=photo*self.coeffs[1]+res[:,:,2]*self.coeffs[0]
            qimg=q2np.array2qimage(res)
            self.setPixmap(QtGui.QPixmap.fromImage(qimg))
            return   
            
        if(self.etype=='drusenViewer'):
            res.fill(0)
            
            if(len(self.overlayedPhotos)>2):
                    photoL=self.overlayedPhotos[1]
                    y,x=np.where(photoL>170)
                    res[y,x,0]=0
                    res[y,x,1]=255
                    res[y,x,2]=255
                    a=(photoL<170).astype(int)
                    a[photoL==0]=0
                    y,x=np.where(a>0)
                    res[y,x,0]=255
                    res[y,x,1]=166
                    res[y,x,2]=0
                    y,x=np.where(photoL==170)
                    res[y,x,0]=255
                    res[y,x,1]=210
                    res[y,x,2]=255
                    
                    photoS=self.overlayedPhotos[2]
                    if(self.ccaMask is None):
                        res[:,:,0]=photoS*self.coeffs[2]+mainPhoto*\
                            self.coeffs[0]+res[:,:,0]*self.coeffs[1]
                        res[:,:,1]=photoS*self.coeffs[2]+res[:,:,1]*self.coeffs[1]
                        res[:,:,2]=photoS*self.coeffs[2]+res[:,:,2]*self.coeffs[1]
                    
                    else:
                        mask=(mainPhoto>0).astype(int)
                        m1=(np.sum(self.ccaMask[self.sliceline[1],:,:],\
                            axis=1)==0).astype(int)
                        m1=m1*(np.sum(mask,axis=0).astype(int))
                        gx=np.where(m1>0)
                        self.ccaMask[self.sliceline[1],gx,0]=0
                        self.ccaMask[self.sliceline[1],gx,1]=255
                        self.ccaMask[self.sliceline[1],gx,2]=0
                        
                        res[:,:,0]=photoS*self.coeffs[2]+\
                            self.ccaMask[self.sliceline[1],:,0][np.newaxis,:]*\
                            mask*self.coeffs[0]+res[:,:,0]*self.coeffs[1]
                        res[:,:,1]=photoS*self.coeffs[2]+\
                            self.ccaMask[self.sliceline[1],:,1][np.newaxis,:]*\
                            mask*self.coeffs[0]+res[:,:,1]*self.coeffs[1]
                        res[:,:,2]=photoS*self.coeffs[2]+\
                            self.ccaMask[self.sliceline[1],:,2][np.newaxis,:]*\
                            mask*self.coeffs[0]+res[:,:,2]*self.coeffs[1]

            qimg=q2np.array2qimage(res)
            self.setPixmap(QtGui.QPixmap.fromImage(qimg))
            return
        if(self.etype=='enfaceDrusenViewer'):
            res.fill(0)
            if(len(self.overlayedPhotos)>1):
                    photo=self.overlayedPhotos[1]
                    if(self.ccaMask is None):
                        res[:,:,0]=photo*self.coeffs[1]+mainPhoto*self.coeffs[0]
                        res[:,:,1]=photo*self.coeffs[1]
                        res[:,:,2]=photo*self.coeffs[1]
                    else:
                        mask=(mainPhoto>0).astype(int)
                        m1=(np.sum(self.ccaMask,axis=2)==0).astype(int)
                        m1=m1*mask
                        gy,gx=np.where(m1>0)
              
                        self.ccaMask[gy,gx,0]=0
                        self.ccaMask[gy,gx,1]=255
                        self.ccaMask[gy,gx,2]=0
                        
                        ccaMask=np.copy(self.ccaMask)
                        if(not self.labels is None):
                            ccaMask=self.show_labels_on_cca()
                        res[:,:,0]=photo*self.coeffs[1]+ccaMask[:,:,0]*\
                            mask*self.coeffs[0]
                        res[:,:,1]=photo*self.coeffs[1]+ccaMask[:,:,1]*\
                            mask*self.coeffs[0]
                        res[:,:,2]=photo*self.coeffs[1]+ccaMask[:,:,2]*\
                            mask*self.coeffs[0]
               
            qimg=q2np.array2qimage(res)
            self.setPixmap(QtGui.QPixmap.fromImage(qimg))
            return
        if(self.etype=='enfaceViewer'):
            res.fill(0)
            res[:,:,0]=mainPhoto
            res[:,:,1]=mainPhoto
            res[:,:,2]=mainPhoto
            if(len(self.overlayedPhotos)>1 and not self.overlayedPhotos[1] is None):
                    photo=self.overlayedPhotos[1]
                    res[:,:,0]=res[:,:,0]*self.coeffs[0]+photo[:,:,0]*self.coeffs[1]
                    res[:,:,1]=res[:,:,1]*self.coeffs[0]+photo[:,:,1]*self.coeffs[1]
                    res[:,:,2]=res[:,:,2]*self.coeffs[0]+photo[:,:,2]*self.coeffs[1]
               
            qimg=q2np.array2qimage(res)
            self.setPixmap(QtGui.QPixmap.fromImage(qimg))
            return
        res[:,:,0]=mainPhoto
        res[:,:,1]=mainPhoto
        res[:,:,2]=mainPhoto
        for i in range(len(self.overlayedPhotos)):
            if(self.etype=='drusenViewer'):
                photo=self.overlayedPhotos[i]
                res[:,:,i]=res[:,:,i]+photo*self.coeffs[i]
            elif(self.etype=='enfaceDrusenViewer'):
                photo=self.overlayedPhotos[i]
                res[:,:,i]=res[:,:,i]+photo*self.coeffs[i]
        qimg=q2np.array2qimage(res)
        self.setPixmap(QtGui.QPixmap.fromImage(qimg))
        
    def show_image(self, image, block = True ):
        plt.imshow( image)
        plt.show(block)    
        QtGui.QApplication.processEvents() 

#==============================================================================
# The widget which contains the major annotation graphics pixmap item
#==============================================================================
class ImageEditor(QtGui.QWidget):
    
    def __init__(self,controller=None,editorType=''):
        super(ImageEditor, self).__init__()
        self.controller=controller
        self.scene = QtGui.QGraphicsScene(self)
        self.etype=editorType
        self.view=MyGraphicsView(self)
        self.view.zoom=0
        self.view.setScene(self.scene)
        self.view.scale(1.,1.)
        if(editorType=='enfaceViewer' or editorType=='enfaceDrusenViewer'):
            hh=self.controller.get_enface_height()
            if(hh<20):
                self.view.scale(1.,4.)
            else:
                self.view.scale(1.,1.5)
        self.imagePanel=None
        self.mainPhoto=None
        self.mainPhotoPixMap=None
        self.view.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        self.view.setFrameShape(QtGui.QFrame.NoFrame)
        self.set_main_image(self.mainPhoto)
       
        layout = QtGui.QHBoxLayout()        
        layout.addWidget(self.view)
        self.setLayout(layout)

    def set_main_image(self,image):
        self.mainPhoto=image
        self.scene.clear()
        if((self.etype=='enfaceViewer' or self.etype=='enfaceDrusenViewer') and\
                (not image is None)):
            if(image.shape[0]<20):
                 self.view.scale(1.,4.)
            else:
                 self.view.scale(1.,1.5)
        self.imagePanel = ImageDrawPanel(scene = self.scene,controller=\
            self.controller,editorType=self.etype)
        self.imagePanel.set_main_photo(image)
        self.scene.addItem(self.imagePanel)
        self.scene.update()
        
    def update_main_image(self,image):
        self.mainPhoto=image
        self.imagePanel.set_main_photo(image)

    def add_overlay_image(self,image,coeff=0):
        self.imagePanel.add_overlay_image(image,coeff)

    def update_overlay_image(self,image,index):
        self.imagePanel.set_overlay_image(image,index)
    
    def set_coeffs(self,coeffs):
        self.imagePanel.set_coeffs(coeffs)
    
    def set_cca_mask(self,mask):
        self.imagePanel.set_cca_mask(mask)
        
    def unset_cca_mask(self):
        self.imagePanel.unset_cca_mask()
    
    def set_pen(self):
        self.imagePanel.set_pen()
        
    def set_line(self):
        self.imagePanel.set_line2()
        
    def set_fill(self):
        self.imagePanel.set_fill()
    
    def set_draw_dru(self):
        self.imagePanel.set_draw_dru()
    
    def set_morphology(self,itLevel):
        self.imagePanel.set_morphology(itLevel)
        
    def set_filter_dru(self,filteringHeight,maxFilteringHeight):
        self.imagePanel.set_filter_dru(filteringHeight,maxFilteringHeight)
        
    def set_grab(self):
        self.imagePanel.set_grab()
    
    def set_bounding_box(self):
        self.imagePanel.set_bounding_box()
     
    def set_cost_point(self):
        self.imagePanel.set_cost_point()
    
    def set_poly_fit(self,value):
        self.imagePanel.set_poly_fit(value)
        
    def all_threshold_value_changed(self,value):
        self.imagePanel.all_threshold_value_changed(value)        
                
    def morphology_value_changed(self,value):
        self.imagePanel.morphology_value_changed(value)
        
    def poly_degree_value_changed(self,value):
        self.imagePanel.poly_degree_value_changed(value)

    def smoothness_value_changed(self,value):
        self.imagePanel.smoothness_value_changed(value)

    def max_threshold_value_changed(self,value):
        self.imagePanel.max_threshold_value_changed(value)
    
    def grab_value_changed(self,position):
        self.imagePanel.grab_value_changed(position)
        
    def apply_height_threholds(self):
        self.imagePanel.apply_height_thresholds()
        
    def update_slice_number(self,sliceNum):
        self.imagePanel.update_slice_number(sliceNum)
        
    def get_hrf_bounding_boxes(self):
        return self.imagePanel.get_hrf_bounding_boxes()
        
    def get_nga_bounding_boxes(self):
        return self.imagePanel.get_nga_bounding_boxes()

    def get_enface_bounding_boxes(self):
        return self.imagePanel.get_enface_bounding_boxes()

    def set_HRF_BBox(self,hrfBBox):
        self.imagePanel.set_HRF_BBox(hrfBBox)

    def set_nga_BBox(self,ngaBBox):
        self.imagePanel.set_nga_BBox(ngaBBox)

    def set_enface_BBox(self,enfaceBBox):
        self.imagePanel.set_enface_BBox(enfaceBBox)
        
    def delete_box(self,rect,sliceNum):
        self.imagePanel.delete_box(rect,sliceNum)

    def add_box(self,rect,sliceNum):
        self.imagePanel.add_box(rect,sliceNum)

    def delete_boxes(self,rects,sliceNum):
        self.imagePanel.delete_boxes(rects,sliceNum)

    def add_boxes(self,rects,sliceNum):
        self.imagePanel.add_boxes(rects,sliceNum)

    def get_manual_markers(self):
        return self.imagePanel.get_manual_markers()

    def toggle_annotation_view(self):
        self.imagePanel.toggle_annotation_view()

    def get_cursor_status(self):
        return self.imagePanel.get_cursor_status()

    def get_drusen_separators(self):
        return self.imagePanel.get_drusen_separators()

    def done_splitting(self):
        self.imagePanel.done_splitting()
        
    def set_separation_threshold(self,value):
        self.imagePanel.set_separation_threshold(value)
        
    def update_line(self,sliceNum):
        self.imagePanel.set_slice(sliceNum)
        
    def set_manual_marker_selection(self):
        self.imagePanel.set_manual_marker_selection()

    def show_drusen_splitting_separators(self,separators,separatorsAvgHeight,\
            labels):    
        self.imagePanel.show_drusen_splitting_separators(separators,\
            separatorsAvgHeight,labels)

    def set_merge_drusen(self):
        self.imagePanel.set_merge_drusen()

    def unset_merge_drusen(self):
        self.imagePanel.unset_merge_drusen()

    def fitInView(self):
        if(self.mainPhoto is None):
            return
        h,w=self.mainPhoto.shape
        self.view.fitInView(QtCore.QRectF(0, 0, h,w), QtCore.Qt.KeepAspectRatio)
        rect=QtCore.QRectF(0,0,h,w)
        if not rect.isNull():
            unity = self.view.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
            self.view.scale(1. / float(unity.width()), 1. / float(unity.height()))
            viewrect = self.view.viewport().rect()
            scenerect = self.view.transform().mapRect(rect)
            
            factor = min(viewrect.width() / scenerect.width(),
                         viewrect.height() / scenerect.height())
            self.view.scale(factor, factor)
            self.view.centerOn(rect.center())

    def set_drusen_separators(self,separators):
        self.imagePanel.set_drusen_separators(separators)

    def set_image_in_editor(self,image):
        self.imagePanel.set_image_in_editor(image)
    
    def zoomFactor(self):
        return self.view.zoom
    
    def show_uncertainties(self,status):
        self.imagePanel.show_uncertainties(status)
        
    
    def set_uncertainties(self,unEnt,unProb,colEnt=None,colProb=None):
        self.imagePanel.set_uncertainties(unEnt,unProb,colEnt,colProb)
    
    def set_uncertainty_type(self,utype):
        self.imagePanel.set_uncertainty_type(utype)

    def spline_to_curve(self):
        self.imagePanel.spline_to_curve()
    
    def curve_to_spline(self):
        self.imagePanel.curve_to_spline()
    def get_smoothness(self):
        return self.imagePanel.get_smoothness()
    def update_suggestion_layers(self,img):
        self.imagePanel.update_suggestion_layers(img)
        
    def set_edited_layers(self,editedLayers):
        self.imagePanel.set_edited_layers(editedLayers)
    
