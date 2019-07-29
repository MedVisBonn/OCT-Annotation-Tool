# -*- coding: utf-8 -*-
"""
Created in 2018

@author: Shekoufeh Gorgi Zadeh
"""

import numpy as np
import scipy as sc
import toolbox as tb
import os, sys, inspect
from skimage import  io
import image_viewer as iv
import settings as settings
import progress_bar_view as pbv
from PyQt4 import QtCore, QtGui
from matplotlib import pyplot as plt
from drusen_info_view import Ui_drusenInfoTable

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
# Software's main window class. Used to handle closing
#==============================================================================
class MainWindow(QtGui.QMainWindow):
    
    def __init__(self, controller):
        super(MainWindow, self).__init__()
        self.controller=controller
        
    def closeEvent(self,event):
        if(self.controller.is_there_unsaved_changes()):
            choice=QtGui.QMessageBox.question(self,"Close",\
                        "Save the changes before closing?",\
                        QtGui.QMessageBox.Discard | QtGui.QMessageBox.Cancel |\
                        QtGui.QMessageBox.Save)
            if(choice==QtGui.QMessageBox.Discard):
                return event.accept()
            if(choice==QtGui.QMessageBox.Cancel):
                return event.ignore()   
            if(choice==QtGui.QMessageBox.Save):
                self.controller.save()
                return event.accept()
            return event.ignore()   
        else:
            return event.accept()
 
#==============================================================================
# Subview class is defined to handle viewers           
#==============================================================================
class SubView(QtGui.QWidget): 
    
    def __init__(self,parentWindow,controller,name):
        QtGui.QWidget.__init__(self)
        self.controller=controller
        self.name=name
        self.parentWindow=parentWindow
        
    def update_further(self):
        self.adjustSize()
        
    def get_name(self):
        return self.name
        
    def closeEvent(self,event):
        self.parentWindow.hide_subwindow(self)
        event.accept()
        
    def keyPressEvent(self,event):
        if(event.key()==QtCore.Qt.Key_B):
            self.controller.annotation_view_toggled(self.name)
        else:
            return QtGui.QWidget.keyPressEvent(self, event)
    
#==============================================================================
# Major Operations defined as classes that can be undone or redone 
#==============================================================================
class DrawCostPointCommand(QtGui.QUndoCommand):
    
    def __init__(self,parent,x,y,smoothness,layerName,viewName,sliceNum):
        QtGui.QUndoCommand.__init__(self)
        super(DrawCostPointCommand,self).__init__()
        self.x=x
        self.y=y
        self.layerName=layerName
        self.viewName=viewName
        self.sliceNum=sliceNum
        self.parentWind=parent
        self.smoothness=smoothness
        self.info=None
        
    def undo(self):
        self.parentWind.draw_cost_point_command_undo_redo(self.x,self.y,\
                self.smoothness,self.layerName,self.viewName,self.info,\
                self.sliceNum,'undo')
        
    def redo(self):
        self.parentWind.oct_controller.set_slice_number(self.sliceNum,self.viewName)
        self.info=self.parentWind.draw_cost_point_command_undo_redo(self.x,\
                self.y,self.smoothness,self.layerName,self.viewName,self.info,\
                self.sliceNum,'redo')

class DrawSplineCommand(QtGui.QUndoCommand):
    
    def __init__(self,parent,layer,knots,layerName,sliceNum):
        QtGui.QUndoCommand.__init__(self)
        super(DrawSplineCommand,self).__init__()
        self.layerName=layerName
        self.sliceNum=sliceNum
        self.viewName='layerViewer'
        self.parentWind=parent
        self.info=None
        self.prevLayer=layer
        self.prevKnots=knots
        self.redoLayer=None
        self.redoKnots=None
    
    def set_redo_values(self,layer,knots):
        self.redoLayer=layer
        self.redoKnots=knots
    def undo(self):
        self.parentWind.draw_spline_command_undo_redo(self.prevLayer,\
                self.prevKnots,self.redoLayer,self.redoKnots,self.layerName,\
                self.viewName,self.info,self.sliceNum,'undo')
        
    def redo(self):
        self.parentWind.oct_controller.set_slice_number(self.sliceNum,self.viewName)
        self.info=self.parentWind.draw_spline_command_undo_redo(self.prevLayer,\
                self.prevKnots,self.redoLayer,self.redoKnots,self.layerName,\
                self.viewName,self.info,self.sliceNum,'redo')
        
class ApplySplitCommand(QtGui.QUndoCommand):
    
    def __init__(self,parent):
        QtGui.QUndoCommand.__init__(self)
        super(ApplySplitCommand,self).__init__()
      
        self.parentWind=parent
        self.info=None
        
    def undo(self):
        self.parentWind.apply_split_command_undo_redo(self.info,'undo')
        
    def redo(self):
        self.info=self.parentWind.apply_split_command_undo_redo(self.info,'redo')

class ExtractDrusenNormalThicknessCommand(QtGui.QUndoCommand):
    def __init__(self,parent,thickness,sliceZ,callerName):
        QtGui.QUndoCommand.__init__(self)
        super(ExtractDrusenNormalThicknessCommand,self).__init__()
        self.thickness=thickness
        self.viewName=callerName
        self.sliceNumZ=sliceZ
        self.parentWind=parent
        self.info=None
        self.drusen=None
        
    def undo(self):
        self.parentWind.extract_drusen_normal_thickness_command_undo_redo(\
            self.thickness,self.viewName,self.drusen,self.sliceNumZ,'undo')
        
    def redo(self):
        self.parentWind.oct_controller.set_slice_number(self.sliceNumZ+1,self.viewName)
        self.drusen=self.parentWind.extract_drusen_normal_thickness_command_undo_redo(\
            self.thickness,self.viewName,self.drusen,self.sliceNumZ,'redo')
                
class DrawPolyFitCommand(QtGui.QUndoCommand):
    
    def __init__(self,parent,image,topLeftX,topLeftY,\
             bottomRightX,bottomRightY,sliceNum,polyDegree,layerName,viewName):
        QtGui.QUndoCommand.__init__(self)
        super(DrawPolyFitCommand,self).__init__()
        self.image=image
        self.topLeftX=topLeftX
        self.topLeftY=topLeftY
        self.bottomRightX=bottomRightX
        self.bottomRightY=bottomRightY
        self.polyDegree=polyDegree
        self.layerName=layerName
        self.viewName=viewName
        self.sliceNum=sliceNum
        self.parentWind=parent
        self.info=None
        
    def undo(self):
        self.parentWind.draw_poly_fit_command_undo_redo(self.image,self.topLeftX,\
                self.topLeftY,self.bottomRightX,self.bottomRightY,self.polyDegree,\
                self.layerName,self.viewName,self.info,self.sliceNum,'undo')
        
    def redo(self):
        self.parentWind.oct_controller.set_slice_number(self.sliceNum,self.viewName)
        self.info=self.parentWind.draw_poly_fit_command_undo_redo(self.image,\
                self.topLeftX,self.topLeftY,self.bottomRightX,self.bottomRightY,\
                self.polyDegree,self.layerName,self.viewName,self.info,self.sliceNum,'redo')
 
 
class DrawDrusenOnEnfaceCommand(QtGui.QUndoCommand):
    
    def __init__(self,parent,x,y,color,normalThickness,callerName):
        QtGui.QUndoCommand.__init__(self)
        super(DrawDrusenOnEnfaceCommand,self).__init__()
        self.x=x
        self.y=y
        self.color=color
        self.viewName=callerName
        self.thickness=normalThickness
        self.parentWind=parent
        self.info=None
        
    def undo(self):
        self.parentWind.draw_drusen_on_enface_command_undo_redo(self.x,self.y,\
        self.color,self.viewName,self.info,self.thickness,'undo')
        
    def redo(self):
        self.info=self.parentWind.draw_drusen_on_enface_command_undo_redo(self.x,self.y,\
        self.color,self.viewName,self.info,self.thickness,'redo')
         
class DrawLineOnDrusenEnface(QtGui.QUndoCommand):
    
    def __init__(self,parent,y,s,color,normalThickness,callerName):
        QtGui.QUndoCommand.__init__(self)
        super(DrawLineOnDrusenEnface,self).__init__()
        self.s=s
        self.y=y
        self.color=color
        self.viewName=callerName
        self.thickness=normalThickness
        self.parentWind=parent
        self.info=None
        
    def undo(self):
        self.parentWind.draw_line_on_enface_command_undo_redo(self.s,self.y,\
        self.color,self.viewName,self.info,self.thickness,'undo')
        
    def redo(self):
        self.info=self.parentWind.draw_line_on_enface_command_undo_redo(self.s,self.y,\
        self.color,self.viewName,self.info,self.thickness,'redo')
         
class AcceptSuggestedSegmentationCommand(QtGui.QUndoCommand):
    
    def __init__(self,parent,sliceNumZ,layerName,smoothness,uncType,extent,csps):
        QtGui.QUndoCommand.__init__(self)
        super(AcceptSuggestedSegmentationCommand,self).__init__()
        self.layerName=layerName
        self.sliceNumZ=sliceNumZ
        self.parentWind=parent
        self.smoothness=smoothness
        self.uncType=uncType
        self.extent=extent
        self.csps=csps
        self.info=None
        
    def undo(self):
        self.parentWind.accept_suggest_seg_command_undo_redo(self.layerName,\
           self.info,self.sliceNumZ,self.smoothness,self.uncType,self.extent,\
           self.csps,'undo')
        
    def redo(self):
        self.parentWind.oct_controller.set_slice_number(self.sliceNumZ,'layerViewer')
        self.info=self.parentWind.accept_suggest_seg_command_undo_redo(\
           self.layerName,self.info,self.sliceNumZ,self.smoothness,self.uncType,\
           self.extent,self.csps,'redo')
                
class DrawPenCommand(QtGui.QUndoCommand):
    
    def __init__(self,parent,x,y,color,viewName,sliceNum,prevValues=[],slices=1,\
                                posY=0,oldValue=[],redoValues=[],layerName=''):
        QtGui.QUndoCommand.__init__(self)
        super(DrawPenCommand,self).__init__()
        self.x=x
        self.y=y
        self.color=color
        self.viewName=viewName
        self.sliceNum=sliceNum
        self.parentWind=parent
        self.prevValues=prevValues
        self.slices=slices
        self.posY=posY
        self.oldValue=oldValue
        self.redoValues=redoValues
        self.info=None
        self.layerName=layerName
        
    def undo(self):
        c=np.copy(self.color)
        
        if(type(self.oldValue)==float):
            c[0]=self.oldValue
            c[1]=self.oldValue
            c[2]=self.oldValue

        else:
            if(self.color[0]==0):
                c[0]=255
                c[1]=255
                c[2]=255
            elif(self.color[0]==255):
                c[0]=0
                c[1]=0
                c[2]=0
                
        if(len(self.redoValues)>0):
            self.parentWind.draw_pen_command_undo_redo(self.x,self.y,c,\
                self.viewName,self.prevValues,self.slices,self.posY,self.sliceNum,\
                'undo',self.info,self.layerName)
            return
        self.parentWind.draw_pen_command_undo_redo(self.x,self.y,c,self.viewName,\
            self.prevValues,self.slices,self.posY,self.sliceNum,'undo',self.info,\
            self.layerName)
        
    def redo(self):
        self.parentWind.oct_controller.set_slice_number(self.sliceNum,self.viewName)
        self.info=self.parentWind.draw_pen_command_undo_redo(self.x,self.y,\
            self.color,self.viewName,self.redoValues,self.slices,self.posY,\
            self.sliceNum,'redo',self.info,self.layerName)

class DrawLineCommand(QtGui.QUndoCommand):
    
    def __init__(self,parent,x1,y1,x2,y2,color,viewName,sliceNum,posX=[],\
            slices=[],posY=[],prevValues=[],redoValues=[],layerName=''):
        QtGui.QUndoCommand.__init__(self)
        super(DrawLineCommand,self).__init__()
        self.x1=x1
        self.y1=y1
        self.x2=x2
        self.y2=y2
        self.color=color
        self.viewName=viewName
        self.sliceNum=sliceNum
        self.parentWind=parent
        self.posX=posX
        self.slices=slices
        self.posY=posY
        self.prevValues=prevValues
        self.redoValues=redoValues
        self.info=None
        self.layerName=layerName
        
    def undo(self):
        self.parentWind.oct_controller.set_slice_number(self.sliceNum,self.viewName)
        c=np.copy(self.color)
        
        if(self.color[0]==0):
            c[0]=255
            c[1]=255
            c[2]=255
        elif(self.color[0]==255):
            c[0]=0
            c[1]=0
            c[2]=0
        self.parentWind.draw_line_command_undo_redo(self.x1,self.y1,self.x2,\
            self.y2,c,self.viewName,self.posX,self.slices,self.posY,self.sliceNum,\
            self.prevValues,'undo',self.info,self.layerName)
        
    def redo(self):
        self.parentWind.oct_controller.set_slice_number(self.sliceNum,self.viewName)
        self.info=self.parentWind.draw_line_command_undo_redo(self.x1,self.y1,\
            self.x2,self.y2,self.color,self.viewName,self.posX,self.slices,\
            self.posY,self.sliceNum,self.redoValues,'redo',self.info,self.layerName)
        
class DrawCurveCommand(QtGui.QUndoCommand):
    
    def __init__(self,parent,s,y,x,color,viewName,sliceNum,prevValues=[],\
            redoValues=[],layerName=''):
        QtGui.QUndoCommand.__init__(self)
        super(DrawCurveCommand,self).__init__()
        
        self.color=color
        self.viewName=viewName
        self.sliceNum=sliceNum
        self.parentWind=parent
        self.posX=x
        self.slices=s
        self.posY=y
        self.prevValues=prevValues
        self.redoValues=redoValues
        self.info=None
        self.layerName=layerName
        
    def undo(self):
        c=np.copy(self.color)
        
        if(self.color[0]==0):
            c[0]=255
            c[1]=255
            c[2]=255
        elif(self.color[0]==255):
            c[0]=0
            c[1]=0
            c[2]=0
        self.parentWind.draw_curve_command_undo_redo(self.posX[::-1],\
            self.slices[::-1],self.posY[::-1],c,self.viewName,self.sliceNum,\
            self.prevValues[::-1],'undo',self.info,self.layerName)
        
    def redo(self):
        
        self.parentWind.oct_controller.set_slice_number(self.sliceNum,self.viewName)
        self.info=self.parentWind.draw_curve_command_undo_redo(self.posX,\
            self.slices,self.posY,self.color,self.viewName,self.sliceNum,\
            self.redoValues,'redo',self.info,self.layerName)        
        
class DrawFillCommand(QtGui.QUndoCommand):
    
    def __init__(self,parent,xs,ys,color,viewName,sliceNum,posX=[]):
        QtGui.QUndoCommand.__init__(self)
        super(DrawFillCommand,self).__init__()
        self.xs=xs
        self.ys=ys
        self.color=color
        self.viewName=viewName
        self.sliceNum=sliceNum
        self.parentWind=parent
        self.posX=posX
    def undo(self):
        c=np.copy(self.color)
        if(self.color[0]==0):
            c[0]=255
            c[1]=255
            c[2]=255
        elif(self.color[0]==255):
            c[0]=0
            c[1]=0
            c[2]=0
        self.parentWind.draw_fill_command_undo_redo(self.xs,self.ys,c,\
            self.viewName,self.sliceNum,self.posX)
        
    def redo(self):
        self.parentWind.oct_controller.set_slice_number(self.sliceNum,self.viewName)
        self.parentWind.draw_fill_command_undo_redo(self.xs,self.ys,self.color,\
            self.viewName,self.sliceNum,self.posX)
                
class DrawDilateCommand(QtGui.QUndoCommand):
    
    def __init__(self,parent,xs,ys,color,viewName,sliceNum):
        QtGui.QUndoCommand.__init__(self)
        super(DrawDilateCommand,self).__init__()
        self.xs=xs
        self.ys=ys
        self.color=color
        self.viewName=viewName
        self.sliceNum=sliceNum
        self.parentWind=parent
        
    def undo(self):
        c=np.copy(self.color)
        if(self.color[0]==0):
            c[0]=255
            c[1]=255
            c[2]=255
        elif(self.color[0]==255):
            c[0]=0
            c[1]=0
            c[2]=0
        self.parentWind.draw_dilate_command_undo_redo(self.xs,self.ys,c,\
            self.viewName,self.sliceNum)
        
    def redo(self):
        self.parentWind.oct_controller.set_slice_number(self.sliceNum,self.viewName)
        self.parentWind.draw_dilate_command_undo_redo(self.xs,self.ys,self.color,\
            self.viewName,self.sliceNum)

class DrawErosionCommand(QtGui.QUndoCommand):
    
    def __init__(self,parent,xs,ys,color,viewName,sliceNum,posX):
        QtGui.QUndoCommand.__init__(self)
        super(DrawErosionCommand,self).__init__()
        self.xs=xs
        self.ys=ys
        self.color=color
        self.viewName=viewName
        self.sliceNum=sliceNum
        self.parentWind=parent
        self.posX=posX
    def undo(self):
        c=np.copy(self.color)
        if(self.color[0]==0):
            c[0]=255
            c[1]=255
            c[2]=255
        elif(self.color[0]==255):
            c[0]=0
            c[1]=0
            c[2]=0
        self.parentWind.draw_erosion_command_undo_redo(self.xs,self.ys,c,\
            self.viewName,self.sliceNum,self.posX)
        
    def redo(self):
        self.parentWind.oct_controller.set_slice_number(self.sliceNum,self.viewName)
        self.parentWind.draw_erosion_command_undo_redo(self.xs,self.ys,\
            self.color,self.viewName,self.sliceNum,self.posX)
        
class DrawFilterCommand(QtGui.QUndoCommand):
    
    def __init__(self,parent,xs,ys,zs,color,viewName,sliceNum):
        QtGui.QUndoCommand.__init__(self)
        super(DrawFilterCommand,self).__init__()
        self.xs=xs
        self.ys=ys
        self.zs=zs
        self.color=color
        self.viewName=viewName
        self.sliceNum=sliceNum
        self.parentWind=parent
    
    def undo(self):
        c=np.copy(self.color)
        if(self.color[0]==0):
            c[0]=255
            c[1]=255
            c[2]=255
        elif(self.color[0]==255):
            c[0]=0
            c[1]=0
            c[2]=0
        self.parentWind.draw_filter_command_undo_redo(self.xs,self.ys,self.zs,\
            c,self.viewName,self.sliceNum)
        
    def redo(self):
        self.parentWind.oct_controller.set_slice_number(self.sliceNum,self.viewName)
        self.parentWind.draw_filter_command_undo_redo(self.xs,self.ys,self.zs,\
            self.color,self.viewName,self.sliceNum)

class DrawDeleteCommand(QtGui.QUndoCommand):
    
    def __init__(self,parent,xs,ys,zs,color,viewName,sliceNum):
        QtGui.QUndoCommand.__init__(self)
        super(DrawDeleteCommand,self).__init__()
        self.xs=xs
        self.ys=ys
        self.zs=zs
        self.color=color
        self.viewName=viewName
        self.sliceNum=sliceNum
        self.parentWind=parent
    
    def undo(self):
        c=np.copy(self.color)
        if(self.color[0]==0):
            c[0]=255
            c[1]=255
            c[2]=255
        elif(self.color[0]==255):
            c[0]=0
            c[1]=0
            c[2]=0
        self.parentWind.draw_delete_command_undo_redo(self.xs,self.ys,self.zs,\
            c,self.viewName,self.sliceNum)
        
    def redo(self):
        self.parentWind.oct_controller.set_slice_number(self.sliceNum,self.viewName)
        self.parentWind.draw_delete_command_undo_redo(self.xs,self.ys,self.zs,\
            self.color,self.viewName,self.sliceNum) 
       
class DrawExtractCommand(QtGui.QUndoCommand):
    
    def __init__(self,parent,xs,ys,zs,xns,yns,zns,color,viewName,sliceNum):
        QtGui.QUndoCommand.__init__(self)
        super(DrawExtractCommand,self).__init__()
        self.xs=xs
        self.ys=ys
        self.zs=zs
        self.xns=xns
        self.yns=yns
        self.zns=zns
        self.color=color
        self.viewName=viewName
        self.sliceNum=sliceNum
        self.parentWind=parent
    
    def undo(self):
        c=np.copy(self.color)
        if(self.color[0]==0):
            c[0]=255
            c[2]=255
        elif(self.color[0]==255):
            c[0]=0
            c[2]=0
        if(self.color[1]==0):
            c[1]=255
        elif(self.color[1]==255):
            c[1]=0

        self.parentWind.draw_extract_command_undo_redo(self.xs,self.ys,\
            self.zs,self.xns,self.yns,self.zns,c,self.viewName,self.sliceNum)
        
    def redo(self):
        self.parentWind.oct_controller.set_slice_number(self.sliceNum,self.viewName)
        self.parentWind.draw_extract_command_undo_redo(self.xs,self.ys,self.zs,\
            self.xns,self.yns,self.zns,self.color,self.viewName,self.sliceNum) 

class DrawRegionCommand(QtGui.QUndoCommand):
    
    def __init__(self,parent,xs,ys,color,viewName,sliceNum,gaType):
        QtGui.QUndoCommand.__init__(self)
        super(DrawRegionCommand,self).__init__()
        self.xs=xs
        self.ys=ys
        self.color=color
        self.viewName=viewName
        self.sliceNum=sliceNum
        self.parentWind=parent
        self.gaType=gaType
    def undo(self):
        c=np.copy(self.color)
        if(self.color[0]==0):
            c[0]=255
            c[1]=255
            c[2]=255
        elif(self.color[0]==255):
            c[0]=0
            c[1]=0
            c[2]=0
        self.parentWind.draw_region_command_undo_redo(self.xs,self.ys,c,\
            self.viewName,self.sliceNum,self.gaType)
        
    def redo(self):
        self.parentWind.draw_region_command_undo_redo(self.xs,self.ys,\
            self.color,self.viewName,self.sliceNum,self.gaType)
        
class DrawBoxCommand(QtGui.QUndoCommand):
    
    def __init__(self,parent,rect,sliceNum,viewName):
        QtGui.QUndoCommand.__init__(self)
        super(DrawBoxCommand,self).__init__()
        self.rect=rect
        self.viewName=viewName
        self.sliceNum=sliceNum
        self.parentWind=parent

    def undo(self):
        self.parentWind.draw_box_command_undo(self.rect,self.sliceNum,\
            self.viewName)
        
    def redo(self):
        self.parentWind.draw_box_command_redo(self.rect,self.sliceNum,\
            self.viewName)        


class RemoveBoxCommand(QtGui.QUndoCommand):
    
    def __init__(self,parent,removedBoxes,sliceNum,viewName):
        QtGui.QUndoCommand.__init__(self)
        super(RemoveBoxCommand,self).__init__()
        self.removedBoxes=removedBoxes
        self.viewName=viewName
        self.sliceNum=sliceNum
        self.parentWind=parent

    def undo(self):
        self.parentWind.remove_box_command_undo(self.removedBoxes,\
            self.sliceNum,self.viewName)
        
    def redo(self):
        self.parentWind.remove_box_command_redo(self.removedBoxes,\
            self.sliceNum,self.viewName)         
#==============================================================================
# End of command classes       
#==============================================================================
            
            
            
            
#==============================================================================
# Main window of OCT Editor            
#==============================================================================
class Ui_MainWindow(object):
    
    def __init__(self):
        self.oct_controller=None
        
        self.subwindowScanViewerUI=None
        self.subwindowLayerViewerUI=None
        self.subwindowDrusenViewerUI=None
        self.subwindowEnfaceViewerUI=None
        self.subwindowHRFViewerUI=None
        self.subwindowGAViewerUI=None
        self.subwindowEnfaceDrusenViewerUI=None
        self.subwindowToolBoxUI=None
        self.uiDrusenInfoTable=None
        
        self.mdiSubwindowToolBox=None
        self.mdiSubwindowScanViewer=None
        self.mdiSubwindowLayerViewer=None
        self.mdiSubwindowDrusenViewer=None
        self.mdiSubwindowHRFViewer=None
        self.mdiSubwindowGAViewer=None
        self.mdiSubwindowEnfaceViewer=None
        self.mdiSubwindowEnfaceDrusenViewer=None
        
        self.morphLevel=0
        self.filterTAll=0
        self.filterTMax=0
        self.polyDegreeValue=0
        self.slittingThreshold=0
        self.smoothness=0
        
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(800, 965)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view","icons",\
            "icons","empty.png"))), QtGui.QIcon.Normal,\
            QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.mdiArea = QtGui.QMdiArea(self.centralwidget)
        self.mdiArea.setObjectName(_fromUtf8("mdiArea"))
        
        self.mainWindow=MainWindow
        
        # Undo stack
        self.undoStack = QtGui.QUndoStack(MainWindow)
        
        self.gridLayout.addWidget(self.mdiArea, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 27))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menu_File = QtGui.QMenu(self.menubar)
        self.menu_File.setObjectName(_fromUtf8("menu_File"))
        self.menu_Edit = QtGui.QMenu(self.menubar)
        self.menu_Edit.setObjectName(_fromUtf8("menu_Edit"))
        self.menu_View = QtGui.QMenu(self.menubar)
        self.menu_View.setObjectName(_fromUtf8("menu_View"))
        self.menu_Window = QtGui.QMenu(self.menubar)
        self.menu_Window.setObjectName(_fromUtf8("menu_Window"))
        self.menu_Help = QtGui.QMenu(self.menubar)
        self.menu_Help.setObjectName(_fromUtf8("menu_Help"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.toolBarBasic = QtGui.QToolBar(MainWindow)
        self.toolBarBasic.setObjectName(_fromUtf8("toolBarBasic"))
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBarBasic)
        self.toolBarOCTperdiction = QtGui.QToolBar(MainWindow)
        self.toolBarOCTperdiction.setObjectName(_fromUtf8("toolBarOCTperdiction"))
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBarOCTperdiction)
        self.toolBarExtraAnalysis = QtGui.QToolBar(MainWindow)
        self.toolBarExtraAnalysis.setObjectName(_fromUtf8("toolBarExtraAnalysis"))
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBarExtraAnalysis)
        
        self.action_Open = QtGui.QAction(MainWindow)
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","folderrr.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_Open.setIcon(icon10)
        self.action_Open.setObjectName(_fromUtf8("action_Open"))
        
        
        self.action_View_Annot = QtGui.QAction(MainWindow)
        self.action_View_Annot.setObjectName(_fromUtf8("action_View_Annot"))
        
        
        self.action_Save = QtGui.QAction(MainWindow)
        icon11 = QtGui.QIcon()
        icon11.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","saveFloppy.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_Save.setIcon(icon11)
        self.action_Save.setObjectName(_fromUtf8("action_Save"))
        
        self.action_Save_As = QtGui.QAction(MainWindow)
        icon20 = QtGui.QIcon()
        icon20.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","saveAsFloppy.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_Save_As.setIcon(icon20)
        self.action_Save_As.setObjectName(_fromUtf8("action_Save_As"))
        
        self.action_Toolbox = QtGui.QAction(MainWindow)
        icon12 = QtGui.QIcon()
        icon12.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","tools.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_Toolbox.setIcon(icon12)
        self.action_Toolbox.setObjectName(_fromUtf8("action_Toolbox"))
        self.action_Reset = QtGui.QAction(MainWindow)
        self.action_Reset.setObjectName(_fromUtf8("action_Reset"))
        
        self.actionShowBscans = QtGui.QAction(MainWindow)
        icon19 = QtGui.QIcon()
        icon19.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","bscan.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionShowBscans.setIcon(icon19)
        self.actionShowBscans.setObjectName(_fromUtf8("actionFindLayers"))
        
        self.actionFindLayers = QtGui.QAction(MainWindow)
        icon13 = QtGui.QIcon()
        icon13.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","layers.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionFindLayers.setIcon(icon13)
        self.actionFindLayers.setObjectName(_fromUtf8("actionFindLayers"))
        
        
        self.actionFindDrusen = QtGui.QAction(MainWindow)
        icon14 = QtGui.QIcon()
        icon14.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","druseSeg.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionFindDrusen.setIcon(icon14)
        self.actionFindDrusen.setObjectName(_fromUtf8("actionFindDrusen"))
        self.actionShowEnface = QtGui.QAction(MainWindow)
        icon15 = QtGui.QIcon()
        icon15.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","enfaceProj.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionShowEnface.setIcon(icon15)
        self.actionShowEnface.setObjectName(_fromUtf8("actionShowEnface"))
        
        
        
        self.actionShowEnfaceDrusen = QtGui.QAction(MainWindow)
        icon16 = QtGui.QIcon()
        icon16.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","enfaceDru.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionShowEnfaceDrusen.setIcon(icon16)
        self.actionShowEnfaceDrusen.setObjectName(_fromUtf8("actionShowEnfaceDrusen"))
        
#==============================================================================
#         HRF and GA Annotation windows
#==============================================================================
        
        self.actionShowHRF = QtGui.QAction(MainWindow)
        icon23 = QtGui.QIcon()
        icon23.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","hrf.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionShowHRF.setIcon(icon23)
        self.actionShowHRF.setObjectName(_fromUtf8("actionShowHRF"))
        
        self.actionShowGA = QtGui.QAction(MainWindow)
        icon24 = QtGui.QIcon()
        icon24.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","GA.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionShowGA.setIcon(icon24)
        self.actionShowGA.setObjectName(_fromUtf8("actionShowGA"))
        
        
        self.actionShow3D = QtGui.QAction(MainWindow)
        icon17 = QtGui.QIcon()
        icon17.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","3DView.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionShow3D.setIcon(icon17)
        self.actionShow3D.setObjectName(_fromUtf8("actionShow3D"))
        
        
        
        self.actionMeasureDrusen = QtGui.QAction(MainWindow)
        icon18 = QtGui.QIcon()
        icon18.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","druAnalysis.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionMeasureDrusen.setIcon(icon18)
        self.actionMeasureDrusen.setObjectName(_fromUtf8("actionMeasureDrusen"))
        
        self.actionUndo = QtGui.QAction(MainWindow)
        icon21 = QtGui.QIcon()
        icon21.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","undo.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionUndo.setIcon(icon21)
        self.actionUndo.setObjectName(_fromUtf8("undo"))
        
        self.actionRedo = QtGui.QAction(MainWindow)
        icon22 = QtGui.QIcon()
        icon22.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join(sfwPath,"view",\
            "icons","icons","redo.png"))),\
            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionRedo.setIcon(icon22)
        self.actionRedo.setObjectName(_fromUtf8("redo"))
        
        self.action_ShowSettings = QtGui.QAction(MainWindow)
        self.action_ShowSettings.setObjectName(_fromUtf8("action_ShowSetting"))
               
        self.action_ShowUncertaintyColorMapEntProb=\
            QtGui.QAction(MainWindow,checkable=True)
        self.action_ShowUncertaintyColorMapEntProb.\
            setObjectName(_fromUtf8("ShowUncertaintyColorMapEntProb"))
        
        self.action_ChangeDruseVisitingOrderSize=QtGui.QAction(MainWindow,checkable=True)
        self.action_ChangeDruseVisitingOrderSize.setObjectName(\
            _fromUtf8("ChangeDruseVisitingOrderSize"))
        self.action_ChangeDruseVisitingOrderHeight=QtGui.QAction(MainWindow,checkable=True)
        self.action_ChangeDruseVisitingOrderHeight.setObjectName(\
            _fromUtf8("ChangeDruseVisitingOrderHeight"))
        self.action_ChangeDruseVisitingOrderBrightness=QtGui.QAction(MainWindow,checkable=True)
        self.action_ChangeDruseVisitingOrderBrightness.setObjectName(\
            _fromUtf8("ChangeDruseVisitingOrderBrightness"))
    
        self.action_ChangeDruseVisitingOrderHeight.setChecked(True)
        
        self.menu_File.addAction(self.action_Open)
        self.menu_File.addAction(self.action_Save)
        self.menu_File.addAction(self.action_Save_As)
        self.menu_File.addAction(self.action_ShowSettings)
        self.menu_View.addAction(self.action_Toolbox)
        
        self.uncertaintyMapMenu = self.menu_View.addMenu('Segmentation uncertainty')
        self.uncertaintyMapMenu.addAction(self.action_ShowUncertaintyColorMapEntProb)
        
        self.enfaceDrusenMenu = self.menu_View.addMenu('Druse visiting order')
        self.enfaceDrusenMenu.addAction(self.action_ChangeDruseVisitingOrderSize)
        self.enfaceDrusenMenu.addAction(self.action_ChangeDruseVisitingOrderHeight)
        self.enfaceDrusenMenu.addAction(self.action_ChangeDruseVisitingOrderBrightness)

        self.uncertaintyMapMenu.setEnabled(False)   
        self.enfaceDrusenMenu.setEnabled(False)
        
        
        self.menu_Window.addAction(self.action_Reset)
        self.menubar.addAction(self.menu_File.menuAction())
        self.menubar.addAction(self.menu_Edit.menuAction())
        self.menubar.addAction(self.menu_View.menuAction())
        self.menubar.addAction(self.menu_Window.menuAction())
        self.menubar.addAction(self.menu_Help.menuAction())
        self.toolBarBasic.addAction(self.action_Open)
        self.toolBarBasic.addAction(self.action_Save)
        self.toolBarBasic.addAction(self.action_Save_As)
        self.toolBarOCTperdiction.addAction(self.actionShowBscans)
        self.toolBarOCTperdiction.addAction(self.actionFindLayers)
        self.toolBarOCTperdiction.addAction(self.actionFindDrusen)
        self.toolBarOCTperdiction.addAction(self.actionShowEnface)
        self.toolBarOCTperdiction.addAction(self.actionShowEnfaceDrusen)
        self.toolBarOCTperdiction.addAction(self.actionShowHRF)
        self.toolBarOCTperdiction.addAction(self.actionShowGA)
        self.toolBarOCTperdiction.addAction(self.action_Toolbox)
        self.toolBarExtraAnalysis.addAction(self.actionUndo)
        self.toolBarExtraAnalysis.addAction(self.actionRedo)
        self.toolBarExtraAnalysis.addAction(self.actionMeasureDrusen)
        
        self.actionUndo.setToolTip("Undo")
        self.actionRedo.setToolTip("Redo")
        
#==============================================================================
#       Connecting actions
#==============================================================================
        self.action_Open.triggered.connect(self.open_action)
        self.action_Save.triggered.connect(self.save_action)
        self.action_Save_As.triggered.connect(self.save_as_action)
        self.action_ShowSettings.triggered.connect(self.show_settings_action)
        self.action_Toolbox.triggered.connect(self.toolbox_action)
        self.actionShowBscans.triggered.connect(self.show_bscans_action)
        self.actionFindLayers.triggered.connect(self.find_layers_action)
        self.actionFindDrusen.triggered.connect(self.find_drusen_action)
        self.actionShowEnface.triggered.connect(self.show_enface_action)
        self.actionShowHRF.triggered.connect(self.find_HRF_action)
        self.actionShowGA.triggered.connect(self.find_GA_action)
        self.actionShowEnfaceDrusen.triggered.connect(self.show_enface_drusen_action)
        self.actionShow3D.triggered.connect(self.show_3d_action)
        self.actionMeasureDrusen.triggered.connect(self.measure_drusen_action)
        self.action_ShowUncertaintyColorMapEntProb.triggered.connect(\
            self.show_uncertainty_color_map_ent_prob)
        
        self.action_ChangeDruseVisitingOrderSize.triggered.connect(\
            self.change_druse_visiting_order_size)
        self.action_ChangeDruseVisitingOrderHeight.triggered.connect(\
            self.change_druse_visiting_order_height)
        self.action_ChangeDruseVisitingOrderBrightness.triggered.connect(\
            self.change_druse_visiting_order_brightness)
        
        self.actionUndo.triggered.connect(self.action_undo)
        self.actionRedo.triggered.connect(self.action_redo)
        
        self.action_Open.setShortcut(QtGui.QKeySequence.Open)
        self.action_Save.setShortcut(QtGui.QKeySequence.Save)
        self.action_Save_As.setShortcut(QtGui.QKeySequence.SaveAs)
        self.actionUndo.setShortcut(QtGui.QKeySequence.Undo)
        self.actionRedo.setShortcut(QtGui.QKeySequence.Redo)
        self.action_View_Annot.setShortcut(QtGui.QKeySequence.New)

        self.mdiArea.subWindowActivated.connect(self.subwindow_activated)
        
#==============================================================================
#         Progress bar
#==============================================================================
        self.toolBoxHidden=True
        self.progressBarShowed=False
        self.disable_all()
                
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        self.progressBar = QtGui.QWidget()
        self.progressBarUI = pbv.Ui_FormProgressBar()
        self.progressBarUI.setupUi(self.progressBar)
        self.hide_progress_bar()
        
        self.settingWindow=QtGui.QWidget()
        self.settingWidnowUI=settings.Ui_settings()
        self.settingWidnowUI.setupUi(self.settingWindow,self.oct_controller)
        self.settingWindow.hide()
        
        self.lastSplineCommand=None
    
    def enable_probability_related_tools(self):
        self.uncertaintyMapMenu.setEnabled(True)
        self.subwindowToolBoxUI.enable_probability_related_tools()
        
    def disable_probability_related_tools(self):
        self.uncertaintyMapMenu.setEnabled(False) 
        self.subwindowToolBoxUI.disable_probability_related_tools()
    def set_uncertainties_per_bscan(self,ent,prob,entCol,probCol):
        if(not self.subwindowLayerViewerUI is None):
            self.subwindowLayerViewerUI.graphicsViewImageViewer.\
                set_uncertainties(ent,prob,entCol,probCol)
        if(not self.subwindowEnfaceViewerUI is None):
            self.subwindowEnfaceViewerUI.graphicsViewImageViewer.\
                set_uncertainties(ent,prob,entCol,probCol)    
    def show_uncertainty_color_map_ent_prob(self):
        if(not self.subwindowLayerViewerUI is None):
            self.oct_controller.write_in_log(self.oct_controller.get_time()+\
                ','+self.action_ShowUncertaintyColorMapEntProb.objectName()+'\n')
            self.subwindowLayerViewerUI.triggerUncertaintyMap('probability','entropy')
            self.oct_controller.update_layer_viewer('layerViewer')
    def change_druse_visiting_order_size(self):
        self.action_ChangeDruseVisitingOrderHeight.setChecked(False)
        self.action_ChangeDruseVisitingOrderBrightness.setChecked(False)
        if(not self.subwindowLayerViewerUI is None):
            self.oct_controller.write_in_log(self.oct_controller.get_time()+\
                ','+self.action_ChangeDruseVisitingOrderSize.objectName()+'\n')
            self.oct_controller.change_drusen_visiting_order('Size')
            
    def change_druse_visiting_order_height(self):
        self.action_ChangeDruseVisitingOrderSize.setChecked(False)
        self.action_ChangeDruseVisitingOrderBrightness.setChecked(False)
        if(not self.subwindowLayerViewerUI is None):
            self.oct_controller.write_in_log(self.oct_controller.get_time()+\
                ','+self.action_ChangeDruseVisitingOrderHeight.objectName()+'\n')
            self.oct_controller.change_drusen_visiting_order('Height')
            
    def change_druse_visiting_order_brightness(self):
        self.action_ChangeDruseVisitingOrderSize.setChecked(False)
        self.action_ChangeDruseVisitingOrderHeight.setChecked(False)
        if(not self.subwindowLayerViewerUI is None):
            self.oct_controller.write_in_log(self.oct_controller.get_time()+\
                ','+self.action_ChangeDruseVisitingOrderBrightness.objectName()+'\n')
            self.oct_controller.change_drusen_visiting_order('Brightness')
            
    def subwindow_activated(self):
        activateWindow=self.mdiArea.activeSubWindow()
        if((not activateWindow is  None) and not self.subwindowToolBoxUI is  None):
            windowName=activateWindow.widget().get_name()
            if(windowName=='toolBox'):
                pass
            elif(windowName=='scanViewer'):
                pass
            elif(windowName=='layerViewer'):
                # Set editing 
                if(self.oct_controller.get_edit_rpe()):
                    self.subwindowToolBoxUI.rpe_editing_selected()
                elif(self.oct_controller.get_edit_bm()):
                    self.subwindowToolBoxUI.bm_editing_selected()
                else:
                    self.subwindowToolBoxUI.rpe_editing_selected()
            elif(windowName=='drusenViewer'):
                # Set editing 
                self.subwindowToolBoxUI.drusen_editing_selected()
            elif(windowName=='enfaceViewer'):
                # Set editing 
                self.subwindowToolBoxUI.enface_editing_selected()
            elif(windowName=='enfaceDrusenViewer'):
                # Set editing 
                self.subwindowToolBoxUI.enface_drusen_editing_selected()
            elif(windowName=='hrfViewer'):
                # Set editing 
                self.subwindowToolBoxUI.hrf_editing_selected()
            elif(windowName=='gaViewer'):
                # Set editing 
                if(self.oct_controller.get_edit_ga()):
                    self.subwindowToolBoxUI.ga_editing_selected()
                elif(self.oct_controller.get_edit_nga()):
                    self.subwindowToolBoxUI.nga_editing_selected()
                else:
                    self.subwindowToolBoxUI.ga_editing_selected()
    
    def accept_suggest_seg_command_undo_redo(self,layerName,\
                    info,sliceNumZ,smoothness,uncType,extent,csps,actionMode):
        if(self.subwindowLayerViewerUI is not None):
                if(actionMode=='redo'):
                    infoL=self.oct_controller.accept_suggest_seg_command_redo(\
                        layerName,sliceNumZ,'layerViewer',smoothness,uncType,extent,csps)
                    self.oct_controller.set_slice_edited(sliceNumZ+1,layerName,True)
                    self.oct_controller.slice_value_changed(sliceNumZ+1,\
                        'layerViewer',furtherUpdate=False)
                    return infoL
                elif(actionMode=='undo'):
                    self.oct_controller.accept_suggest_seg_command_undo(layerName,sliceNumZ,info,\
                        'layerViewer',smoothness,uncType,extent,csps)   
                    self.oct_controller.slice_value_changed(sliceNumZ+1,\
                        'layerViewer',furtherUpdate=False)
    
    def extract_drusen_normal_thickness_command_undo_redo(self,thickness,\
            viewName,drusen,sliceNumZ,actionMode):
               
            if(actionMode=='redo'):
                drusen=self.oct_controller.extract_drunsen_using_normal_thickness_redo(\
                    thickness,viewName,sliceNumZ)
                self.oct_controller.slice_value_changed(sliceNumZ+1,\
                    viewName,furtherUpdate=False)
                return drusen
            elif(actionMode=='undo'):
                self.oct_controller.extract_drunsen_using_normal_thickness_undo(\
                    thickness,viewName,sliceNumZ,drusen)   
                self.oct_controller.slice_value_changed(sliceNumZ+1,\
                    'drusenViewer',furtherUpdate=False)
                    
    def draw_poly_fit_command_undo_redo(self,image,topLeftX,topLeftY,bottomRightX,\
                bottomRightY,polyDegree,layerName,viewName,info,sliceNum,actionMode):

        if( viewName=='layerViewer'):
            if(self.subwindowLayerViewerUI is not None):
                if(actionMode=='redo'):
                    infoL=self.oct_controller.poly_fit_redo(image,topLeftX,\
                        topLeftY,bottomRightX,bottomRightY,polyDegree,\
                        layerName,sliceNum,viewName)
                    self.oct_controller.set_slice_edited(sliceNum,layerName,True)
                    self.oct_controller.slice_value_changed(sliceNum,\
                        'layerViewer',furtherUpdate=False)
                    return infoL
                elif(actionMode=='undo'):
                    self.oct_controller.poly_fit_undo(layerName,sliceNum,info,\
                        viewName)   
                    self.oct_controller.slice_value_changed(sliceNum,\
                        'layerViewer',furtherUpdate=False)
                        
    def draw_drusen_on_enface_command_undo_redo(self,x,y,\
        color,viewName,info,thickness,actionMode):
            if(viewName=="enfaceDrusenViewer"):
                if(actionMode=="undo"):
                    self.oct_controller.draw_drusen_on_enface_undo(x,y,color,thickness,info)
                    self.oct_controller.slice_value_changed(y+1,\
                        'enfaceDrusenViewer',furtherUpdate=False)
                    self.oct_controller.slice_value_changed(y+1,\
                        'drusenViewer',furtherUpdate=False)   
                elif(actionMode=="redo"):
                    infoL=self.oct_controller.draw_drusen_on_enface_redo(x,y,color,thickness)
                    self.oct_controller.slice_value_changed(y+1,\
                        'enfaceDrusenViewer',furtherUpdate=False)
                    self.oct_controller.slice_value_changed(y+1,\
                        'drusenViewer',furtherUpdate=False)   
                    return infoL
        
    
    def draw_line_on_enface_command_undo_redo(self,s,y,\
        color,viewName,info,thickness,actionMode):
            if(viewName=="enfaceDrusenViewer"):
                if(actionMode=="undo"):
                    self.oct_controller.draw_line_on_enface_undo(s,y,color,thickness,info)
                    self.oct_controller.slice_value_changed(s[0]+1,\
                        'enfaceDrusenViewer',furtherUpdate=False)
                    self.oct_controller.slice_value_changed(s[0]+1,\
                        'drusenViewer',furtherUpdate=False)   
                elif(actionMode=="redo"):
                    infoL=self.oct_controller.draw_line_on_enface_redo(s,y,color,thickness)
                    self.oct_controller.slice_value_changed(s[0]+1,\
                        'enfaceDrusenViewer',furtherUpdate=False)
                    self.oct_controller.slice_value_changed(s[0]+1,\
                        'drusenViewer',furtherUpdate=False)   
                    return infoL
    
    def draw_spline_command_undo_redo(self,prevLayer,prevKnots,redoLayer,redoKnots,layerName,\
                viewName,info,sliceNum,actionMode):
        if( viewName=='layerViewer'):
            if(self.subwindowLayerViewerUI is not None):
                if(actionMode=='redo'):
                    infoL=self.oct_controller.update_layer_spline_redo(redoLayer,redoKnots,layerName,sliceNum)
                    self.oct_controller.set_slice_edited(sliceNum,layerName,True)
                    self.oct_controller.slice_value_changed(sliceNum,\
                        'layerViewer',furtherUpdate=False)
                    return infoL
                elif(actionMode=='undo'):
                    self.oct_controller.update_layer_spline_undo(prevLayer,prevKnots,info)   
                    self.oct_controller.slice_value_changed(sliceNum,\
                        'layerViewer',furtherUpdate=False)
                
    def apply_split_command_undo_redo(self,info,actionMode):
        if(actionMode=='redo'):   
             infoL=self.oct_controller.apply_split_redo()
             return infoL
        elif(actionMode=='undo'):
             self.oct_controller.apply_split_undo(info)
    
    def get_smoothness(self):
        if(not self.subwindowLayerViewerUI is None):
            return self.subwindowLayerViewerUI.graphicsViewImageViewer.get_smoothness()
        else:
            return 1
            
    def draw_cost_point_command_undo_redo(self,x,y,smoothness,layerName,\
            viewName,info,sliceNum,actionMode):

        if( viewName=='layerViewer'):
            if(self.subwindowLayerViewerUI is not None):
                if(actionMode=='redo'):
                    infoL=self.oct_controller.update_cost_image_redo(x,y,\
                        smoothness,layerName,sliceNum,viewName)
                    self.oct_controller.set_slice_edited(sliceNum,layerName,True)
                    self.oct_controller.slice_value_changed(sliceNum,\
                        'layerViewer',furtherUpdate=False)   
                    return infoL
                elif(actionMode=='undo'):
                    self.oct_controller.update_cost_image_undo(layerName,\
                        sliceNum,info,viewName)
                    self.oct_controller.slice_value_changed(sliceNum,\
                        'layerViewer',furtherUpdate=False)   
                        
    def draw_pen_command_undo_redo(self,x,y,color,viewName,prevValues=[],\
            slices=[],posY=[],sliceNum=1,mode='undo',info=None,layerName=''):
        if(viewName=='drusenViewer' ):
            if(self.subwindowDrusenViewerUI is not None):
                self.oct_controller.oct.instert_druse_at_slice(sliceNum,[y],\
                    [x],color[0])
                self.oct_controller.slice_value_changed(sliceNum,'drusenViewer',\
                    furtherUpdate=False)
        if( viewName=='layerViewer'):
            if(self.subwindowLayerViewerUI is not None):
                self.oct_controller.oct.instert_layer_at_slice(sliceNum,y,x,prevValues)
                self.oct_controller.slice_value_changed(sliceNum,'layerViewer',\
                    furtherUpdate=False)
                
                if(mode=='undo'):
                    self.oct_controller.pen_or_line_undo(sliceNum,info,layerName)
                    self.oct_controller.slice_value_changed(sliceNum,'layerViewer',\
                        furtherUpdate=False)
                elif(mode=='redo'):
                    info=self.oct_controller.pen_or_line_redo(sliceNum,layerName)
                    self.oct_controller.set_slice_edited(sliceNum,layerName,True)
                    self.oct_controller.slice_value_changed(sliceNum,'layerViewer',\
                        furtherUpdate=False)
                    return info
                
        if(viewName=='hrfViewer' ):
            if(self.subwindowHRFViewerUI is not None):
                self.oct_controller.oct.insert_hrf_at_slice(sliceNum,[y],[x],color[0])
                self.oct_controller.slice_value_changed(sliceNum,'hrfViewer',\
                    furtherUpdate=False)
                self.oct_controller.update_HRF_status(sliceNum-1,\
                    self.oct_controller.oct.hrf_exist_in_slice(sliceNum),True)
                
        if(viewName=='enfaceDrusenViewer'):
            if(self.subwindowEnfaceDrusenViewerUI is not None):
                if(color[0]==255.):
                    self.oct_controller.oct.insert_druse_at(slices,posY,prevValues)
                    self.oct_controller.slice_value_changed(sliceNum,\
                        'enfaceDrusenViewer',furtherUpdate=False)
                else:
                    self.subwindowEnfaceDrusenViewerUI.graphicsViewImageViewer.\
                        imagePanel.draw_point_undo_redo(x,y,color)
            if(self.subwindowDrusenViewerUI is not None):
                    self.oct_controller.slice_value_changed(sliceNum,\
                        'drusenViewer',furtherUpdate=False)
                    
    def draw_line_command_undo_redo(self,x1,y1,x2,y2,color,viewName,posX=[],\
            slices=[],posY=[],sliceNum=1,prevValues=[],mode='undo',info=None,\
            layerName=''):
        if(viewName=='hrfViewer' ):
            if(self.subwindowHRFViewerUI is not None):
                self.oct_controller.oct.insert_hrf_at_slice(sliceNum,slices,\
                    posY,color[0])
                self.oct_controller.slice_value_changed(sliceNum,'hrfViewer',\
                    furtherUpdate=False)
                self.oct_controller.update_HRF_status(sliceNum-1,\
                    self.oct_controller.oct.hrf_exist_in_slice(sliceNum),True)
                
        if(viewName=='drusenViewer' ):
            if(self.subwindowDrusenViewerUI is not None):
                self.oct_controller.oct.instert_druse_at_slice(sliceNum,slices,\
                    posY,color[0])
                self.oct_controller.slice_value_changed(sliceNum,'drusenViewer',\
                    furtherUpdate=False)
                
        if(viewName=='layerViewer'):
            if(self.subwindowLayerViewerUI is not None):
                self.oct_controller.oct.instert_layer_at_slice(sliceNum,slices,\
                    posY,prevValues)
                self.oct_controller.slice_value_changed(sliceNum,'layerViewer',\
                    furtherUpdate=False)
                
                if(mode=='undo'):
                    self.oct_controller.pen_or_line_undo(sliceNum,info,layerName)
                    self.oct_controller.slice_value_changed(sliceNum,'layerViewer',\
                        furtherUpdate=False)
                elif(mode=='redo'):
                    info=self.oct_controller.pen_or_line_redo(sliceNum,layerName)
                    self.oct_controller.set_slice_edited(sliceNum,layerName,True)
                    self.oct_controller.slice_value_changed(sliceNum,'layerViewer',\
                        furtherUpdate=False)
                    return info
                    
        if(viewName=='enfaceDrusenViewer'):
            if(self.subwindowEnfaceDrusenViewerUI is not None):
                if(color[0]==255.):
                    self.oct_controller.oct.insert_druse_at(slices,posY,posX)
                    self.oct_controller.slice_value_changed(sliceNum,\
                        'enfaceDrusenViewer',furtherUpdate=False)
                else:
                    self.oct_controller.oct.remove_druse_at(slices,posY)
            if(self.subwindowDrusenViewerUI is not None):
                    self.oct_controller.slice_value_changed(sliceNum,\
                        'drusenViewer',furtherUpdate=False)
                    
    def draw_curve_command_undo_redo(self,x,s,y,color,viewName,sliceNum=1,\
            prevValues=[],mode='undo',info=None,layerName=''):
        if(viewName=='layerViewer'):
            if(self.subwindowLayerViewerUI is not None):
                for i in range(len(s)):
                    self.oct_controller.oct.instert_layer_at_slice(sliceNum,\
                        s[i],y[i],prevValues[i])
                self.oct_controller.slice_value_changed(sliceNum,'layerViewer',\
                    furtherUpdate=False)
                if(mode=='undo'):
                    self.oct_controller.pen_or_line_undo(sliceNum,info,layerName)
                    self.oct_controller.slice_value_changed(sliceNum,'layerViewer',\
                        furtherUpdate=False)
                elif(mode=='redo'):
                    info=self.oct_controller.pen_or_line_redo(sliceNum,layerName)
                    self.oct_controller.set_slice_edited(sliceNum,layerName,True)
                    self.oct_controller.slice_value_changed(sliceNum,'layerViewer',\
                        furtherUpdate=False)
                    return info
        if(viewName=='drusenViewer'):
            if(self.subwindowDrusenViewerUI is not None):
                for i in range(len(s)):
                    self.oct_controller.oct.instert_druse_at_slice(sliceNum,\
                        s[i],y[i],color[0])
                self.oct_controller.slice_value_changed(sliceNum,'drusenViewer',\
                    furtherUpdate=False)
        if(viewName=='hrfViewer'):
            if(self.subwindowHRFViewerUI is not None):
                for i in range(len(s)):
                    self.oct_controller.oct.insert_hrf_at_slice(sliceNum,s[i],\
                        y[i],color[0])
                self.oct_controller.slice_value_changed(sliceNum,'hrfViewer',\
                    furtherUpdate=False)
                self.oct_controller.update_HRF_status(sliceNum-1,\
                    self.oct_controller.oct.hrf_exist_in_slice(sliceNum),True)
                
        if(viewName=='enfaceDrusenViewer'):
            if(self.subwindowEnfaceDrusenViewerUI is not None):
                if(color[0]==255.):
                    for i in range(len(x)):
                        self.oct_controller.oct.insert_druse_at(s[i],y[i],x[i])
                    self.oct_controller.slice_value_changed(sliceNum,\
                        'enfaceDrusenViewer',furtherUpdate=False)
                else:
                    for i in range(len(y)):
                        self.oct_controller.oct.remove_druse_at(s[i],y[i])
            if(self.subwindowDrusenViewerUI is not None):
                    self.oct_controller.slice_value_changed(sliceNum,\
                        'drusenViewer',furtherUpdate=False)                
      
    def draw_fill_command_undo_redo(self,xs,ys,color,viewName,sliceNum=1,posX=[]):
        if(viewName=='drusenViewer'):
            if(self.subwindowDrusenViewerUI is not None):
                self.oct_controller.oct.instert_druse_at_slice(sliceNum,xs,ys,\
                    color[0])
                self.oct_controller.slice_value_changed(sliceNum,'drusenViewer',\
                    furtherUpdate=False)
        if(viewName=='hrfViewer'):
            if(self.subwindowHRFViewerUI is not None):
                self.oct_controller.oct.insert_hrf_at_slice(sliceNum,xs,ys,color[0])
                self.oct_controller.slice_value_changed(sliceNum,'hrfViewer',\
                    furtherUpdate=False)
                self.oct_controller.update_HRF_status(sliceNum-1,\
                    self.oct_controller.oct.hrf_exist_in_slice(sliceNum),True)
        if(viewName=='enfaceDrusenViewer'):
            if(self.subwindowEnfaceDrusenViewerUI is not None):
                if(color[0]==255.):
                    self.oct_controller.oct.insert_druse_at(xs,ys,posX)
                    self.oct_controller.slice_value_changed(sliceNum,\
                        'enfaceDrusenViewer',furtherUpdate=False)
                else:
                    self.oct_controller.remove_druse_at(xs,ys)
                    self.oct_controller.slice_value_changed(sliceNum,\
                        'enfaceDrusenViewer',furtherUpdate=False)

            if(self.subwindowDrusenViewerUI is not None):
                    self.oct_controller.slice_value_changed(sliceNum,\
                        'drusenViewer',furtherUpdate=False)
                    
    def draw_dilate_command_undo_redo(self,xs,ys,color,viewName,sliceNum=1):
        if(viewName=='drusenViewer'):
            if(self.subwindowDrusenViewerUI is not None):
                self.oct_controller.oct.instert_druse_at_slice(sliceNum,xs,ys,color[0])
                self.oct_controller.slice_value_changed(sliceNum,\
                    'drusenViewer',furtherUpdate=False)
        elif(viewName=='hrfViewer'):
            if(self.subwindowHRFViewerUI is not None):
                self.oct_controller.oct.insert_hrf_at_slice(sliceNum,xs,ys,color[0])
                self.oct_controller.slice_value_changed(sliceNum,'hrfViewer',\
                    furtherUpdate=False)
                self.oct_controller.update_HRF_status(sliceNum-1,\
                    self.oct_controller.oct.hrf_exist_in_slice(sliceNum),True)
                
    def draw_erosion_command_undo_redo(self,xs,ys,color,viewName,sliceNum=1,posX=[]):
        if(viewName=='drusenViewer'):
            if(self.subwindowDrusenViewerUI is not None):
                self.oct_controller.oct.instert_druse_at_slice(sliceNum,xs,\
                    ys,color[0])
                self.oct_controller.slice_value_changed(sliceNum,\
                    'drusenViewer',furtherUpdate=False)
        if(viewName=='enfaceDrusenViewer'):
            if(self.subwindowEnfaceDrusenViewerUI is not None):
                if(color[0]==255.):
                    self.oct_controller.oct.insert_druse_at(xs,ys,posX)
                    self.oct_controller.slice_value_changed(sliceNum,\
                        'enfaceDrusenViewer',furtherUpdate=False)
                else:
                    self.oct_controller.remove_druse_at(xs,ys)
                    self.oct_controller.slice_value_changed(sliceNum,\
                        'enfaceDrusenViewer',furtherUpdate=False)

            if(self.subwindowDrusenViewerUI is not None):
                    self.oct_controller.slice_value_changed(sliceNum,\
                        'drusenViewer',furtherUpdate=False)  
        if(viewName=='hrfViewer'):
            if(self.subwindowHRFViewerUI is not None):
                self.oct_controller.oct.insert_hrf_at_slice(sliceNum,xs,ys,\
                    color[0])
                self.oct_controller.slice_value_changed(sliceNum,'hrfViewer',\
                    furtherUpdate=False)
                self.oct_controller.update_HRF_status(sliceNum-1,\
                    self.oct_controller.oct.hrf_exist_in_slice(sliceNum),True)
                
    def draw_filter_command_undo_redo(self,xs,ys,zs,color,viewName,sliceNum=1):
        if(viewName=='drusenViewer'):
            if(self.subwindowDrusenViewerUI is not None):
                self.oct_controller.oct.instert_druse_at_slice(sliceNum,xs,ys,color[0])
                self.oct_controller.slice_value_changed(sliceNum,'drusenViewer',\
                    furtherUpdate=False)
        if(viewName=='enfaceDrusenViewer'):
            if(self.subwindowEnfaceDrusenViewerUI is not None):
                if(color[0]==255.):
                    self.oct_controller.oct.insert_druse_at_pos(xs,ys,zs)
                    self.oct_controller.slice_value_changed(sliceNum,\
                        'enfaceDrusenViewer',furtherUpdate=False)
                else:
                    self.oct_controller.remove_druse_at(zs,ys)
                    self.oct_controller.slice_value_changed(sliceNum,\
                        'enfaceDrusenViewer',furtherUpdate=False)

            if(self.subwindowDrusenViewerUI is not None):
                    self.oct_controller.slice_value_changed(sliceNum,\
                        'drusenViewer',furtherUpdate=False)             
               
    def draw_delete_command_undo_redo(self,xs,ys,zs,color,viewName,sliceNum=1):
        if(viewName=='hrfViewer'):
            if(self.subwindowHRFViewerUI is not None):
                self.oct_controller.oct.insert_hrf_at_slice(sliceNum,xs,ys,color[0])
                self.oct_controller.slice_value_changed(sliceNum,'hrfViewer',\
                    furtherUpdate=False)
                self.oct_controller.update_HRF_status(sliceNum-1,\
                    self.oct_controller.oct.hrf_exist_in_slice(sliceNum),True)
        if(viewName=='drusenViewer'):
            if(self.subwindowDrusenViewerUI is not None):
                self.oct_controller.oct.instert_druse_at_slice(sliceNum,xs,ys,color[0])
                self.oct_controller.slice_value_changed(sliceNum,'drusenViewer',\
                    furtherUpdate=False)
        if(viewName=='enfaceDrusenViewer'):
            if(self.subwindowEnfaceDrusenViewerUI is not None):
                if(color[0]==255.):
                    self.oct_controller.oct.insert_druse_at_pos(xs,ys,zs)
                    self.oct_controller.slice_value_changed(sliceNum,\
                        'enfaceDrusenViewer',furtherUpdate=False)
                else:
                    self.oct_controller.remove_druse_at(zs,ys)
                    self.oct_controller.slice_value_changed(sliceNum,\
                        'enfaceDrusenViewer',furtherUpdate=False)

            if(self.subwindowDrusenViewerUI is not None):
                    self.oct_controller.slice_value_changed(sliceNum,\
                        'drusenViewer',furtherUpdate=False)   
                    
    def draw_extract_command_undo_redo(self,xs,ys,zs,xns,yns,zns,color,\
            viewName,sliceNum=1):
        if(viewName=='drusenViewer'):
            if(self.subwindowDrusenViewerUI is not None):
                self.oct_controller.oct.instert_druse_at_slice(sliceNum,xs,ys,\
                    color[0])
                self.oct_controller.oct.instert_druse_at_slice(sliceNum,xns,yns,\
                    color[1])
                self.oct_controller.slice_value_changed(sliceNum,\
                    'drusenViewer',furtherUpdate=False)
        if(viewName=='hrfViewer'):
            if(self.subwindowHRFViewerUI is not None):
                self.oct_controller.oct.insert_hrf_at_slice(sliceNum,xs,ys,color[0])
                self.oct_controller.slice_value_changed(sliceNum,'hrfViewer',\
                    furtherUpdate=False)
                self.oct_controller.update_HRF_status(sliceNum-1,\
                    self.oct_controller.oct.hrf_exist_in_slice(sliceNum),True)
        if(viewName=='enfaceDrusenViewer'):
            if(self.subwindowEnfaceDrusenViewerUI is not None):
                if(color[0]==255.):
                    self.oct_controller.oct.insert_druse_at_pos(xs,ys,zs)
                    
                else:
                    self.oct_controller.oct.insert_value_at(xs,ys,zs,color[0])
                if(color[1]==255.):
                    self.oct_controller.oct.insert_druse_at_pos(xns,yns,zns)
                else:
                    self.oct_controller.oct.insert_value_at(xns,yns,zns,color[1])
                self.oct_controller.slice_value_changed(sliceNum,\
                    'enfaceDrusenViewer',furtherUpdate=False)
            if(self.subwindowDrusenViewerUI is not None):
                    self.oct_controller.slice_value_changed(sliceNum,\
                        'drusenViewer',furtherUpdate=False) 
                    
    def draw_region_command_undo_redo(self,xs,ys,color,viewName,sliceNum=1,gaType='GA'):
        if(viewName=='gaViewer'):
            if(self.subwindowGAViewerUI is not None):
                if(gaType=='GA'):
                    self.oct_controller.oct.insert_ga_at_slice(sliceNum,xs,ys,color[0])
                elif(gaType=='NGA'):
                    self.oct_controller.oct.insert_nga_at_slice(sliceNum,xs,ys,color[0])
                self.oct_controller.slice_value_changed(sliceNum,'gaViewer',\
                    furtherUpdate=False)
        
    def draw_box_command_undo(self,rect,sliceNum,viewName):
         if(viewName=='hrfViewer'):
            if(self.subwindowHRFViewerUI is not None):
                self.subwindowHRFViewerUI.graphicsViewImageViewer.\
                    delete_box(rect,sliceNum)
                self.oct_controller.slice_value_changed(sliceNum,\
                    'hrfViewer',furtherUpdate=False)
         if(viewName=='enfaceViewer'):
            if(self.subwindowEnfaceViewerUI is not None):
                self.subwindowEnfaceViewerUI.graphicsViewImageViewer.\
                    delete_box(rect,sliceNum)
         if(viewName=='gaViewer'):
            if(self.subwindowGAViewerUI is not None):
                self.subwindowGAViewerUI.graphicsViewImageViewer.\
                    delete_box(rect,sliceNum)
                self.oct_controller.slice_value_changed(sliceNum,\
                    'gaViewer',furtherUpdate=False)
                    
    def draw_box_command_redo(self,rect,sliceNum,viewName):
        if(viewName=='hrfViewer'):
            if(self.subwindowHRFViewerUI is not None):
                self.subwindowHRFViewerUI.graphicsViewImageViewer.\
                    add_box(rect,sliceNum)
                self.oct_controller.slice_value_changed(sliceNum,\
                    'hrfViewer',furtherUpdate=False)
        if(viewName=='enfaceViewer'):
            if(self.subwindowEnfaceViewerUI is not None):
                self.subwindowEnfaceViewerUI.graphicsViewImageViewer.\
                    add_box(rect,sliceNum)
        if(viewName=='gaViewer'):
            if(self.subwindowGAViewerUI is not None):
                self.subwindowGAViewerUI.graphicsViewImageViewer.\
                    add_box(rect,sliceNum)  
                self.oct_controller.slice_value_changed(sliceNum,\
                    'gaViewer',furtherUpdate=False)
                
    def remove_box_command_undo(self,rects,sliceNum,viewName):
        if(viewName=='hrfViewer'):
            if(self.subwindowHRFViewerUI is not None):
                self.subwindowHRFViewerUI.graphicsViewImageViewer.\
                    add_boxes(rects,sliceNum)
                self.oct_controller.slice_value_changed(sliceNum,\
                    'hrfViewer',furtherUpdate=False)
        if(viewName=='enfaceViewer'):
            if(self.subwindowEnfaceViewerUI is not None):
                self.subwindowEnfaceViewerUI.graphicsViewImageViewer.\
                    add_boxes(rects,sliceNum)
        if(viewName=='gaViewer'):
            if(self.subwindowGAViewerUI is not None):
                self.subwindowGAViewerUI.graphicsViewImageViewer.\
                    add_boxes(rects,sliceNum)      
                self.oct_controller.slice_value_changed(sliceNum,\
                    'gaViewer',furtherUpdate=False)
                    
    def remove_box_command_redo(self,rects,sliceNum,viewName):
         if(viewName=='hrfViewer'):
            if(self.subwindowHRFViewerUI is not None):
                self.subwindowHRFViewerUI.graphicsViewImageViewer.\
                    delete_boxes(rects,sliceNum)
                self.oct_controller.slice_value_changed(sliceNum,'hrfViewer',\
                    furtherUpdate=False)
         if(viewName=='enfaceViewer'):
            if(self.subwindowEnfaceViewerUI is not None):
                self.subwindowEnfaceViewerUI.graphicsViewImageViewer.\
                    delete_boxes(rects,sliceNum)
         if(viewName=='gaViewer'):
            if(self.subwindowGAViewerUI is not None):
                self.subwindowGAViewerUI.graphicsViewImageViewer.\
                    delete_boxes(rects,sliceNum)       
                self.oct_controller.slice_value_changed(sliceNum,'gaViewer',\
                    furtherUpdate=False)
                    
    def draw_pen_command(self,x,y,color,callerName,sliceNum=1,prevValues=[],\
            slices=1,posY=0,oldValue=[],redoValues=[],layerName=''):
        command=DrawPenCommand(self,x,y,color,callerName,sliceNum,prevValues,\
            slices,posY,oldValue,redoValues,layerName)
        self.undoStack.push(command)
        self.actionUndo.setEnabled(True)
    
    def draw_drusen_on_enface_command(self,x,y,color,normalThickness,callerName):    
        command=DrawDrusenOnEnfaceCommand(self,x,y,color,normalThickness,callerName)
        self.undoStack.push(command)
        self.actionUndo.setEnabled(True)
     
    def draw_line_on_enface_command(self,y,s,color,normalThickness,callerName):
        command=DrawLineOnDrusenEnface(self,y,s,color,normalThickness,callerName)
        self.undoStack.push(command)
        self.actionUndo.setEnabled(True)
        
    def draw_cost_point_command(self,x,y,smoothness,layerName,callerName,sliceNum):
        command=DrawCostPointCommand(self,x,y,smoothness,layerName,callerName,sliceNum)
        self.undoStack.push(command)
        self.actionUndo.setEnabled(True)   
        
    def draw_poly_fit_command(self,image,topLeftX,topLeftY,\
             bottomRightX,bottomRightY,sliceNum,polyDegree,currLayer,callerName):
        command=DrawPolyFitCommand(self,image,topLeftX,topLeftY,\
             bottomRightX,bottomRightY,sliceNum,polyDegree,currLayer,callerName)
        self.undoStack.push(command)
        self.actionUndo.setEnabled(True) 
   
    def extract_drunsen_using_normal_thickness_command(self,thickness,sliceZ,callerName):
        command=ExtractDrusenNormalThicknessCommand(self,thickness,sliceZ,callerName)
        self.undoStack.push(command)
        self.actionUndo.setEnabled(True)
        
    def accept_suggested_segmentation_command(self,sliceNumZ,layerName,\
            smoothness,uncType,extent,csps):
        command=AcceptSuggestedSegmentationCommand(self,sliceNumZ,layerName,\
                    smoothness,uncType,extent,csps)
        self.undoStack.push(command)
        self.actionUndo.setEnabled(True) 
   
    def draw_spline_layer_command(self,layer,knots,currLayer,currSlice):
        command=DrawSplineCommand(self,layer,knots,currLayer,currSlice)
        self.lastSplineCommand=command
        self.undoStack.push(command)
        self.actionUndo.setEnabled(True) 
    
    def draw_spline_layer_update_command(self,layer,knots):
        if(not self.lastSplineCommand is None):
            self.lastSplineCommand.set_redo_values(layer,knots)
    
    def apply_split_command(self):
        command=ApplySplitCommand(self)
        self.undoStack.push(command)
        self.actionUndo.setEnabled(True)  
        
    def draw_line_command(self,x1,y1,x2,y2,color,callerName,sliceNum=1,posX=[],\
            posY=[],slices=[],prevValues=[],redoValues=[],layerName=''):
        command=DrawLineCommand(self,x1,y1,x2,y2,color,callerName,sliceNum,posX,\
            slices,posY,prevValues,redoValues,layerName)
        self.undoStack.push(command)
        self.actionUndo.setEnabled(True)
    
    def draw_curve_command(self,x,y,s,color,callerName,sliceNum=1,prevValues=[],\
            redoValues=[],layerName=''):
        command=DrawCurveCommand(self,s,y,x,color,callerName,sliceNum,\
            prevValues,redoValues,layerName)
        self.undoStack.push(command)
        self.actionUndo.setEnabled(True)
    
    def draw_fill_command(self,xs,ys,color,callerName,sliceNum=1,posX=[]):
        command=DrawFillCommand(self,xs,ys,color,callerName,sliceNum,posX)
        self.undoStack.push(command)
        self.actionUndo.setEnabled(True)
    
    def draw_dilate_command(self,xs,ys,color,callerName,sliceNum):
        command=DrawDilateCommand(self,xs,ys,color,callerName,sliceNum)
        self.undoStack.push(command)
        self.actionUndo.setEnabled(True)
    
    def draw_erosion_command(self,xs,ys,color,callerName,sliceNum,posX=[]):
        command=DrawErosionCommand(self,xs,ys,color,callerName,sliceNum,posX)
        self.undoStack.push(command)
        self.actionUndo.setEnabled(True)
        
    def draw_filter_command(self,xs,ys,zs,color,callerName,sliceNum):
        command=DrawFilterCommand(self,xs,ys,zs,color,callerName,sliceNum)
        self.undoStack.push(command)
        self.actionUndo.setEnabled(True)
        
    def draw_delete_command(self,xs,ys,zs,color,callerName,sliceNum):
        command=DrawDeleteCommand(self,xs,ys,zs,color,callerName,sliceNum)
        self.undoStack.push(command)
        self.actionUndo.setEnabled(True)
        
    def draw_extract_command(self,xs,ys,zs,xns,yns,zns,color,callerName,sliceNum):
        command=DrawExtractCommand(self,xs,ys,zs,xns,yns,zns,color,callerName,sliceNum)
        self.undoStack.push(command)
        self.actionUndo.setEnabled(True)

    def draw_region_command(self,xs,ys,color,callerName,sliceNum,gaType):
        command=DrawRegionCommand(self,xs,ys,color,callerName,sliceNum,gaType)
        self.undoStack.push(command)
        self.actionUndo.setEnabled(True)
     
    def draw_box_command(self,rect,sliceNum,callerName):
        command=DrawBoxCommand(self,rect,sliceNum,callerName)
        self.undoStack.push(command)
        self.actionUndo.setEnabled(True)
    
    def remove_box_command(self,removedBoxes,sliceNum,callerName):
        command=RemoveBoxCommand(self,removedBoxes,sliceNum,callerName)
        self.undoStack.push(command)
        self.actionUndo.setEnabled(True)
        
    def slice_edited(self,sliceNum,callerName,layerName): 
        if(callerName=='layerViewer'):
            self.subwindowLayerViewerUI.slice_edited(sliceNum,layerName)
            
    def set_uncertainties(self,uncertainties,sliceNumZ):
        self.subwindowLayerViewerUI.set_uncertainty_value(uncertainties,sliceNumZ)
        if(not self.subwindowEnfaceViewerUI is None):
            self.subwindowEnfaceViewerUI.set_uncertainty_value(uncertainties,sliceNumZ)
        
    def action_undo(self):     
        self.oct_controller.write_in_log(self.oct_controller.get_time()+','+\
            self.actionUndo.objectName()+'\n')
        self.undoStack.undo()
        
        if(not self.undoStack.canUndo()):
            self.actionUndo.setEnabled(False)
        else:
            self.actionUndo.setEnabled(True)
        if(not self.undoStack.canRedo()):
            self.actionRedo.setEnabled(False)
        else:
            self.actionRedo.setEnabled(True)
            
    def action_redo(self):  
        self.oct_controller.write_in_log(self.oct_controller.get_time()+','+\
            self.actionRedo.objectName()+'\n')
        self.undoStack.redo()
        if(not self.undoStack.canRedo()):
            self.actionRedo.setEnabled(False)
        else:
            self.actionRedo.setEnabled(True)
        if(not self.undoStack.canUndo()):
            self.actionUndo.setEnabled(False)
        else:
            self.actionUndo.setEnabled(True)
   
    def get_scan_path(self,lastPath):
        # The QWidget widget is the base class of all user interface objects in PyQt4.
        w = QtGui.QWidget(self.centralwidget)
        # Set window size. 
        w.resize(320, 240)
        # Get file name
        filename=QtGui.QFileDialog.getExistingDirectory(w,'Open Dir',lastPath)
        self.statusbar.showMessage(filename)
        
        return filename 
   
    def set_status_bar(self,path):
        self.statusbar.showMessage(path)
   
    def get_save_path(self,lastPath,fname='',saveFormat=''):
        # The QWidget widget is the base class of all user interface objects in PyQt4.
        w = QtGui.QWidget(self.centralwidget)
        # Set window size. 
        w.resize(320, 240)
        # Get file name
        filename = QtGui.QFileDialog.getExistingDirectory(w, 'Save Dir', '')
        return filename 
        
    def show_progress_bar(self,message="Loading"):
        self.progressBarShowed=True
        self.progressBarUI.set_text(message)
        self.progressBar.show()
        QtGui.QApplication.processEvents()
    
    def hide_progress_bar(self):
        self.progressBar.hide()
        QtGui.QApplication.processEvents()
        if(self.progressBarShowed):
            self.progressBarShowed=False
            del self.progressBar
            self.progressBar = QtGui.QWidget()
            self.progressBarUI = pbv.Ui_FormProgressBar()
            self.progressBarUI.setupUi(self.progressBar)
            self.hide_progress_bar()
        
    def set_progress_bar_value(self,value):
        self.progressBarUI.set_progress_bar_value(value)
        
    def get_progress_var_value(self):
        return self.progressBarUI.get_progress_bar_value()
    
    def update_progress_using_step(self,step):
        self.progressBarUI.update_progress_using_step(step)
   
    def reset_value(self):
        self.progressBarUI.reset_value()
   
    def disable_all(self):
        self.action_Toolbox.setCheckable(True)
        self.action_Reset.setCheckable(True)
        self.actionShowBscans.setCheckable(True)
        self.actionFindLayers.setCheckable(True)
        self.actionFindDrusen.setCheckable(True)
        self.actionShowEnface.setCheckable(True)
        self.actionShowHRF.setCheckable(True)
        self.actionShowGA.setCheckable(True)
        self.actionShowEnfaceDrusen.setCheckable(True)
        self.actionShow3D.setCheckable(True)
        
        self.action_Toolbox.setChecked(False)
        self.action_Reset.setChecked(False)
        self.actionShowBscans.setChecked(False)
        self.actionFindLayers.setChecked(False)
        self.actionFindDrusen.setChecked(False)
        self.actionShowEnface.setChecked(False)
        self.actionShowEnfaceDrusen.setChecked(False)
        self.actionShowHRF.setChecked(False)
        self.actionShowGA.setChecked(False)
        self.actionShow3D.setChecked(False)
        
        self.action_Toolbox.setEnabled(False)
        self.action_Reset.setEnabled(False)
        self.actionShowBscans.setEnabled(False)
        self.actionFindLayers.setEnabled(False)
        self.actionFindDrusen.setEnabled(False)
        self.actionShowEnface.setEnabled(False)
        self.actionShowEnfaceDrusen.setEnabled(False)
        self.actionShow3D.setEnabled(False)
        self.actionShowHRF.setEnabled(False)
        self.actionShowGA.setEnabled(False)
        
        self.actionMeasureDrusen.setEnabled(False)
        self.action_Save.setEnabled(False)
        self.action_Save_As.setEnabled(False)
        self.actionUndo.setEnabled(False)
        self.actionRedo.setEnabled(False)
        
        self.uncertaintyMapMenu.setEnabled(False)   
        self.enfaceDrusenMenu.setEnabled(False)
        
        self.action_ShowUncertaintyColorMapEntProb.setChecked(False)
        
        self.action_ChangeDruseVisitingOrderSize.setChecked(False)
        self.action_ChangeDruseVisitingOrderHeight.setChecked(True)
        self.action_ChangeDruseVisitingOrderBrightness.setChecked(False)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "OCT Editor", None))
        self.menu_File.setTitle(_translate("MainWindow", "&File", None))
        self.menu_Edit.setTitle(_translate("MainWindow", "&Edit", None))
        self.menu_View.setTitle(_translate("MainWindow", "&View", None))
        self.menu_Window.setTitle(_translate("MainWindow", "&Window", None))
        self.menu_Help.setTitle(_translate("MainWindow", "&Help", None))
        self.toolBarBasic.setWindowTitle(_translate("MainWindow", "toolBar", None))
        self.toolBarOCTperdiction.setWindowTitle(_translate("MainWindow",\
            "toolBar_2", None))
        self.toolBarExtraAnalysis.setWindowTitle(_translate("MainWindow",\
            "toolBar_3", None))
        self.action_Open.setText(_translate("MainWindow", "&Open", None))
        self.action_Save.setText(_translate("MainWindow", "&Save", None))
        self.action_Save_As.setText(_translate("MainWindow", "Save As", None))
        self.action_Toolbox.setText(_translate("MainWindow", "&Toolbox", None))
        self.action_ShowUncertaintyColorMapEntProb.setText(\
            _translate("MainWindow", "Probability and Entropy", None))
        self.action_ShowSettings.setText(_translate("MainWindow", "Settings", None))
        self.action_ChangeDruseVisitingOrderSize.setText(_translate(\
            "MainWindow", "Use druse volume", None))
        self.action_ChangeDruseVisitingOrderHeight.setText(_translate(\
            "MainWindow", "Use druse height", None))
        self.action_ChangeDruseVisitingOrderBrightness.setText(_translate(\
            "MainWindow", "Use druse brightness", None))
               
        self.action_Reset.setText(_translate("MainWindow", "&Reset", None))
        self.actionFindLayers.setText(_translate(\
            "MainWindow", "FindLayers", None))
        self.actionFindLayers.setToolTip(_translate(\
            "MainWindow", "Find layers", None))
        self.actionFindDrusen.setText(_translate(\
            "MainWindow", "findDrusen", None))
        self.actionFindDrusen.setToolTip(_translate(\
            "MainWindow", "Find Drusen", None))
        self.actionShowEnface.setText(_translate(\
            "MainWindow", "showEnface", None))
        self.actionShowEnface.setToolTip(_translate(\
            "MainWindow", "Show Enface", None))
        self.actionShowEnfaceDrusen.setText(_translate(\
            "MainWindow", "showEnfaceDrusen", None))
        self.actionShowEnfaceDrusen.setToolTip(_translate(\
            "MainWindow", "Show enface drusen", None))
        
        self.actionShowGA.setText(_translate("MainWindow", "showGA", None))
        self.actionShowGA.setToolTip(_translate("MainWindow",\
            "Show Geographic Atrophy", None))
        
        self.actionShowHRF.setText(_translate("MainWindow", "showHRF", None))
        self.actionShowHRF.setToolTip(_translate("MainWindow",\
            "Show Hyperreflective Foci", None))
        
        self.actionShow3D.setText(_translate("MainWindow", "show3D", None))
        self.actionShow3D.setToolTip(_translate("MainWindow",\
            "Show PED volume", None))
        self.actionMeasureDrusen.setText(_translate("MainWindow",\
            "measureDrusen", None))
        self.actionMeasureDrusen.setToolTip(_translate("MainWindow",\
            "Measure Drusen", None))
    
    def show_toolbox(self):
        if(self.subwindowToolBoxUI is None):
            toolBox=SubView(self,self.oct_controller,"toolBox")
            ui = tb.Ui_toolBox(self.oct_controller)
            ui.setupUi(toolBox,self.actionUndo,self.actionRedo)
            self.action_Toolbox.setChecked(True)
            self.subwindowToolBoxUI=ui
            self.subwindowToolBox=toolBox
            self.subwindowToolBox.setWindowTitle(_translate("MainWindow",\
                "Toolbox", None))
            self.toolBoxHidden=False
            self.mdiSubwindowToolBox=self.mdiArea.addSubWindow(self.subwindowToolBox)
            self.subwindowToolBox.show()
        
    def deactivate_all(self):
        self.activePen=False
        self.activeLine=False
        self.activeDrawDru=False
        self.activeMorph=False
        self.activeFilter=False
        self.activeFill=False
        self.activeGrab=False
        self.activeBBox=False
        self.activeCostPnt=False
        self.activePolyFit=False
        self.activeDruseSplit=False
        
    def activate_tool_for_window(self):
        if(self.activePen):
            self.set_pen()
        elif(self.activeLine):
            self.set_line()
        elif(self.activeFill):
            self.set_fill()
        elif(self.activeDrawDru):
            self.set_draw_dru()
        elif(self.activeMorph):
            self.set_morphology(self.morphLevel)
        elif(self.activeFilter):
            self.set_filter_dru(self.filterTAll,self.filterTMax)
        elif(self.activeGrab):
            self.set_grab()
        elif(self.activeBBox):
            self.set_bounding_box()
        elif(self.activeCostPnt):
            self.set_cost_point(self.smoothness)
        elif(self.activePolyFit):
            self.set_poly_fit(self.polyDegreeValue)
        elif(self.activeDruseSplit):
            self.set_druse_splitting(self.splittingThreshold)
        else:
            pass
        
    def update_tool_box(self):
        if(not self.toolBoxHidden and self.subwindowToolBoxUI is not None):
            self.mdiSubwindowToolBox.adjustSize()
        
    # DrawPanel activations
    def set_pen(self):
        self.deactivate_all()
        if(self.subwindowDrusenViewerUI is not None):
            self.subwindowDrusenViewerUI.set_pen()
        if(self.subwindowLayerViewerUI is not None):
            self.subwindowLayerViewerUI.set_pen()
        if(self.subwindowEnfaceDrusenViewerUI is not None):
            self.subwindowEnfaceDrusenViewerUI.set_pen()
        if(self.subwindowHRFViewerUI is not None):
            self.subwindowHRFViewerUI.set_pen()
        self.activePen=True
        
    def set_line(self):
        self.deactivate_all()
        if(self.subwindowDrusenViewerUI is not None):
            self.subwindowDrusenViewerUI.set_line()
        if(self.subwindowLayerViewerUI is not None):
            self.subwindowLayerViewerUI.set_line()
        if(self.subwindowEnfaceDrusenViewerUI is not None):
            self.subwindowEnfaceDrusenViewerUI.set_line()
        if(self.subwindowHRFViewerUI is not None):
            self.subwindowHRFViewerUI.set_line()
        if(self.subwindowGAViewerUI is not None):
            self.subwindowGAViewerUI.set_line()
        self.activeLine=True
        
    def set_fill(self):
        self.deactivate_all()
        if(self.subwindowDrusenViewerUI is not None):
            self.subwindowDrusenViewerUI.set_fill()
        if(self.subwindowEnfaceDrusenViewerUI is not None):
            self.subwindowEnfaceDrusenViewerUI.set_fill()
        if(self.subwindowHRFViewerUI is not None):
            self.subwindowHRFViewerUI.set_fill()
        self.activeFill=True
    
    def set_draw_dru(self):
        self.deactivate_all()
        if(self.subwindowDrusenViewerUI is not None):
            self.subwindowDrusenViewerUI.set_draw_dru()
        if(self.subwindowEnfaceDrusenViewerUI is not None):
            self.subwindowEnfaceDrusenViewerUI.set_draw_dru()
        if(self.subwindowHRFViewerUI is not None):
            self.subwindowHRFViewerUI.set_draw_dru()
        self.activeDrawDru=True
        
    def set_morphology(self,itLevel):
        self.deactivate_all()
        if(self.subwindowDrusenViewerUI is not None):
            self.subwindowDrusenViewerUI.set_morphology(itLevel)
        if(self.subwindowEnfaceDrusenViewerUI is not None):
            self.subwindowEnfaceDrusenViewerUI.set_morphology(itLevel)
        if(self.subwindowHRFViewerUI is not None):
            self.subwindowHRFViewerUI.set_morphology(itLevel)
            
        self.activeMorph=True
        self.morphLevel=itLevel
        
    def set_filter_dru(self,filteringHeight,maxFilteringHeight):
        self.deactivate_all()
        if(self.subwindowDrusenViewerUI is not None):
            self.subwindowDrusenViewerUI.set_filter_dru(\
                filteringHeight,maxFilteringHeight)
        if(self.subwindowEnfaceDrusenViewerUI is not None):
            self.subwindowEnfaceDrusenViewerUI.set_filter_dru(\
                filteringHeight,maxFilteringHeight)
        self.activeFilter=True
        self.filterTAll=filteringHeight
        self.filterTMax=maxFilteringHeight
        
    def set_grab(self):
        self.deactivate_all()
        if(self.subwindowDrusenViewerUI is not None):
            self.subwindowDrusenViewerUI.set_grab()
        if(self.subwindowEnfaceViewerUI is not None):
            self.subwindowEnfaceViewerUI.set_grab()
        if(self.subwindowEnfaceDrusenViewerUI is not None):
            self.subwindowEnfaceDrusenViewerUI.set_grab()
        self.activeGrab=True
        
    def set_bounding_box(self):
        self.deactivate_all()
        if(self.subwindowHRFViewerUI is not None):
            self.subwindowHRFViewerUI.graphicsViewImageViewer.\
                set_bounding_box()
        if(self.subwindowEnfaceViewerUI is not None):
            self.subwindowEnfaceViewerUI.graphicsViewImageViewer.\
                set_bounding_box()
        if(self.subwindowGAViewerUI is not None):
            self.subwindowGAViewerUI.graphicsViewImageViewer.\
                set_bounding_box()
        self.activeBBox=True
        
    def set_cost_point(self,value):
        self.deactivate_all()
        if(self.subwindowLayerViewerUI is not None):
            self.subwindowLayerViewerUI.graphicsViewImageViewer.set_cost_point()
        self.activeCostPnt=True
        self.smoothness=value
    def set_poly_fit(self,degree):
        self.deactivate_all()
        if(self.subwindowLayerViewerUI is not None):
            self.subwindowLayerViewerUI.graphicsViewImageViewer.set_poly_fit(degree)
        self.activePolyFit=True  
        self.polyDegreeValue=degree
        
    def set_druse_splitting(self,value):
        self.deactivate_all()
        self.activeDruseSplit=True  
        self.splittingThreshold=value
        
    def apply_threshold_immediately(self,scope): 
        
        if((self.subwindowDrusenViewerUI is not None) and (scope=='bscan')):
            self.subwindowDrusenViewerUI.graphicsViewImageViewer.\
                apply_height_threholds()
            
        if(self.subwindowEnfaceDrusenViewerUI is not None and (scope=='volume')):
            self.subwindowEnfaceDrusenViewerUI.graphicsViewImageViewer.\
                apply_height_threholds()
        
    def all_threshold_value_changed(self,value,applyImmediately=False):
        if(self.subwindowDrusenViewerUI is not None):
            self.subwindowDrusenViewerUI.all_threshold_value_changed(value)
        if(self.subwindowEnfaceDrusenViewerUI is not None):
            self.subwindowEnfaceDrusenViewerUI.all_threshold_value_changed(value)
        self.filterTAll=value
 
    def morphology_value_changed(self,value):
        if(self.subwindowDrusenViewerUI is not None):
            self.subwindowDrusenViewerUI.morphology_value_changed(value)
        if(self.subwindowEnfaceDrusenViewerUI is not None):
            self.subwindowEnfaceDrusenViewerUI.morphology_value_changed(value)
        if(self.subwindowHRFViewerUI is not None):
            self.subwindowHRFViewerUI.morphology_value_changed(value)
        self.morphLevel=value
    
    def normal_thickness_value_changed(self,value):    
#        print "ToDo - in edited_main_window in normal_thickness_value_changed:",value
        pass
    
    def polydegree_value_changed(self,value):
        if(self.subwindowLayerViewerUI is not None):
            self.subwindowLayerViewerUI.poly_degree_value_changed(value)
        self.polyDegreeValue=value
    
    def smoothness_value_changed(self,value):
        if(self.subwindowLayerViewerUI is not None):
            self.subwindowLayerViewerUI.smoothness_value_changed(value)
        self.smoothness=value
        
    def max_threshold_value_changed(self,value):
        if(self.subwindowDrusenViewerUI is not None):
            self.subwindowDrusenViewerUI.max_threshold_value_changed(value)
        if(self.subwindowEnfaceDrusenViewerUI is not None):
            self.subwindowEnfaceDrusenViewerUI.max_threshold_value_changed(value)
        self.filterTMax=value

    def set_grab_position(self,position,callerName=''):  
        if(self.subwindowLayerViewerUI is not None):
            self.subwindowLayerViewerUI.grab_value_changed(position)
        if(self.subwindowDrusenViewerUI is not None):
            self.subwindowDrusenViewerUI.grab_value_changed(position)
        if(self.subwindowEnfaceDrusenViewerUI is not None):
            self.subwindowEnfaceDrusenViewerUI.grab_value_changed(position)
        if(self.subwindowEnfaceViewerUI is not None):
            self.subwindowEnfaceViewerUI.grab_value_changed(position)
            
    def set_cca_mask(self,mask,sliceNum):
        if(self.subwindowDrusenViewerUI is not None):
            self.subwindowDrusenViewerUI.graphicsViewImageViewer.set_cca_mask(mask)
            self.oct_controller.slice_value_changed(sliceNum,callerName=\
                'drusenViewer',furtherUpdate=False)
        if(self.subwindowEnfaceDrusenViewerUI is not None):
            self.subwindowEnfaceDrusenViewerUI.graphicsViewImageViewer.set_cca_mask(mask)
            self.oct_controller.slice_value_changed(sliceNum,callerName=\
                'enfaceDrusenViewer',furtherUpdate=False)
        
    def unset_cca_mask(self):
        if(self.subwindowDrusenViewerUI is not None):
            self.subwindowDrusenViewerUI.graphicsViewImageViewer.unset_cca_mask()
        
        if(self.subwindowEnfaceDrusenViewerUI is not None):
            self.subwindowEnfaceDrusenViewerUI.graphicsViewImageViewer.unset_cca_mask()  
   
    def show_viewer(self,image,viewerName,numSlices,uncertaintyValues=None,\
            entropyVals=None,probVals=None):
        imageEditor=SubView(self,self.oct_controller,viewerName)
        ui = iv.Ui_imageEditor(self.oct_controller,viewerName)
        ui.setupUi(imageEditor)
        if(viewerName=='scanViewer'):
            self.subwindowScanViewerUI=ui
            self.subwindowScanViewer=imageEditor
            self.subwindowScanViewerUI.graphicsViewImageViewer.set_main_image(image)
            self.subwindowScanViewerUI.set_max_possible_value(numSlices)
            self.subwindowScanViewer.setWindowTitle(_translate("MainWindow",\
                "B-Scan Viewer", None))
            self.mdiSubwindowScanViewer=self.mdiArea.addSubWindow(self.subwindowScanViewer)
            self.subwindowScanViewer.show()
            self.mdiSubwindowScanViewer.adjustSize()
            
        elif(viewerName=='layerViewer'):
            self.subwindowLayerViewerUI=ui
            self.subwindowLayerViewer=imageEditor
            self.subwindowLayerViewer.setWindowTitle(_translate("MainWindow",\
                "Layer Map Editor", None))
            self.subwindowLayerViewerUI.graphicsViewImageViewer.\
                set_main_image(image)
            self.subwindowLayerViewerUI.set_max_possible_value(numSlices)
            self.subwindowLayerViewerUI.set_uncertaintyValues(uncertaintyValues,\
                entropyVals,probVals)
            self.mdiSubwindowLayerViewer=self.mdiArea.addSubWindow(self.subwindowLayerViewer)
            self.subwindowLayerViewer.show()
        elif(viewerName=='hrfViewer'):
            self.subwindowHRFViewerUI=ui
            self.subwindowHRFViewer=imageEditor
            self.subwindowHRFViewer.setWindowTitle(_translate("MainWindow",\
                "Hyperreflective Foci Map Editor", None))
            self.subwindowHRFViewerUI.graphicsViewImageViewer.set_main_image(image)
            self.subwindowHRFViewerUI.set_max_possible_value(numSlices)
            self.mdiSubwindowHRFViewer=self.mdiArea.addSubWindow(self.subwindowHRFViewer)
            self.subwindowHRFViewer.show()
        elif(viewerName=='gaViewer'):
            self.subwindowGAViewerUI=ui
            self.subwindowGAViewer=imageEditor
            self.subwindowGAViewer.setWindowTitle(_translate("MainWindow",\
                "Geographic Atrophy Map Editor", None))
            self.subwindowGAViewerUI.graphicsViewImageViewer.set_main_image(image)
            self.subwindowGAViewerUI.set_max_possible_value(numSlices)
            self.mdiSubwindowGAViewer=self.mdiArea.addSubWindow(self.subwindowGAViewer)
            self.subwindowGAViewer.show()
        elif(viewerName=='drusenViewer'):
            self.subwindowDrusenViewerUI=ui
            self.subwindowDrusenViewer=imageEditor
            self.subwindowDrusenViewer.setWindowTitle(_translate("MainWindow",\
                "Drusen Map Editor", None))
            self.subwindowDrusenViewerUI.graphicsViewImageViewer.set_main_image(image)
            self.subwindowDrusenViewerUI.set_max_possible_value(numSlices)
            self.mdiSubwindowDrusenViewer=self.mdiArea.addSubWindow(self.subwindowDrusenViewer)
            self.subwindowDrusenViewer.show()
        elif(viewerName=='enfaceViewer'):   
            self.subwindowEnfaceViewerUI=ui
            self.subwindowEnfaceViewer=imageEditor
            self.subwindowEnfaceViewer.setWindowTitle(_translate("MainWindow",\
                "Enface Viewer", None))
            self.subwindowEnfaceViewerUI.graphicsViewImageViewer.set_main_image(image)
            self.subwindowEnfaceViewerUI.set_max_possible_value(numSlices)
            self.mdiSubwindowEnfaceViewer=self.mdiArea.addSubWindow(self.subwindowEnfaceViewer)
            self.subwindowEnfaceViewer.show()
        elif(viewerName=='enfaceDrusenViewer'):
          
#            self.mdiArea.removeSubWindow(self.mdiSubwindowScanViewer)
#            self.mdiArea.removeSubWindow(self.mdiSubwindowLayerViewer)
#            self.hide_subwindow(self.subwindowToolBox)
#            self.hide_subwindow(self.subwindowLayerViewer)
#            self.hide_subwindow(self.subwindowScanViewer)
            
            
            self.subwindowEnfaceDrusenViewerUI=ui
            self.subwindowEnfaceDrusenViewer=imageEditor
            self.subwindowEnfaceDrusenViewer.setWindowTitle(_translate(\
                "MainWindow", "Enface Drusen Editor", None))
            self.subwindowEnfaceDrusenViewerUI.graphicsViewImageViewer.set_main_image(image)
            self.subwindowEnfaceDrusenViewerUI.set_max_possible_value(numSlices)
#            self.mdiSubwindowToolBox=self.mdiArea.addSubWindow(self.subwindowToolBox)
#            self.mdiSubwindowDrusenViewer=self.mdiArea.addSubWindow(self.subwindowDrusenViewer)
            self.mdiSubwindowEnfaceDrusenViewer=self.mdiArea.addSubWindow(self.subwindowEnfaceDrusenViewer)
            
#            self.mdiSubwindowDrusenViewer.show()
#            self.subwindowToolBox.show()
#            self.subwindowDrusenViewer.show()
            self.subwindowEnfaceDrusenViewer.show()
            
        self.activate_tool_for_window()
        
    def add_overlay(self,overlayImages,viewerName,coeff=0.0):    
        if(viewerName=='layerViewer'):    
            self.subwindowLayerViewerUI.graphicsViewImageViewer.\
                add_overlay_image(overlayImages[0],coeff)# Scan
        elif(viewerName=='hrfViewer'):    
            self.subwindowHRFViewerUI.graphicsViewImageViewer.\
                add_overlay_image(overlayImages[0],coeff)# Scan
        elif(viewerName=='gaViewer'):    
            self.subwindowGAViewerUI.graphicsViewImageViewer.\
                add_overlay_image(overlayImages[0],coeff)# Scan
        elif(viewerName=='drusenViewer'):
            self.subwindowDrusenViewerUI.graphicsViewImageViewer.\
                add_overlay_image(overlayImages[0])# Scan
            self.subwindowDrusenViewerUI.graphicsViewImageViewer.\
                add_overlay_image(overlayImages[1])# Layer
        elif(viewerName=='enfaceViewer'):
            self.subwindowEnfaceViewerUI.graphicsViewImageViewer.\
                add_overlay_image(overlayImages[0],coeff)
        elif(viewerName=='enfaceDrusenViewer'):
            self.subwindowEnfaceDrusenViewerUI.graphicsViewImageViewer.\
                add_overlay_image(overlayImages[0],coeff)

    def update_viewer(self,images,coeffs,viewerName,sliceNumber):
        if(viewerName=='scanViewer'):
            self.subwindowScanViewerUI.graphicsViewImageViewer.update_slice_number(sliceNumber)
            self.subwindowScanViewerUI.graphicsViewImageViewer.update_main_image(images[0])
        elif(viewerName=='layerViewer'):
            self.subwindowLayerViewerUI.graphicsViewImageViewer.update_slice_number(sliceNumber)
            self.subwindowLayerViewerUI.graphicsViewImageViewer.set_coeffs(coeffs[::-1])
            self.subwindowLayerViewerUI.graphicsViewImageViewer.update_suggestion_layers(images[2])
            self.subwindowLayerViewerUI.graphicsViewImageViewer.update_main_image(images[1])
            self.subwindowLayerViewerUI.graphicsViewImageViewer.update_overlay_image(images[0],1)
        elif(viewerName=='hrfViewer'):
            self.subwindowHRFViewerUI.graphicsViewImageViewer.update_slice_number(sliceNumber)
            self.subwindowHRFViewerUI.graphicsViewImageViewer.set_coeffs(coeffs[::-1])
            self.subwindowHRFViewerUI.graphicsViewImageViewer.update_main_image(images[1])
            self.subwindowHRFViewerUI.graphicsViewImageViewer.update_overlay_image(images[0],1)
        elif(viewerName=='gaViewer'):
            self.subwindowGAViewerUI.graphicsViewImageViewer.update_slice_number(sliceNumber)
            self.subwindowGAViewerUI.graphicsViewImageViewer.set_coeffs(coeffs[::-1])
            self.subwindowGAViewerUI.graphicsViewImageViewer.update_main_image(images[1])
            self.subwindowGAViewerUI.graphicsViewImageViewer.update_overlay_image(images[0],1)
        elif(viewerName=='drusenViewer'):
            self.subwindowDrusenViewerUI.graphicsViewImageViewer.update_slice_number(sliceNumber)
            self.subwindowDrusenViewerUI.graphicsViewImageViewer.update_line(sliceNumber-1)
            self.subwindowDrusenViewerUI.graphicsViewImageViewer.set_coeffs(coeffs[::-1])
            self.subwindowDrusenViewerUI.graphicsViewImageViewer.update_main_image(images[2])
            self.subwindowDrusenViewerUI.graphicsViewImageViewer.update_overlay_image(images[0],2)
            self.subwindowDrusenViewerUI.graphicsViewImageViewer.update_overlay_image(images[1],1)
            self.subwindowDrusenViewerUI.graphicsViewImageViewer.set_drusen_separators(self.get_drusen_separators())
        elif(viewerName=='enfaceViewer' and self.subwindowEnfaceViewerUI is not None):
            self.subwindowEnfaceViewerUI.graphicsViewImageViewer.update_slice_number(sliceNumber)
            self.subwindowEnfaceViewerUI.graphicsViewImageViewer.update_line(sliceNumber-1)
            self.subwindowEnfaceViewerUI.graphicsViewImageViewer.set_coeffs(coeffs[::-1])
            self.subwindowEnfaceViewerUI.graphicsViewImageViewer.update_main_image(images[0])
            self.subwindowEnfaceViewerUI.graphicsViewImageViewer.update_overlay_image(images[1],1)
        elif(viewerName=='enfaceDrusenViewer' and self.subwindowEnfaceDrusenViewerUI is not None):
            self.subwindowEnfaceDrusenViewerUI.graphicsViewImageViewer.update_slice_number(sliceNumber)
            self.subwindowEnfaceDrusenViewerUI.graphicsViewImageViewer.update_line(sliceNumber-1)
            self.subwindowEnfaceDrusenViewerUI.graphicsViewImageViewer.set_coeffs(coeffs[::-1])
            self.subwindowEnfaceDrusenViewerUI.graphicsViewImageViewer.update_main_image(images[1])
            self.subwindowEnfaceDrusenViewerUI.graphicsViewImageViewer.update_overlay_image(images[0],1)
            
            
    def show_scan(self,image,numSlices):          
        self.show_viewer(image,'scanViewer',numSlices)
    
    def get_hrf_bounding_boxes(self):
        if(self.subwindowHRFViewerUI is not None):
            return self.subwindowHRFViewerUI.graphicsViewImageViewer.get_hrf_bounding_boxes()
        return None
    
    def get_nga_bounding_boxes(self):
        if(self.subwindowGAViewerUI is not None):
            return self.subwindowGAViewerUI.graphicsViewImageViewer.get_nga_bounding_boxes()
        return None
    
    def get_enface_bounding_boxes(self):
        if(self.subwindowEnfaceViewerUI is not None):
            return self.subwindowEnfaceViewerUI.graphicsViewImageViewer.get_enface_bounding_boxes()
        return None
   
    def set_hrf_check_box(self,status):
        self.subwindowHRFViewerUI.set_check_box(status)   
        
    def set_hrf_status(self,hrfStatus):
        self.subwindowHRFViewerUI.set_all_HRF_status(hrfStatus)
    
    def set_hrf_bbox(self,hrfBBox):
        self.subwindowHRFViewerUI.graphicsViewImageViewer.set_HRF_BBox(hrfBBox)   
    
    def set_nga_bbox(self,ngaBBox):
        self.subwindowGAViewerUI.graphicsViewImageViewer.set_nga_BBox(ngaBBox)   
        
    def set_enface_bbox(self,hrfBBox):
        self.subwindowEnfaceViewerUI.graphicsViewImageViewer.set_enface_BBox(hrfBBox)   
    
    def content_changed(self,viewerName):
        if(viewerName=='drusenViewer'):
            if(self.subwindowDrusenViewerUI is not None):
                self.subwindowDrusenViewer.setWindowTitle(_translate(\
                    "MainWindow", "Drusen Map Editor *", None))
                self.oct_controller.drusen_value_changed(viewerName)
        if(viewerName=='layerViewer'):
            if(self.subwindowLayerViewerUI is not None):
                self.subwindowLayerViewer.setWindowTitle(_translate(\
                    "MainWindow", "Layer Map Editor *", None))
                self.oct_controller.layer_value_changed(viewerName)
        if(viewerName=='hrfViewer'):
            if(self.subwindowHRFViewerUI is not None):
                self.subwindowHRFViewer.setWindowTitle(_translate(\
                    "MainWindow", "Hyperreflective Foci Map Editor *", None))
                self.oct_controller.hrf_value_changed(viewerName)
        if(viewerName=='gaViewer'):
            if(self.subwindowGAViewerUI is not None):
                self.subwindowGAViewer.setWindowTitle(_translate(\
                    "MainWindow", "Geographic Atrophy Map Editor *", None))
                self.oct_controller.ga_value_changed(viewerName)
        if(viewerName=='enfaceViewer'):
            if(self.subwindowEnfaceViewerUI is not None):
                self.subwindowEnfaceViewer.setWindowTitle(_translate(\
                    "MainWindow", "Enface Viewer *", None))
                self.oct_controller.enface_value_changed(viewerName)
        if(viewerName=='enfaceDrusenViewer'):
            if(self.subwindowEnfaceDrusenViewerUI is not None):
                self.subwindowEnfaceDrusenViewer.setWindowTitle(_translate(\
                    "MainWindow", "Enface Drusen Editor *", None))
                self.oct_controller.drusen_value_changed(viewerName)
                self.subwindowDrusenViewer.setWindowTitle(_translate(\
                    "MainWindow", "Drusen Map Editor *", None))
                self.oct_controller.drusen_value_changed(viewerName)
                
    def saved_changes(self):
        if(self.subwindowLayerViewerUI is not None):
            self.subwindowLayerViewer.setWindowTitle(_translate(\
                "MainWindow", "Layer Map Editor", None))
        if(self.subwindowHRFViewerUI is not None):
            self.subwindowHRFViewer.setWindowTitle(_translate(\
                "MainWindow", "Hyperreflective Foci Map Editor", None))
        if(self.subwindowGAViewerUI is not None):
            self.subwindowGAViewer.setWindowTitle(_translate(\
                "MainWindow", "Geographic Atrophy Map Editor", None))
        if(self.subwindowDrusenViewerUI is not None):
            self.subwindowDrusenViewer.setWindowTitle(_translate(\
                "MainWindow", "Drusen Map Editor", None))
        if(self.subwindowEnfaceViewerUI is not None):
            self.subwindowEnfaceViewer.setWindowTitle(_translate(\
                "MainWindow", "Enface Viewer", None))
        if(self.subwindowEnfaceDrusenViewerUI is not None):
            self.subwindowEnfaceDrusenViewer.setWindowTitle(_translate(\
                "MainWindow", "Enface Drusen Editor", None))

#==============================================================================
#   Action Functions
#==============================================================================
    def close_current_scan(self):
        choice=QtGui.QMessageBox.question(self.mainWindow,\
            "Close Scan","Save the changes before closing?",\
                QtGui.QMessageBox.Discard | QtGui.QMessageBox.Cancel |\
                QtGui.QMessageBox.Save)
        if(choice==QtGui.QMessageBox.Discard):
            return True
        if(choice==QtGui.QMessageBox.Cancel):
            return False
        if(choice==QtGui.QMessageBox.Save):
            self.save_action()
            return True
        return False
   
    def open_action(self):
        openFlag=True
        if(self.oct_controller.is_there_unsaved_changes()):
            openFlag=self.close_current_scan()
            
        if(openFlag):
            
            status=self.oct_controller.open_scan()
            if(status==0):
                self.oct_controller.write_in_log(self.oct_controller.get_time()+\
                    ','+self.action_Open.objectName()+','+\
                    self.oct_controller.get_scan_path()+'\n')
                self.actionShowBscans.setChecked(True)
                self.actionShowBscans.setEnabled(False)
                self.action_Toolbox.setEnabled(False)
                self.actionFindLayers.setEnabled(True)
                self.action_Save.setEnabled(True)
                self.action_Save_As.setEnabled(True)
                self.actionShowHRF.setEnabled(True)
                self.actionShowGA.setEnabled(True)
            
    def show_settings_action(self):
        self.oct_controller.write_in_log(self.oct_controller.get_time()+','+\
            self.action_ShowSettings.objectName()+'\n')
        self.settingWindow.show()
        
    def get_network_info(self):
        return self.settingWidnowUI.get_network_info()
        
    def save_action(self):
        self.oct_controller.write_in_log(self.oct_controller.get_time()+','+\
            self.action_Save.objectName()+'\n')
        self.oct_controller.save()
    
    def save_as_action(self):
        self.oct_controller.write_in_log(self.oct_controller.get_time()+','+\
            self.action_Save_As.objectName()+'\n')
        self.oct_controller.save_as()
        
    def find_HRF_action(self):
        self.show_toolbox()
        if(self.subwindowHRFViewerUI is None):
            self.oct_controller.write_in_log(self.oct_controller.get_time()+\
                ','+self.actionShowHRF.objectName()+'\n')
            self.oct_controller.get_hrfs()
            self.actionShowHRF.setChecked(True)
            self.actionShowHRF.setEnabled(False)
        else:
            if(self.actionShowHRF.isEnabled()):
                self.mdiArea.addSubWindow(self.subwindowHRFViewer)
                self.subwindowHRFViewer.show()
                self.actionShowHRF.setChecked(True)
                self.actionShowHRF.setEnabled(False)
                
    def set_edited_layers(self,editedLayers):
        if(not self.subwindowLayerViewerUI is None):
            self.subwindowLayerViewerUI.graphicsViewImageViewer.set_edited_layers(editedLayers)
        if(not self.subwindowEnfaceViewerUI is None):
            self.subwindowEnfaceViewerUI.graphicsViewImageViewer.set_edited_layers(editedLayers)
            
    def find_GA_action(self):
        self.show_toolbox()
        if(self.subwindowGAViewerUI is None):
            self.oct_controller.write_in_log(self.oct_controller.get_time()+\
                ','+self.actionShowGA.objectName()+'\n')
            self.oct_controller.get_gas()
            self.actionShowGA.setChecked(True)
            self.actionShowGA.setEnabled(False)
        else:
            if(self.actionShowGA.isEnabled()):
                self.mdiArea.addSubWindow(self.subwindowGAViewer)
                self.subwindowGAViewer.show()
                self.actionShowGA.setChecked(True)
                self.actionShowGA.setEnabled(False)
    
    def delete_previous(self):
        sw=self.mdiArea.subWindowList()
        for s in sw:
            self.mdiArea.removeSubWindow(s)
        self.subwindowScanViewerUI=None
        self.subwindowLayerViewerUI=None
        self.subwindowDrusenViewerUI=None
        self.subwindowEnfaceViewerUI=None
        self.subwindowEnfaceDrusenViewerUI=None
        
        self.subwindowHRFViewerUI=None
        self.subwindowGAViewerUI=None
        
        self.subwindowToolBoxUI=None
        self.uiDrusenInfoTable=None
        self.mdiSubwindowToolBox=None
        self.mdiSubwindowScanViewer=None
        self.mdiSubwindowLayerViewer=None
        self.mdiSubwindowDrusenViewer=None
        
        self.mdiSubwindowHRFViewer=None
        self.mdiSubwindowGAViewer=None        
        
        self.mdiSubwindowEnfaceViewer=None
        self.mdiSubwindowEnfaceDrusenViewer=None
        
        self.undoStack.clear()
        self.toolBoxHidden=True
        self.progressBarShowed=False
        self.disable_all()
        self.deactivate_all()
        
    def toolbox_action(self):
        if(not self.subwindowScanViewerUI is  None):
            if((self.action_Toolbox.isEnabled()) and\
                    (not self.subwindowToolBoxUI is  None)):
                self.oct_controller.write_in_log(self.oct_controller.get_time()+\
                    ','+self.action_Toolbox.objectName()+'\n')
                self.mdiSubwindowToolBox=self.mdiArea.addSubWindow(self.subwindowToolBox)
                self.subwindowToolBox.show()
                self.action_Toolbox.setChecked(True)
                self.action_Toolbox.setEnabled(False)
    
    def show_bscans_action(self):
        if(self.subwindowScanViewerUI is not None):
            if(self.actionShowBscans.isEnabled()):
                self.mdiSubwindowScanViewer=self.mdiArea.addSubWindow(\
                    self.subwindowScanViewer)
                self.subwindowScanViewer.show()
                self.actionShowBscans.setChecked(True)
                self.actionShowBscans.setEnabled(False)
                self.mdiSubwindowScanViewer.adjustSize()
    def find_layers_action(self):
        self.show_toolbox()
        if(self.subwindowLayerViewerUI is None):
            self.oct_controller.write_in_log(self.oct_controller.get_time()+\
                ','+self.actionFindLayers.objectName()+'\n')
            self.oct_controller.get_layers()
            self.actionFindLayers.setChecked(True)
            self.actionFindLayers.setEnabled(False)
            self.actionFindDrusen.setEnabled(True)
            self.actionShowEnface.setEnabled(True)
            self.actionShow3D.setEnabled(True)
            
        else:
            if(self.actionFindLayers.isEnabled()):
                self.mdiSubwindowLayerViewer=self.mdiArea.addSubWindow(self.subwindowLayerViewer)
                self.subwindowLayerViewer.show()
                self.actionFindLayers.setChecked(True)
                self.actionFindLayers.setEnabled(False)
        
    def find_drusen_action(self):
        if(self.subwindowDrusenViewerUI is None):
            self.oct_controller.write_in_log(self.oct_controller.get_time()+\
                ','+self.actionFindDrusen.objectName()+'\n')
            self.oct_controller.get_drusen()
            self.actionFindDrusen.setChecked(True)
            self.actionFindDrusen.setEnabled(False)
            self.actionShowEnfaceDrusen.setEnabled(True)
            self.actionMeasureDrusen.setEnabled(True)
        else:
            if(self.actionFindDrusen.isEnabled()):
                self.mdiSubwindowDrusenViewer=self.mdiArea.addSubWindow(self.subwindowDrusenViewer)
                self.subwindowDrusenViewer.show()
                self.actionFindDrusen.setChecked(True)
                self.actionFindDrusen.setEnabled(False)
        self.activate_tool_for_window()
        
    def show_enface_action(self):
        if(self.subwindowEnfaceViewerUI is None):
            self.oct_controller.write_in_log(self.oct_controller.get_time()+\
                ','+self.actionShowEnface.objectName()+'\n')
            self.oct_controller.get_enface()
            self.actionShowEnface.setChecked(True)
            self.actionShowEnface.setEnabled(False)
        else:
            if(self.actionShowEnface.isEnabled()):
                self.mdiSubwindowEnfaceViewer=self.mdiArea.addSubWindow(self.subwindowEnfaceViewer)
                self.subwindowEnfaceViewer.show()
                self.actionShowEnface.setChecked(True)
                self.actionShowEnface.setEnabled(False)
        self.activate_tool_for_window()
        
    def show_enface_drusen_action(self):
        
        if(self.subwindowEnfaceDrusenViewerUI is None):
            self.oct_controller.write_in_log(self.oct_controller.get_time()+\
                ','+self.actionShowEnfaceDrusen.objectName()+'\n')
            self.oct_controller.get_enface_drusen()
            self.actionShowEnfaceDrusen.setChecked(True)
            self.actionShowEnfaceDrusen.setEnabled(False)
        else:
            if(self.actionShowEnfaceDrusen.isEnabled()):
                
                
                
                self.mdiSubwindowEnfaceDrusenViewer=self.mdiArea.addSubWindow(self.subwindowEnfaceDrusenViewer)
                self.subwindowEnfaceDrusenViewer.show()
                self.actionShowEnfaceDrusen.setChecked(True)
                self.actionShowEnfaceDrusen.setEnabled(False)
        self.activate_tool_for_window()
        self.enfaceDrusenMenu.setEnabled(True)
        
    def show_3d_action(self):
        
        vol=self.oct_controller.get_PED_volume()
        self.show_PED_volume(vol,0.1)

    def prepare_special_layout(self):
        pass
#        for ss in self.mdiArea.subWindowList():
#            self.mdiArea.removeSubWindow(ss)
#        self.mdiSubwindowDrusenViewer=self.mdiArea.addSubWindow(self.subwindowDrusenViewer)
#        self.mdiSubwindowEnfaceDrusenViewer=self.mdiArea.addSubWindow(self.subwindowEnfaceDrusenViewer)
#        self.mdiSubwindowToolBox=self.mdiArea.addSubWindow(self.subwindowToolBox)
#        self.subwindowDrusenViewer.show()
#        self.subwindowEnfaceDrusenViewer.show()
#        self.subwindowToolBox.show()

    def update_drusen_table(self,cx,cy,area,height,volume,largeR,smallR,theta):
        self.uiDrusenInfoTable.set_data(cx,cy,area,height,volume,largeR,smallR,theta)
        
    def update_drusen_analysis(self):
        cx,cy,area,height,volume,largeR,smallR,theta=self.oct_controller.oct.quantify_drusen()
        self.uiDrusenInfoTable.set_data(cx,cy,area,height,volume,largeR,smallR,theta)
        self.uiDrusenInfoTable.enable_export()
        
    def measure_drusen_action(self):
        self.oct_controller.write_in_log(self.oct_controller.get_time()+','+\
            self.actionMeasureDrusen.objectName()+'\n')
        drusenInfoTable =SubView(self,self.oct_controller,"drusenInfo")
        self.oct_controller.oct.get_enface_drusen()
        ui = Ui_drusenInfoTable()
        ui.setupUi(drusenInfoTable,self.oct_controller)
        self.uiDrusenInfoTable=ui
        self.subwindowDrusenInfoViewer = drusenInfoTable
        self.mdiArea.addSubWindow(drusenInfoTable)
        cx,cy,area,height,volume,largeR,smallR,theta=self.oct_controller.oct.quantify_drusen()
        self.uiDrusenInfoTable.set_data(cx,cy,area,height,volume,largeR,smallR,theta)
        self.uiDrusenInfoTable.enable_export()
        self.subwindowDrusenInfoViewer.show()
      
    def show_image(self, image, block = True ):
        plt.imshow( image, cmap = plt.get_cmap('gray'))
        plt.show(block)
        
    def show_PED_volume(self,scan, value, interplolation='bilinear', block = True):
        debug=True
        if(debug):
            img=io.imread("/home/gorgi/Desktop/OCT-GUI/dummyData/PED3D.png")
            self.show_image(img)
        else:
            fig = plt.figure(figsize=(15.0,6.0))       
            ax = fig.add_subplot( 1, 1,1, projection='3d')
            
            value=value
            h, w, d = scan.shape
            img = (scan>200).astype(float)
            img = sc.ndimage.filters.gaussian_filter(img,0.5)
            Z, X, Y = np.where( img >= 0.2 )
            ax.plot_trisurf(X, Y, Z,cmap=plt.cm.RdYlBu_r, antialiased=False,\
                edgecolor='none')
            ax.set_zlim(180, 300)
            ax.view_init(45,26)
            plt.show(block)
            
    def hide_subwindow(self,subwindow):
        if(subwindow.name=='toolBox'):
            self.toolBoxHidden=True
            self.mdiArea.removeSubWindow(self.subwindowToolBox)
            self.action_Toolbox.setChecked(False)
            self.action_Toolbox.setEnabled(True)
        if(subwindow.name=='scanViewer'):
            self.mdiArea.removeSubWindow(self.subwindowScanViewer)
            self.actionShowBscans.setChecked(False)
            self.actionShowBscans.setEnabled(True)    
        if(subwindow.name=='drusenViewer'):
            self.mdiArea.removeSubWindow(self.subwindowDrusenViewer)
            self.actionFindDrusen.setChecked(False)
            self.actionFindDrusen.setEnabled(True)  
        if(subwindow.name=='layerViewer'):
            self.mdiArea.removeSubWindow(self.subwindowLayerViewer)
            self.actionFindLayers.setChecked(False)
            self.actionFindLayers.setEnabled(True) 
        if(subwindow.name=='hrfViewer'):
            self.mdiArea.removeSubWindow(self.subwindowHRFViewer)
            self.actionShowHRF.setChecked(False)
            self.actionShowHRF.setEnabled(True) 
        if(subwindow.name=='gaViewer'):
            self.mdiArea.removeSubWindow(self.subwindowGAViewer)
            self.actionShowGA.setChecked(False)
            self.actionShowGA.setEnabled(True)     
        if(subwindow.name=='enfaceViewer'):
            self.mdiArea.removeSubWindow(self.subwindowEnfaceViewer)
            self.actionShowEnface.setChecked(False)
            self.actionShowEnface.setEnabled(True) 
        if(subwindow.name=='enfaceDrusenViewer'):
            self.mdiArea.removeSubWindow(self.subwindowEnfaceDrusenViewer)
            self.actionShowEnfaceDrusen.setChecked(False)
            self.actionShowEnfaceDrusen.setEnabled(True) 
        if(subwindow.name=='drusenInfo'):
            self.mdiArea.removeSubWindow(self.subwindowDrusenInfoViewer)
            self.actionMeasureDrusen.setChecked(False)
            self.actionMeasureDrusen.setEnabled(True)   
            
    def set_spinbox_value(self,value,viewerName):
        if(viewerName=='scanViewer' and self.subwindowScanViewerUI is not None):
            self.subwindowScanViewerUI.set_spinbox_value(value)
        elif(viewerName=='layerViewer' and self.subwindowLayerViewerUI is not None):
            self.subwindowLayerViewerUI.set_spinbox_value(value)
        elif(viewerName=='hrfViewer' and self.subwindowHRFViewerUI is not None):
            self.subwindowHRFViewerUI.set_spinbox_value(value)
        elif(viewerName=='gaViewer' and self.subwindowGAViewerUI is not None):
            self.subwindowGAViewerUI.set_spinbox_value(value)
        elif(viewerName=='drusenViewer' and self.subwindowDrusenViewerUI is not None):
            self.subwindowDrusenViewerUI.set_spinbox_value(value)
        elif(viewerName=='enfaceViewer' and self.subwindowEnfaceViewerUI is not None):
            self.subwindowEnfaceViewerUI.set_spinbox_value(value)
        elif(viewerName=='enfaceDrusenViewer' and\
                self.subwindowEnfaceDrusenViewerUI is not None):
            self.subwindowEnfaceDrusenViewerUI.set_spinbox_value(value)
            
    def annotation_view_toggled(self,viewerName):
        if(viewerName=='layerViewer' and self.subwindowLayerViewerUI is not None):
            self.subwindowLayerViewerUI.graphicsViewImageViewer.toggle_annotation_view()
        elif(viewerName=='hrfViewer' and self.subwindowHRFViewerUI is not None):
            self.subwindowHRFViewerUI.graphicsViewImageViewer.toggle_annotation_view()
        elif(viewerName=='gaViewer' and self.subwindowGAViewerUI is not None):
            self.subwindowGAViewerUI.graphicsViewImageViewer.toggle_annotation_view()
        elif(viewerName=='drusenViewer' and self.subwindowDrusenViewerUI is not None):
            self.subwindowDrusenViewerUI.graphicsViewImageViewer.toggle_annotation_view()
        elif(viewerName=='enfaceViewer' and self.subwindowEnfaceViewerUI is not None):
            self.subwindowEnfaceViewerUI.graphicsViewImageViewer.toggle_annotation_view()
        elif(viewerName=='enfaceDrusenViewer' and self.subwindowEnfaceDrusenViewerUI is not None):
            self.subwindowEnfaceDrusenViewerUI.graphicsViewImageViewer.toggle_annotation_view()    
            
    def set_up_manual_marker_selection(self):
        if(not self.subwindowEnfaceDrusenViewerUI is None):
            self.subwindowEnfaceDrusenViewerUI.graphicsViewImageViewer.\
                set_manual_marker_selection()
            
    def get_manual_markers(self):
        if(not self.subwindowEnfaceDrusenViewerUI is None):
            return self.subwindowEnfaceDrusenViewerUI.graphicsViewImageViewer.\
                get_manual_markers()
            
    def show_drusen_splitting_separators(self,separators,separatorsAvgHeight,labels):
        if(not self.subwindowEnfaceDrusenViewerUI is None):
            self.subwindowEnfaceDrusenViewerUI.graphicsViewImageViewer.\
                show_drusen_splitting_separators(separators,separatorsAvgHeight,labels)
            self.subwindowToolBoxUI.set_separation_threshold_range(\
                np.min(separatorsAvgHeight),np.max(separatorsAvgHeight))
    
    def set_merge_drusen(self):
        if(not self.subwindowEnfaceDrusenViewerUI is None):
            self.subwindowEnfaceDrusenViewerUI.\
                graphicsViewImageViewer.set_merge_drusen()
    
    def unset_merge_drusen(self):
        if(not self.subwindowEnfaceDrusenViewerUI is None):
            self.subwindowEnfaceDrusenViewerUI.\
                graphicsViewImageViewer.unset_merge_drusen()
    
    def get_drusen_separators(self):
        if(not self.subwindowEnfaceDrusenViewerUI is None):
            return self.subwindowEnfaceDrusenViewerUI.\
                graphicsViewImageViewer.get_drusen_separators()
            
    def done_splitting(self):
        if(not self.subwindowEnfaceDrusenViewerUI is None):
            return self.subwindowEnfaceDrusenViewerUI.\
                graphicsViewImageViewer.done_splitting()
            
    def set_druse_info(self,height,volume,brightness):
        if(not self.subwindowToolBoxUI is None):
            return self.subwindowToolBoxUI.\
                set_druse_info(height,volume,brightness)
            
    def set_image_in_editor(self,image):
        if(not self.subwindowLayerViewerUI is None):
            return self.subwindowLayerViewerUI.graphicsViewImageViewer.\
                set_image_in_editor(image)
    
    def separation_theshold_changed(self,value):
        if(not self.subwindowEnfaceDrusenViewerUI is None):
            return self.subwindowEnfaceDrusenViewerUI.graphicsViewImageViewer.\
                set_separation_threshold(value)
            
    def warn_cannot_open_network(self):
        msg = QtGui.QMessageBox()
        msg.setIcon(QtGui.QMessageBox.Information)
        msg.setText("Cannot use deep learning network for layer segmentation."+\
            " Check the settings uncer File menu.")
        msg.setWindowTitle("Layer Segmentation Error")
        msg.show()
        
    def get_uncertainties(self):
        if(not self.subwindowLayerViewerUI is None):
            return self.subwindowLayerViewerUI.get_uncertainties()
    
    def get_current_active_window(self):
        if(not self.subwindowToolBoxUI is None):
            return self.subwindowToolBoxUI.get_current_active_window()
        return 'None'

    def curve_to_spline(self):
        if(not self.subwindowLayerViewerUI is None):
            return self.subwindowLayerViewerUI.graphicsViewImageViewer.\
                curve_to_spline()

    def spline_to_curve(self):
        if(not self.subwindowLayerViewerUI is None):
            return self.subwindowLayerViewerUI.graphicsViewImageViewer.\
                spline_to_curve()
