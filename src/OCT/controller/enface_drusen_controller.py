# -*- coding: utf-8 -*-
"""
Created in 2018

@author: Shekoufeh Gorgi Zadeh
"""

import copy
import numpy as np
import scipy as sc
import os, sys, inspect
from PyQt4 import QtGui
from scipy import ndimage as ndi
from matplotlib import pyplot as plt
from skimage.morphology import watershed
from skimage.feature import peak_local_max

viewPath=os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))[:-10]+"view"
modelPath=os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))[:-10]+"model"

if viewPath not in sys.path:
     sys.path.insert(0, viewPath)
     
if modelPath not in sys.path:
     sys.path.insert(0, modelPath)     

#==============================================================================
# Takes care of the possible functionalities over the overall drusen segmentation
# map such as splitting drusen into smaller ones. Drusen visiting order and etc
#==============================================================================
class EnfaceDrusenController:
    def __init__(self,drusen,enfaceProjection,octController):
        
        self.enfaceDrusen=((np.sum(drusen,axis=0)>0).astype(int).T)
        self.enfaceProjection=enfaceProjection
        self.octController=octController
        self.sortMethod='Height' #Size, Brightness, Height
        self.height=None
        self.drusenHeights=dict()
        self.drusenSizes=dict()
        self.drusenBrightnesses=dict()
        self.cca=None
        self.currentIndex=0
        self.idCount=0
        self.checkedDrusen=set()
        self.allDrusen=set()
        self.sortedLabels=list()
        self.background=list()
        self.mask=None
        self.prevIndex=0
        self.sortedLabelsSize=None
        self.sortedLabelsHeight=None
        self.sortedLabelsBrightness=None
        self.compute_cca(drusen)
    
    def get_data(self):
        
        data=dict()
        
        data['enfaceDrusen']=np.copy(self.enfaceDrusen)
        data['enfaceProjection']=np.copy(self.enfaceProjection)
        data['sortMethod']=self.sortMethod #Size, Brightness, Height
        data['height']=np.copy(self.height)
        data['drusenHeights']=dict()
        data['drusenSizes']=dict()
        data['drusenBrightnesses']=dict()
        
        for l in self.drusenHeights.keys():
            data['drusenHeights'][l]=copy.deepcopy(self.drusenHeights[l])
            data['drusenSizes'][l]=copy.deepcopy(self.drusenSizes[l])
            data['drusenBrightnesses'][l]=copy.deepcopy(self.drusenBrightnesses[l])
        
        data['cca']=np.copy(self.cca)
        data['currentIndex']=self.currentIndex
        data['idCount']=self.idCount
        data['checkedDrusen']=self.checkedDrusen
        data['allDrusen']=self.allDrusen
        data['sortedLabels']=self.sortedLabels
        data['background']=self.background
        data['mask']=np.copy(self.mask)
        data['prevIndex']=self.prevIndex
        data['sortedLabelsSize']=np.copy(np.asarray(self.sortedLabelsSize))
        data['sortedLabelsHeight']=np.copy(np.asarray(self.sortedLabelsHeight))
        data['sortedLabelsBrightness']=np.copy(np.asarray(self.sortedLabelsBrightness))
        
        return data
        
    def get_current_position(self):
        
        if(self.sortMethod=='Height'):
            a=self.drusenHeights[self.sortedLabels[self.currentIndex]][1][0]
            b=self.drusenHeights[self.sortedLabels[self.currentIndex]][2][0]
            return a,b
        
        elif(self.sortMethod=='Size'):
            a=self.drusenSizes[self.sortedLabels[self.currentIndex]][1][0]
            b=self.drusenSizes[self.sortedLabels[self.currentIndex]][2][0]
            return a,b
        
        elif(self.sortMethod=='Brightness'):
            a=self.drusenBrightnesses[self.sortedLabels[self.currentIndex]][1][0]
            b=self.drusenBrightnesses[self.sortedLabels[self.currentIndex]][2][0]  
            return a,b
        
    def get_mask(self):
        return self.mask  
        
     
    def set_data(self,data):
    
        self.enfaceDrusen=data['enfaceDrusen']
        self.enfaceProjection=data['enfaceProjection']
        self.sortMethod=data['sortMethod'] #Size, Brightness, Height
        self.height=data['height']
        self.drusenHeights=data['drusenHeights']
        self.drusenSizes=data['drusenSizes']
        self.drusenBrightnesses=data['drusenBrightnesses']
        self.cca=data['cca']
        self.currentIndex=data['currentIndex']
        self.idCount=data['idCount']
        self.checkedDrusen=data['checkedDrusen']
        self.allDrusen=data['allDrusen']
        self.sortedLabels=data['sortedLabels']
        self.background=data['background']
        self.mask=data['mask']
        self.prevIndex=data['prevIndex']
        self.sortedLabelsSize=list(data['sortedLabelsSize'])
        self.sortedLabelsHeight=list(data['sortedLabelsHeight'])
        self.sortedLabelsBrightness=list(data['sortedLabelsBrightness'])
        
    def set_sort_method(self,m):
        self.unmark_current_druse()
        self.sortMethod=m        

    def set_enface_drusen(self,enfaceDrusen):
        self.enfaceDrusen=enfaceDrusen
         
     
    def compute_cca(self,drusen):
        cca, numDrusen = sc.ndimage.measurements.label( self.enfaceDrusen )
        self.idCount=numDrusen-1
        self.prevIndex=numDrusen-1
        self.cca=cca
        
        labels=np.unique(cca)
        self.compute_height(drusen)
        brightmap=np.empty(self.enfaceProjection.shape)
        
        hs=list()
        sizes=list()
        brightness=list()
        for l in labels:
            brightmap.fill(0.)
            y,x=np.where(cca==l) # Extract certain component
            brightmap[y,x]=1
            denom=np.sum(brightmap)
            brightmapTmp=sc.ndimage.morphology.binary_dilation(brightmap,\
                                                      iterations=1).astype(int)
            brightmap=brightmapTmp-brightmap
            if(denom>0 and np.sum(brightmap)>0):
                innerVal=np.sum(self.enfaceProjection[y,x])/float(denom)
                bondVal=np.sum(self.enfaceProjection[np.where(brightmap==1)])/\
                                                       float(np.sum(brightmap))
                dbrightness=innerVal-bondVal
            else:
                dbrightness=0.
            maxHeight=np.max(self.height[y,x])
            dsize=np.sum(self.height[y,x])
            if(maxHeight>0):
                self.drusenHeights[l]=[maxHeight,y,x]
                self.drusenSizes[l]=[dsize,y,x]
                self.drusenBrightnesses[l]=[dbrightness,y,x]
                self.allDrusen.add(l)
                self.sortedLabels.append(l)
                hs.append(maxHeight)
                sizes.append(dsize)
                brightness.append(dbrightness)
            else: # Background
                self.background=[l,y,x]
        
        indx=np.argsort(hs)
        sortedArray=np.asarray(self.sortedLabels)[indx]
        sortedArray=sortedArray[::-1]
        self.sortedLabelsHeight=np.copy(sortedArray)
        indx=np.argsort(sizes)
        sortedArray=np.asarray(self.sortedLabels)[indx]
        sortedArray=sortedArray[::-1]
        self.sortedLabelsSize=np.copy(sortedArray)
        indx=np.argsort(brightness)
        sortedArray=np.asarray(self.sortedLabels)[indx]
        self.sortedLabelsBrightness=np.copy(sortedArray)
        
        # Sort labels
        self.update_sorting_method(False)     
     
    def compute_height(self,drusen):
        self.height=np.sum(((drusen>0).astype(int)),axis=0).T
     
    def compute_separator_average_height(self,sepHeights,labels):
        visitedNeighbors=dict()
        ls=np.unique(labels)
        res=np.empty(sepHeights.shape)
        res.fill(0.)
        for l in ls:
            
            if(l==0): # Background
                continue
            if(not l in visitedNeighbors.keys()):
                visitedNeighbors[l]=list()
            enlargedL=sc.ndimage.morphology.binary_dilation((labels==l).\
                                                      astype(int),iterations=1)
            neigh=np.unique(enlargedL*labels)
            for nl in neigh:
                if(nl==0 or nl==l): 
                    continue
                if(not nl in visitedNeighbors.keys()):
                    visitedNeighbors[nl]=list()
                if(l in visitedNeighbors[nl]):
                    continue
                
                visitedNeighbors[l].append(nl)
                visitedNeighbors[nl].append(l)
                enlargedNL=sc.ndimage.morphology.binary_dilation((labels==nl).\
                                                      astype(int),iterations=1)
                intersection=enlargedL*enlargedNL
                intersection=intersection*sepHeights
                union=enlargedL+enlargedNL
                union[np.where(union==2)]=1
                res[np.where(intersection>0)]=np.average(intersection[np.where(intersection>0)])
                
        return res


    def next_drusen_id(self):
        existingLabels=np.unique((self.enfaceDrusen>0).astype(int)*self.cca)
        self.prevIndex=self.currentIndex
        self.currentIndex=(self.currentIndex+1)%self.idCount
        it=0
        while(not self.sortedLabels[self.currentIndex] in existingLabels):
            if(it>self.idCount):
                break
            self.currentIndex=(self.currentIndex+1)%self.idCount
            it+=1
        self.update_mask()
        
    def previous_drusen_id(self):
        existingLabels=np.unique((self.enfaceDrusen>0).astype(int)*self.cca)
        self.prevIndex=self.currentIndex
        self.currentIndex=(self.currentIndex-1)%self.idCount    
        it=0
        while(not self.sortedLabels[self.currentIndex] in existingLabels):
            if(it>self.idCount):
                break
            self.currentIndex=(self.currentIndex-1)%self.idCount
            it+=1
        self.update_mask()
        
    def check_current_drusen(self):
        self.checkedDrusen.add(self.sortedLabels[self.currentIndex])
        self.next_drusen_id()
        
    def uncheck_current_drusen(self):
        self.checkedDrusen.discard(self.sortedLabels[self.currentIndex])
        self.previous_drusen_id()
        
    def split_druse_using_manual_markers(self,markers):
        if(self.sortMethod=='Height'):
            druLoc=self.drusenHeights
        elif(self.sortMethod=='Size'):
            druLoc=self.drusenSizes
        elif(self.sortMethod=='Brightness'):
            druLoc=self.drusenBrightnesses
            
        y,x=druLoc[self.sortedLabels[self.currentIndex]][1],\
                druLoc[self.sortedLabels[self.currentIndex]][2]    

        drusen=self.octController.oct.get_drusen()
        
        self.compute_height(drusen)
        height=np.empty(self.height.shape)
        height.fill(0.)
        height[y,x]=self.height[y,x]
        hmask=(height>0).astype(int)
        local_maxi=np.empty(height.shape)
        local_maxi.fill(0.)
        for m in markers:
            local_maxi[m[1],m[0]]=1.
        markers = ndi.label(local_maxi)[0]
        labelsWithLines = watershed(-height, markers, mask=hmask,watershed_line=True)
        labels=watershed(-height, markers, mask=hmask,watershed_line=False)
        labelsWithLines[np.where(hmask==0)]=-1
        separators=np.empty(labelsWithLines.shape)
        separators.fill(0.)
        separators[np.where(labelsWithLines==0)]=1.
        sepHeights=height*separators
        separatorAvgHeight=self.compute_separator_average_height(sepHeights,labels)
        
        hmaskdilated=sc.ndimage.morphology.binary_dilation(hmask,iterations=1)
        hmaskdilated=hmaskdilated-hmask

        separators[np.where(hmaskdilated==1)]=2
        return separators,separatorAvgHeight,labels    
        
    def split_druse(self,neighborhoodSize):
        if(self.sortMethod=='Height'):
            druLoc=self.drusenHeights
        elif(self.sortMethod=='Size'):
            druLoc=self.drusenSizes
        elif(self.sortMethod=='Brightness'):
            druLoc=self.drusenBrightnesses
            
        y,x=druLoc[self.sortedLabels[self.currentIndex]][1],\
                druLoc[self.sortedLabels[self.currentIndex]][2]
        drusen=self.octController.oct.get_drusen()
        self.compute_height(drusen)
        height=np.empty(self.height.shape)
        height.fill(0.)
        height[y,x]=self.height[y,x]
        hmask=(height>0).astype(int)
        heightImg=sc.ndimage.filters.gaussian_filter(height,1.)
        local_maxi = peak_local_max(heightImg, indices=False, footprint=\
                                 np.ones((neighborhoodSize, neighborhoodSize)),\
                                 labels=hmask)                    
        markers = ndi.label(local_maxi)[0]
        labelsWithLines = watershed(-height, markers, mask=hmask,watershed_line=True)
        labels=watershed(-height, markers, mask=hmask,watershed_line=False)
        labelsWithLines[np.where(hmask==0)]=-1
        separators=np.empty(labelsWithLines.shape)
        separators.fill(0.)
        separators[np.where(labelsWithLines==0)]=1.
        sepHeights=height*separators
        separatorAvgHeight=self.compute_separator_average_height(sepHeights,labels)
        hmaskdilated=sc.ndimage.morphology.binary_dilation(hmask,iterations=1)
        hmaskdilated=hmaskdilated-hmask
        separators[np.where(hmaskdilated==1)]=2

        return separators,separatorAvgHeight,labels
        
    def split_drusen_everywhere(self):
        drusen=self.octController.oct.get_drusen()
        self.compute_height(drusen)
        hmask=(self.height>0).astype(int)
        heightImg=sc.ndimage.filters.gaussian_filter(self.height,1.)
        local_maxi = peak_local_max(heightImg, indices=False, footprint=\
                                                np.ones((10, 10)),labels=hmask)                     
        markers = ndi.label(local_maxi)[0]
        labels = watershed(-self.height, markers, mask=hmask,watershed_line=True)
        labels[np.where(hmask==0)]=-1
        separators=np.empty(labels.shape)
        separators.fill(0.)
        separators[np.where(labels==0)]=self.height[np.where(labels==0)]
        enface=self.enfaceProjection*0.5
        enface[local_maxi]=255.
        self.show_image(labels)    
        
    def unmark_current_druse(self):
        if(self.sortMethod=='Height'):
            druLoc=self.drusenHeights
        elif(self.sortMethod=='Size'):
            druLoc=self.drusenSizes
        elif(self.sortMethod=='Brightness'):
            druLoc=self.drusenBrightnesses
        y,x=druLoc[self.sortedLabels[self.currentIndex]][1],\
                druLoc[self.sortedLabels[self.currentIndex]][2]
        self.mask[y,x,0]=255
        self.mask[y,x,1]=0
        self.mask[y,x,2]=200   
        
    
    def select_component(self,x,y):
        l=self.cca[x-1,y]
        existingLabels=np.unique((self.enfaceDrusen>0).astype(int)*self.cca)
        
        if( l in existingLabels ):
            indx=np.where(self.sortedLabels==l)
            if(len(indx)>0 and len(indx[0])>0):
                if(self.currentIndex!=indx[0][0]):
                    self.prevIndex=self.currentIndex
                    self.currentIndex=indx[0][0]
                    self.update_mask()
                    
                
    def update_sorting_method(self,keepLast=True):
        currLabel=self.sortedLabels[self.currentIndex]
        if(self.sortMethod=='Height'):
            self.sortedLabels=self.sortedLabelsHeight
        elif(self.sortMethod=='Size'):
            self.sortedLabels=self.sortedLabelsSize
        elif(self.sortMethod=='Brightness'):
            self.sortedLabels=self.sortedLabelsBrightness
        if(keepLast):
            self.currentIndex=np.where(self.sortedLabels==currLabel)[0][0]
        self.update_mask()

    def update_mask(self):
        if(self.sortMethod=='Height'):
            druLoc=self.drusenHeights
        elif(self.sortMethod=='Size'):
            druLoc=self.drusenSizes
        elif(self.sortMethod=='Brightness'):
            druLoc=self.drusenBrightnesses
        
        # Get current druse info for the view to be updated
        dheight=self.drusenHeights[self.sortedLabels[self.currentIndex]][0]
        dvolume=self.drusenSizes[self.sortedLabels[self.currentIndex]][0]
        dbrightness=self.drusenBrightnesses[self.sortedLabels[self.currentIndex]][0]
        self.octController.set_druse_info(dheight,dvolume,dbrightness)
        
        if(self.mask is None):
            
            self.mask=np.zeros((self.enfaceDrusen.shape[0],self.enfaceDrusen.shape[1],3))
            for l in self.allDrusen:
                y,x=druLoc[l][1],druLoc[l][2]
                self.mask[y,x,0]=255
                self.mask[y,x,1]=0
                self.mask[y,x,2]=200
            y,x=druLoc[self.sortedLabels[self.currentIndex]][1],\
                druLoc[self.sortedLabels[self.currentIndex]][2]
            self.mask[y,x,0]=255
            self.mask[y,x,1]=255
            self.mask[y,x,2]=0
            
        else:
            y,x=druLoc[self.sortedLabels[self.currentIndex]][1],\
                druLoc[self.sortedLabels[self.currentIndex]][2]
            self.mask[y,x,0]=255
            self.mask[y,x,1]=255
            self.mask[y,x,2]=0
            y,x=druLoc[self.sortedLabels[self.prevIndex]][1],\
                druLoc[self.sortedLabels[self.prevIndex]][2]
            if(self.sortedLabels[self.prevIndex] in self.checkedDrusen):
                self.mask[y,x,0]=0
                self.mask[y,x,1]=255
                self.mask[y,x,2]=0
            elif(not self.sortedLabels[self.prevIndex] in self.checkedDrusen):
                self.mask[y,x,0]=255
                self.mask[y,x,1]=0
                self.mask[y,x,2]=200
                   

    def show_image(self, image, block = True ):
        plt.imshow( image, cmap = plt.get_cmap('gray'))
        plt.show(block)    
        QtGui.QApplication.processEvents()    
        
    def show_label(self, image, block = True ):
        plt.imshow( image, cmap = plt.cm.nipy_spectral)
        plt.show(block)    
        QtGui.QApplication.processEvents()   
     
    def show_images(self, images,r,c,titles=[],d=0,save_path="",block = True):
        i = 1
        
        for img in images:
            ax = plt.subplot( r, c, i )
            ax.xaxis.set_visible( False )
            ax.yaxis.set_visible( False )
            if( len(titles) != 0 ):
                ax.set_title( titles[i-1] )
            if( len(img.shape) > 2 ):
                plt.imshow( img )
            else:
                plt.imshow( img , cmap = plt.cm.get_cmap('gray'))
            i += 1
        if( save_path != "" ):
            plt.close()
        else:
            plt.show(block)
        QtGui.QApplication.processEvents()  
        
    def show_labels(self, images,r,c,titles=[],d=0,save_path="",block = True):
        i = 1
        for img in images:
            ax = plt.subplot( r, c, i )
            ax.xaxis.set_visible( False )
            ax.yaxis.set_visible( False )
            if( len(titles) != 0 ):
                ax.set_title( titles[i-1] )
            if( len(img.shape) > 2 ):
                plt.imshow( img )
            else:
                plt.imshow( img , cmap = plt.cm.nipy_spectral)
            i += 1
        if( save_path != "" ):
            plt.close()
        else:
            plt.show(block)
        QtGui.QApplication.processEvents()  
        
    def run_watershed(self,method,neighborhood):
        
        if(method=='localMax'):
            separators, separatorsAvgHeight,labels=self.split_druse(neighborhood)
            return separators,separatorsAvgHeight, labels
            
        elif(method=='manual'):
            markers=self.octController.get_manual_markers()
            if(len(markers)>0):
                separators,separatorsAvgHeight,labels=\
                    self.split_druse_using_manual_markers(markers)
                return separators,separatorsAvgHeight, labels
                
        return None,None,None

    def apply_separator(self,separators):
        # Get the current drusen index (label)
        currLabel=self.sortedLabels[self.currentIndex]
        if(self.sortMethod=='Height'):
            druLoc=self.drusenHeights
        elif(self.sortMethod=='Size'):
            druLoc=self.drusenSizes
        elif(self.sortMethod=='Brightness'):
            druLoc=self.drusenBrightnesses
        y,x=druLoc[self.sortedLabels[self.currentIndex]][1],\
                druLoc[self.sortedLabels[self.currentIndex]][2]
        self.mask[y,x,0]=255
        self.mask[y,x,1]=0
        self.mask[y,x,2]=200 
        self.sortedLabels=list(self.sortedLabels)
        
        if(currLabel in self.drusenHeights.keys()):
            h,y,x=self.drusenHeights.pop(currLabel)
        if(currLabel in self.drusenSizes.keys()):
            self.drusenSizes.pop(currLabel)
        if(currLabel in self.drusenBrightnesses.keys()):
            self.drusenBrightnesses.pop(currLabel)
        if(currLabel in self.allDrusen):
            self.allDrusen.discard(currLabel)  
        if(currLabel in self.checkedDrusen):
            self.checkedDrusen.discard(currLabel)
        if(currLabel in self.sortedLabels):
            self.sortedLabels.remove(currLabel)
           
        mask=np.empty(self.enfaceDrusen.shape)
        mask.fill(2.)
        mask[y,x]=1.
        mask=mask-1
        
        slices,ys=np.where(separators==1)
        self.octController.oct.remove_druse_at(slices,ys)
        drusen=self.octController.oct.get_drusen()
        self.enfaceDrusen=((np.sum(drusen,axis=0)>0).astype(int).T)
        self.compute_height(drusen)
      
        enfaceDrusen=np.copy(self.enfaceDrusen)
        enfaceDrusen[np.where(mask==1)]=0.
        
        # Compute connected components for the new drusen
        cca, numDrusen = sc.ndimage.measurements.label( enfaceDrusen )
        
        self.cca[y,x]=0
        maxLabel=max(np.unique(self.cca))+1
        self.idCount=numDrusen-1+self.idCount
        
        labels=np.unique(cca)
        brightmap=np.empty(self.enfaceProjection.shape)
        
        hs=list()
        sizes=list()
        brightness=list()
        self.height[np.where(mask==1)]=0.
        del self.sortedLabels
        self.sortedLabels=list()
        
        for l in self.drusenHeights.keys():
            hs.append(self.drusenHeights[l][0])
            sizes.append(self.drusenSizes[l][0])
            brightness.append(self.drusenBrightnesses[l][0])
            self.sortedLabels.append(l)
            
        for l in labels:
            brightmap.fill(0.)
            y,x=np.where(cca==l) # Extract certain component
            newl=maxLabel+l
            
            brightmap[y,x]=1
            denom=np.sum(brightmap)
            brightmapTmp=sc.ndimage.morphology.binary_dilation(brightmap,\
                                                      iterations=1).astype(int)
            brightmap=brightmapTmp-brightmap
            if(denom>0 and np.sum(brightmap)>0):
                innerVal=np.sum(self.enfaceProjection[y,x])/float(denom)
                bondVal=np.sum(self.enfaceProjection[np.where(brightmap==1)])/\
                                                       float(np.sum(brightmap))
                dbrightness=innerVal-bondVal
            else:
                dbrightness=0.
            maxHeight=np.max(self.height[y,x])
            dsize=np.sum(self.height[y,x])
            if(maxHeight>0):
                currLabel=newl
                self.cca[y,x]=newl
                self.drusenHeights[newl]=[maxHeight,y,x]
                self.drusenSizes[newl]=[dsize,y,x]
                self.drusenBrightnesses[newl]=[dbrightness,y,x]
                self.allDrusen.add(newl)
                self.sortedLabels.append(newl)
                hs.append(maxHeight)
                sizes.append(dsize)
                brightness.append(dbrightness)
            else: # Background
                self.background=[newl,y,x]
                
        self.compute_height(drusen)
        indx=np.argsort(hs)
        sortedArray=np.asarray(self.sortedLabels)[indx]
        sortedArray=sortedArray[::-1]
        self.sortedLabelsHeight=np.copy(sortedArray)
        indx=np.argsort(sizes)
        sortedArray=np.asarray(self.sortedLabels)[indx]
        sortedArray=sortedArray[::-1]
        self.sortedLabelsSize=np.copy(sortedArray)
        indx=np.argsort(brightness)
        sortedArray=np.asarray(self.sortedLabels)[indx]
        self.sortedLabelsBrightness=np.copy(sortedArray)
        
        self.update_sorting_method()
        self.prevIndex=-1
        self.currentIndex=0
        self.next_drusen_id()
        self.previous_drusen_id()
