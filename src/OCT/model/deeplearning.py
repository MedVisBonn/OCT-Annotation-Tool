# -*- coding: utf-8 -*-
"""
Created in 2018

@author: Shekoufeh Gorgi Zadeh
"""

import os
import imp
import sys
import numpy as np
import scipy as sc
from PyQt4 import QtGui
from skimage import data
import skimage.graph as skg
from skimage import filters
from matplotlib import pyplot as plt

try:
    imp.find_module('caffe')
    caffeFound = True
except ImportError:
    caffeFound = False
    
if( caffeFound ):   
    import caffe

#==============================================================================
# General functions used for image trasfomation and shortest path finding    
#==============================================================================
def im2double(img):
    return (img.astype('float64') ) / 255.0

def permute(A, dim):
    return np.transpose( A , dim )
        
def transform_score_image_sqrt(scoreImg, gamma=1.0):
    maxPerCol = np.max(scoreImg, axis=0)
    maxPerCol[np.where(maxPerCol==0.)] = 1e-10
    maxImg = np.tile(maxPerCol, scoreImg.shape[0])
    maxImg = maxImg.reshape(scoreImg.shape)
    maxImg[np.where(maxImg==0.0)]=1e-10
    return np.sqrt(maxImg-scoreImg)
    
def transform_score_image(scoreImg, gamma=1.0):
    nom=np.copy(scoreImg)
    nom[np.where(nom==0)]=1.e-10
    maxPerCol = np.max(nom, axis=0)
    
    # Use max all over the score
    maxScore=np.max(scoreImg)
    maxPerCol.fill(maxScore)
    
    maxPerCol[np.where(maxPerCol==0.)] = 1.e-10
    maxImg = np.tile(maxPerCol, nom.shape[0])
    maxImg = maxImg.reshape(nom.shape)
    maxImg[np.where(maxImg==0.0)]=1e-10
    return -1.0 * gamma * np.log10(np.divide(nom,maxImg))

def shortest_path_in_score_image(scoreImg,returnPath=False,yLength=1.):
    mcp = skg.MCP_Geometric(scoreImg,sampling=(yLength,1.),\
        offsets=[(1,0),(0,1),(-1,0),(1,1),(-1,1)])
    starts = np.zeros((scoreImg.shape[0],2))
    starts[:,0]=np.arange(scoreImg.shape[0])
    ends = np.zeros((scoreImg.shape[0],2))
    ends.fill(scoreImg.shape[1]-1)
    ends[:,0]=np.arange(scoreImg.shape[0])
    cumCosts, trace = mcp.find_costs(starts=starts,ends=ends)
    
    yMinEnd = np.argmin(cumCosts[:,-1])
    minPath = mcp.traceback([yMinEnd,scoreImg.shape[1]-1])
    p = np.array(minPath).T
    pathImg=np.zeros(scoreImg.shape,dtype='int')
    pathImg[p[0],p[1]]=1.
    if(returnPath):
        return pathImg,p
    return pathImg

def forbid_area_in_score_image(score,boundary):
    score[boundary==1]=np.inf
    return score
    
#==============================================================================
# Use a CNN to perform retinal layer segmentation 
#==============================================================================
class DeepLearningLayerSeg:

    def __init__(self,octScan):
        self.octScan=octScan
        self.uncertaintyValues=list()
        self.entropyVals=list()
        self.probabilityVals=list()
        self.entropyValsPerScan=list()
        self.probabilityValsPerScan=list()
        self.set_network_info()
        self.padInput=(((self.d4a_size *2 +2 +2)*2 +2 +2)*2 +2 +2)*2 +2 +2
        self.padOut=((((self.d4a_size -2 -2)*2-2 -2)*2-2 -2)*2-2 -2)*2-2 -2
        self.yLength=5.
        self.entropyMethod='probFreq'
        
        for p in sys.path:
            if( p == self.caffePath ):
                sys.path.remove(p)
                break
        sys.path.append(self.caffePath)
        
    def set_yLength(self,val):
        self.yLength=val
        
    def set_network_info(self):
        sett=self.octScan.controller.get_network_info()
        
        self.netPath=sett['netPath']
        self.processor=sett['processor']
        self.processorId=sett['processorId']
        self.normImage=sett['normImage']
        self.scaleImage=sett['scaleImage']
        self.zeroCenter=sett['zeroCenter']
        self.numOfTiles=sett['numOfTiles']
        self.caffePath =sett['caffePath']
        self.trainModelFile=sett['trainModelFile']
        self.modelFile=sett['modelFile']
        self.downSampleFactor=sett['downSampleFactor']
        self.d4a_size=sett['d4a_size']
        return    
        
    def set_uncertainties(self,uncertainties,sliceNumZ):
        if(uncertainties[0] is None):
            pass
        else:
            self.uncertaintyValues[sliceNumZ]=uncertainties[0]
            self.entropyVals[sliceNumZ]=uncertainties[1]
            self.probabilityVals[sliceNumZ]=uncertainties[2]
            
        
    def get_yLength(self):
        return self.yLength
        
    def get_uncertainties(self,sliceNumZ):
        if(len(self.uncertaintyValues)>0):
            return (self.uncertaintyValues[sliceNumZ],\
                self.entropyVals[sliceNumZ],self.probabilityVals[sliceNumZ])
        else:
            return (None,None,None)
            
    def get_uncertainties_per_bscan_at(self,sliceNumZ):
        if(len(self.entropyValsPerScan)>0):
            return (self.entropyValsPerScan[sliceNumZ],\
                self.probabilityValsPerScan[sliceNumZ])
        return (None,None)
        
    def get_uncertainties_per_bscan(self):
        if(len(self.entropyValsPerScan)>0):
            return (self.entropyValsPerScan,self.probabilityValsPerScan)
        return (None,None)
        
    def get_layer_seg_from_deepnet(self,scans):
        if(caffeFound):
            # Initial steps, scan preparation
            scans=self.preprocess_data(scans)
            
            # Do image tiling / Feed into the network
            scores=self.tiled_forward(scans)
        
            # Keep probability images
            probMaps=self.convert_scores_to_probabilities(scores)
            self.octScan.set_prob_maps(probMaps)
           
            # Shortest path
            shortestPaths= self.update_shortest_path()
            return shortestPaths
        else:
            self.octScan.controller.warn_cannot_open_network()
            return None
            
        
    def preprocess_data(self,indata):
        if(self.normImage):
              indata=indata-np.min(indata)
              if(np.max(indata)!= 0.0):
                  indata = indata / np.max(indata)
              for t in range(data.shape[2]): 
            	slice_t = data[:,:,t]
            	slice_t = slice_t - np.median(slice_t)
            	indata[:,:,t] = slice_t
    
        if(self.scaleImage!= 1):
          indata = sc.misc.imresize( indata, self.scaleImage*100, 'bilinear')

        indata = np.reshape( indata.astype('float32'),\
               [indata.shape[0], indata.shape[1], 1, indata.shape[2]])
        indata=np.transpose(indata,(1,0,2,3))
        return indata
        
        
    def tiled_forward(self,indata):
        
        debug=False
        #  compute input and output sizes (for v-shaped 4-resolutions network)
        a = np.asarray([indata.shape[0], np.ceil(indata.shape[1])/self.numOfTiles])
        b = self.padOut
        c = self.downSampleFactor
        d4a_size = np.ceil((a - b)/c)
        input_size =self.downSampleFactor*d4a_size +self.padInput
        output_size=self.downSampleFactor*d4a_size +self.padOut   
        #  create padded volume mit maximal border   
        border = np.round(input_size-output_size)/2
        paddedFullVolume = np.zeros((int(indata.shape[0] + 2*border[0]),\
                                     int(indata.shape[1] + 2*border[1]),\
                                     int(indata.shape[2]),\
                                     int(indata.shape[3])),dtype='float32',order='F')
        
        paddedFullVolume[int(border[0]):int(border[0]+indata.shape[0]),\
                         int(border[1]):int(border[1]+indata.shape[1]),\
                         :, : ] = indata
    
        #  create Network with fitting dimensions
        
        net_path = self.netPath
        fid = open(os.path.join(net_path,self.trainModelFile), 'rb')
        trainPrototxt = fid.read()
        fid.close()
        model_def_file = 'atmp-test.prototxt'
        fid = open(os.path.join(net_path,model_def_file),'w');
 
        fid.write('input: "data"\n')
   
        fid.write('input_dim: 1\n')
        fid.write('input_dim: '+str(int(indata.shape[2]))+'\n')
        fid.write('input_dim: '+str(int(input_size[1]))+'\n') 
        fid.write('input_dim: '+str(int(input_size[0]))+'\n')

        fid.write('state: { phase: TEST }\n')
        fid.write(trainPrototxt)
        fid.close()
   
        progressVal=self.octScan.get_progress_val()
        self.octScan.set_progress_val(progressVal+2)
        self.octScan.update_progress_bar()
        
        caffe.set_device(self.processorId)
        net = caffe.Net( os.path.join(net_path , model_def_file),\
            os.path.join(net_path,self.modelFile),caffe.TEST)
        
        progressVal=self.octScan.get_progress_val()
        self.octScan.set_progress_val(progressVal+2)
        self.octScan.update_progress_bar()
        
        if(self.processor== 'gpu'):
            caffe.set_mode_gpu()   
        else:
            caffe.set_mode_cpu()

       #  do the classification (tiled)
        per_data = (3,2,1,0)
        per_scor = (2,1,0)
        scores = []
        
        progressStep=60/float(indata.shape[3])
        for num in range(indata.shape[3]):
              progressVal=self.octScan.get_progress_val()
              self.octScan.set_progress_val(progressVal+progressStep)
              self.octScan.update_progress_bar()
              validReg = [0,0]
              yi = 0
              # crop input data
              for yi in range(self.numOfTiles):
                
                    paddedInputSlice = np.zeros((int(input_size[0]),\
                        int(input_size[1]),indata.shape[2],1),dtype='float32',\
                        order='C')
                    validReg[0] = int(min(input_size[0],paddedFullVolume.shape[0]))
                    validReg[1] = int(min(input_size[1],paddedFullVolume.shape[1]-\
                                yi*output_size[1]))
                    
                    paddedInputSlice[0:validReg[0],0:validReg[1],:,0] =\
                       paddedFullVolume[0:validReg[0], int(yi*output_size[1]):int(\
                       yi*output_size[1]+validReg[1]), :, num]
                    if(debug):
                        for layer_name, blob in net.blobs.iteritems():
                            print layer_name + '\t' + str(blob.data.shape)
                        
                        for layer_name, param in net.params.iteritems():
                            print layer_name + '\t' + str(param[0].data.shape),\
                                str(param[1].data.shape)
        
                    scores_caffe = net.forward(data=(np.transpose(\
                        paddedInputSlice,per_data)))
                    scoreSlice = scores_caffe['score'][0,:,:,:] 
                    scoreSlice = np.transpose(scoreSlice,per_scor)
        
                    labels = np.argmax(scoreSlice, axis=2)
                    labels = labels.squeeze()
                    labels = np.transpose(labels, (1,0))
          
                    if( num==0 and yi==0):
                    	  nClasses = scoreSlice.shape[2]
                    	  scores = np.zeros((indata.shape[0], indata.shape[1],\
                           nClasses,indata.shape[3]))
              
                    validReg[0] = int(min(output_size[0], scores.shape[0]))
                    validReg[1] = int(min(output_size[1], scores.shape[1] -\
                        yi*output_size[1]))
                    scores[0:validReg[0], int(yi*output_size[1]):int(\
                        yi*output_size[1]+validReg[1]),:,num] =\
                        scoreSlice[0:validReg[0],0:validReg[1],:]	

        if(self.scaleImage!= 1):
            scores = sc.misc.imresize( scores, (indata.shape[0],\
                indata.shape[1]), 'bilinear')
        return scores
    
    def convert_scores_to_probabilities(self,scores):
        probs = np.exp(scores)
        
        progressStep=10/float(probs.shape[3])
        
        for jj in range(probs.shape[3]):
            
            progressVal=self.octScan.get_progress_val()
            self.octScan.set_progress_val(progressVal+progressStep)
            self.octScan.update_progress_bar()
        
            meanImg = np.sum(probs[:,:,:,jj], axis=2)
            for kk in range(probs.shape[2]):
                probs[:,:,kk,jj] = probs[:,:,kk,jj]/meanImg
        return probs
        
    def compute_segmentation_uncertainty(self,certainSlices):
        if(len(self.uncertaintyValues)>0):
            for c in certainSlices:
                self.uncertaintyValues[c]=0.05
                self.probabilityVals[c]=0.05
                self.entropyVals[c]=0.05
            return
        
        
        
        self.octScan.controller.show_progress_bar()

        probMaps=self.octScan.get_prob_maps()
        layers=self.octScan.get_layers()
        if(probMaps.shape[0]!=layers.shape[0]):
            #Flip
            probMaps=np.transpose(probMaps,(1,0,2,3))
        progressVal=0
        progressStep=98/float(probMaps.shape[3])
        entropyVals=list()
        for i in range(probMaps.shape[3]):
            progressVal+=progressStep
            self.octScan.controller.set_progress_bar_value(progressVal)
            
            p1=np.copy(probMaps[:,:,1,i])
            p2=np.copy(probMaps[:,:,2,i])

            c1=np.copy(p1)
            c2=np.copy(p2)
            
            l1=np.copy(p1)
            l2=np.copy(p2)
            
            c2[np.where(layers[:,:,i]!=255)]=0.
            c1[np.where(layers[:,:,i]!=127)]=0.
            
            p1[1:-1,:]=np.diff(p1,n=2,axis=0)
            p2[1:-1,:]=np.diff(p2,n=2,axis=0)
            
            t1=filters.threshold_otsu(p1)
            t2=filters.threshold_otsu(p2)
            
            p2[np.where(layers[:,:,i]!=255)]=0.
            p1[np.where(layers[:,:,i]!=127)]=0.

            # For each
            ue1=list()
            ue2=list()
            ue1m10=list()
            ue2m10=list()
            up1=list()
            up2=list()
            for j in range(probMaps.shape[1]):
                
                x11=p1[np.where(p1[:,j]>t1),j][0]
                x21=p2[np.where(p2[:,j]>t2),j][0]
                
                x1=p1[:,j]
                x2=p2[:,j]
                
                n1=float(len(x1))
                n2=float(len(x2))
                
                n11=float(len(x11))
                n21=float(len(x21))
                if(self.entropyMethod!='probFreq'):
                    e1=(np.exp(-(self.entropy(l1[:,j]))**3.))
                    e2=(np.exp(-(self.entropy(l2[:,j]))**3.))
                    
                else:
                    e1=(np.exp(-(self.entropy(l1[:,j]))**1.))*1.
                    e2=(np.exp(-(self.entropy(l2[:,j]))**1.))*1.
                    e1m10=e1*10.
                    e2m10=e2*10.
                    
                ue1.append(e1)
                ue2.append(e2)
                    
                ue1m10.append(e1m10)
                ue2m10.append(e2m10)
                    
                if(n1==0):
                    up1.append(0.)
                elif(n1==1):
                    up1.append(1.)
                else:
                    max1=max(x1)
                    unc1=np.sum(np.power(max1-x1,1))
                    unc1=((unc1/(n1-1))+(np.exp(-np.sqrt(n11-1))))/2.
                    unc1=np.sum(np.abs(x1**2))+np.sum(np.abs(c1[:,j]))
                    xx=np.where(c1[:,j]>0)[0]

                    if(len(xx)==1):
                        unc1=c1[xx[0],j]
                        up1.append(unc1)
                    elif(len(xx)>1):
                        unc1=c1[xx[0],j]
                        up1.append(unc1)
                    else:
                        up1.append(1.0)
                if(n2==0):
                    up2.append(0.)
                elif(n2==1):
                    up2.append(1.)
                else:
                    max2=max(x2)
                    unc2=np.sum(np.power(max2-x2,1))
                    unc2=((unc2/(n2-1))+(np.exp(-np.sqrt(n21-1))))/2.
                    unc2=np.sum(np.abs(x2**2))+np.sum(np.abs(c2[:,j]))
                    xx=np.where(c2[:,j]>0)[0]

                    if(len(xx)==1):
                        unc2=c2[xx[0],j]
                        up2.append(unc2)
                    elif(len(xx)>1):
                        unc2=c2[xx[0],j]

                        up2.append(unc2)
                    else:
                        up2.append(1.0)
            
            sig=2
            ue1=sc.ndimage.filters.gaussian_filter(ue1,sig)
            ue2=sc.ndimage.filters.gaussian_filter(ue2,sig)
            ue1m10=sc.ndimage.filters.gaussian_filter(ue1m10,sig)
            ue2m10=sc.ndimage.filters.gaussian_filter(ue2m10,sig)
            bnd=50
            ue1[:bnd]=1.
            ue1[-bnd:]=1.
            ue2[:bnd]=1.
            ue2[-bnd:]=1.
            ue1m10[:bnd]=1.
            ue1m10[-bnd:]=1.
            ue2m10[:bnd]=1.
            ue2m10[-bnd:]=1.
            up1=sc.ndimage.filters.gaussian_filter(up1,sig)
            up2=sc.ndimage.filters.gaussian_filter(up2,sig)
            
            up1[:bnd]=1.
            up1[-bnd:]=1.
            up2[:bnd]=1.
            up2[-bnd:]=1.
            
            mmethod='min'
            if(mmethod=='quartile'):
                u1eq=np.percentile(ue1,25)
                ue1=ue1[np.where(ue1<u1eq)]
                u1eqm10=np.percentile(ue1m10,25)
                ue1m10=ue1m10[np.where(ue1m10<u1eqm10)]
                
                u2eq=np.percentile(ue2,25)
                ue2=ue2[np.where(ue2<u2eq)]
                u2eqm10=np.percentile(ue2m10,25)
                ue2m10=ue2m10[np.where(ue2m10<u2eqm10)]
                
                u1pq=np.percentile(up1,25)
                up1=up1[np.where(up1<u1pq)]
                
                u2pq=np.percentile(up2,25)
                up2=up2[np.where(up2<u2pq)]
                
                self.uncertaintyValues.append(min(min((ue1+up1)/2.),\
                    min((ue2+up2)/2.)))
                self.entropyVals.append(1.-min(np.average(u1eqm10),\
                    np.average(u2eqm10)))
                entropyVals.append(1.-min(np.average(u1eq),np.average(u2eq)))
                self.probabilityVals.append(1.-min(np.average(u1pq),\
                    np.average(u2pq)))
            elif(mmethod=='average'):
                self.uncertaintyValues.append(min(min((ue1+up1)/2.),\
                    min((ue2+up2)/2.)))
                self.entropyVals.append(1.-min(np.average(ue1m10),\
                    np.average(ue2m10)))
                entropyVals.append(1.-min(np.average(ue1),np.average(ue2)))
                self.probabilityVals.append(1.-min(np.average(up1),\
                    np.average(up2)))
            else:
                self.uncertaintyValues.append(min(min((ue1+up1)/2.),\
                    min((ue2+up2)/2.)))
                self.entropyVals.append(1.-min(min(ue1m10),min(ue2m10)))
                entropyVals.append(1.-min(min(ue1),min(ue2)))
                self.probabilityVals.append(1.-min(min(up1),min(up2)))
                self.entropyValsPerScan.append([ue1m10,ue2m10])
                self.probabilityValsPerScan.append([up1,up2])
                
        for c in certainSlices:
            self.uncertaintyValues[c]=0.05
            self.probabilityVals[c]=0.05
            self.entropyVals[c]=0.05
            entropyVals[c]=0.05
        np.savetxt(os.path.join(self.octScan.get_current_path(),\
            'entropy-'+mmethod+'.txt'),entropyVals)
        np.savetxt(os.path.join(self.octScan.get_current_path(),\
            'probability-'+mmethod+'.txt'),self.probabilityVals)

        self.octScan.set_progress_val(100)
        self.octScan.update_progress_bar()
        self.octScan.controller.hide_progress_bar()
        
    # Input a pandas series 
    def entropy(self,data):
        if(self.entropyMethod!='probFreq'):        
            data=data*1000
            data=np.abs(data.astype(int))
            data[data<0]=0.
            hist=np.bincount(data)
            data2=hist[np.where(hist>0)]
            p_data= data2/float(len(data)) # calculates the probabilities
        else:
            p_data=data/(np.sum(data))
            p_data=p_data[np.where(p_data>0)]
        entropy=sc.stats.entropy(p_data)  # input probabilities to get the entropy
        return entropy         
        
    def show_image(self, image, block = True ):
        plt.imshow( image, cmap = plt.get_cmap('gray'))
        plt.show(block)
        QtGui.QApplication.processEvents()    
        
    def show_images(self,images,r,c,titles=[],d=0,save_path = "",block = True):
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
                plt.imshow( img , cmap = plt.get_cmap('gray'))
            i += 1
        
        plt.show(block)
        QtGui.QApplication.processEvents()    
        
    def find_shortest_path(self,probMaps,fromDisk=False):
        if(fromDisk):
            labels=np.empty((probMaps.shape[0],probMaps.shape[1],probMaps.shape[3]))
        else:
            labels=np.empty((probMaps.shape[1],probMaps.shape[0],probMaps.shape[3]))
        
        progressStep=20/float(probMaps.shape[3])
        for ii in range(probMaps.shape[3]): 
            
            progressVal=self.octScan.get_progress_val()
            self.octScan.set_progress_val(progressVal+progressStep)
            self.octScan.update_progress_bar()
            
            if(fromDisk):
                score1 = probMaps[:,:,1,ii]+probMaps[:,:,3,ii]
                score2 = probMaps[:,:,2,ii]+probMaps[:,:,3,ii]
            else:
                # Get the score for labels 1 and 2
                score1 = permute(probMaps[:,:,1,ii], [1,0])+probMaps[:,:,3,ii].T
                score2 = permute(probMaps[:,:,2,ii], [1,0])+probMaps[:,:,3,ii].T

            score1 = transform_score_image(score1, gamma=1.0)
            score2 = transform_score_image(score2, gamma=1.0)

            path1,p1 = shortest_path_in_score_image(score1,True)
            score2=forbid_area_in_score_image(score2,path1)
            path2 =shortest_path_in_score_image(score2)
            labels[:,:,ii] = path2 * 2. + path1
            
        if(fromDisk):  
            labels[np.where(labels==2)]=255.
            labels[np.where(labels==1)]=127.
            return labels
        else:
            return labels
            
    def gaussian(self,x,sigma):
        return (1./(sigma*np.sqrt(2.*np.pi)))*np.exp(-(x**2/(2.*sigma**2)))
        
    def update_shortest_path_in_slice(self,sliceNum,yLength=1.):
            probMaps=self.octScan.get_prob_maps()
            
            h,w=probMaps[:,:,1,sliceNum].shape
            
            tmpjs1=np.where(np.sum(probMaps[:,:,1,sliceNum],axis=0)==1.)
            iss1=list()
            js1=list()
            for j in tmpjs1[0]:
                ypos=np.where(probMaps[:,j,1,sliceNum]==1.)
                if(len(ypos[0])==1):
                    iss1.append(ypos[0][0])
                    js1.append(j)
            
            tmpjs2=np.where(np.sum(probMaps[:,:,2,sliceNum],axis=0)==1.)
            
            iss2=list()
            js2=list()
            for j in tmpjs2[0]:
                ypos=np.where(probMaps[:,j,2,sliceNum]==1.)
                if(len(ypos[0])==1):
                    iss2.append(ypos[0][0])
                    js2.append(j)
            score1 = probMaps[:,:,1,sliceNum]+probMaps[:,:,3,sliceNum]
            score2 = probMaps[:,:,2,sliceNum]+probMaps[:,:,3,sliceNum]
            
            score1 = transform_score_image(score1, gamma=1.0)
            score2 = transform_score_image(score2, gamma=1.0)

            largeNum=1000000
            score1[:,js1]=largeNum
            score1[iss1,js1]=0.
            score2[:,js2]=largeNum
            score2[iss2,js2]=0.       

            path1,p1 = shortest_path_in_score_image(score1,True,yLength)
            score2=forbid_area_in_score_image(score2,path1)
            path2 =shortest_path_in_score_image(score2,yLength=yLength)

            labels = path2 * 2. + path1
            
            labels[np.where(labels==2)]=255.
            labels[np.where(labels==1)]=127.
            
            return labels
      
    def update_probability_image(self,i,j,sliceNum,layerName,smoothness):
        eps=0.
        probMaps=self.octScan.get_prob_maps()
        layers=self.octScan.get_layers()
        if(layerName=='RPE'):
            probMaps[:,j,3,sliceNum]=eps
            probMaps[:,j,2,sliceNum]=eps
            probMaps[i,j,2,sliceNum]=1.
            
        elif(layerName=='BM'):
            probMaps[:,j,3,sliceNum]=eps
            probMaps[:,j,1,sliceNum]=eps
            probMaps[i,j,1,sliceNum]=1.
            
        layers[:,:,sliceNum]=\
            self.update_shortest_path_in_slice(sliceNum,smoothness)
        self.octScan.set_prob_maps(probMaps)
        self.octScan.set_layers(layers)
        return layers[:,:,sliceNum]
        
    def update_shortest_path(self,permuteAxis=False):
        probMaps=self.octScan.get_prob_maps()
        if(permuteAxis):
            return self.find_shortest_path(probMaps,fromDisk=True)
        return self.find_shortest_path(probMaps)