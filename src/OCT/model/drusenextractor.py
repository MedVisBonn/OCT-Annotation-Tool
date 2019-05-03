# -*- coding: utf-8 -*-
"""
Created in 2018

@author: Shekoufeh Gorgi Zadeh
"""
import numpy as np

#==============================================================================
# This class uses RPE and BM layer location to automatically segment out the 
# drusen.
#==============================================================================
class DrusenSeg:
    
    def __init__(self,controller=None):
        self.drusen=list()
        self.controller=controller
        self.lambdaVec=[1.,1.,0.1*2,0.1**4] 
        self.polyFitType='Reguilarized' # 'Reguilarized' or 'None'
        
    def get_drusen_seg_polyfit(self,layers):
        """
        For each segmentation map, estimates the normal RPE and then finds the
        area btw the RPE and the normal RPE layer as the drusen.
        
        layers: 3D numpy array that contains RPE and BM layer segmentation maps
        for all the OCT volume B-scans. Third dimension is the B-scan index.
        """
        self.drusen=np.zeros(layers.shape)
        pStep=25/float(max(1,layers.shape[2]))
        
        for i in range(layers.shape[2]):
            self.controller.update_progress_bar_value(pStep)
            l=layers[:,:,i]
            y,x= self.get_RPE_layer(l)
            y_n, x_n = self.normal_RPE_estimation(l,useWarping=True)
            vr = np.zeros((l.shape[1]))
            vr[x] = y
            vn = np.zeros((l.shape[1]))
            vn[x_n] = y_n
            mask=np.zeros(l.shape)
            try:
                h,w=l.shape
                y[np.where(y>=h)]=h-1
                x[np.where(x>=w)]=w-1
                y_n[np.where(y_n>=h)]=h-1
                x_n[np.where(x_n>=w)]=w-1
                mask[y,x]=1
                mask[y_n,x_n]=2
                self.drusen[:,:,i]=self.find_area_btw_RPE_normal_RPE( mask )
            except:
                print "Error in get_drusen_seg_polyfit function"
        return self.drusen
        
    def get_RPE_layer(self,img):
        """
        Extract the pixel-wise RPE location in the img (layer segmentation map).
        """
        y = []
        x = []
        
        if( len(np.unique(img)) == 4 ):
            tmp = np.zeros(img.shape)
            tmp[np.where(img==170)] = 255
            tmp[np.where(img==255)] = 255
            y, x = np.where(tmp==255)
          
        else:
            y, x = np.where(img==255)
            
        tmp = np.zeros(img.shape)
        tmp[y,x] = 255
        y,x = np.where(tmp>0)
        return y, x
        
    def get_BM_location(self,seg_img ):
        """
        Extract the pixel-wise BM location in the img (layer segmentation map).
        """
        y = []
        x = []
        tmp = np.copy(seg_img)
        if( np.sum(seg_img)==0.0):
            return y, x
        if( 85 in seg_img ):
            tmp2 = np.zeros(tmp.shape)
            tmp2[np.where(tmp==170)] = 255
            tmp2[np.where(tmp==85)] = 255
            y, x = np.where(tmp2==255)
        elif(170 in seg_img):
            tmp2 = np.zeros(tmp.shape)
            tmp2[np.where(tmp==170)] = 255
            tmp2[np.where(tmp==127)] = 255
            y, x = np.where(tmp2==255)
        else:
            y, x = np.where(tmp==127)
        return y, x  
        
    def get_RPE_location(self,seg_img ):
        """
        Extract the pixel-wise RPE location in the img (layer segmentation map).
        """
        y = []
        x = []
        tmp = np.copy(seg_img)
        if( np.sum(seg_img)==0.0):
            return y, x
        if( len(np.unique(tmp)) == 4 ):
            tmp2 = np.zeros(tmp.shape)
            tmp2[np.where(tmp==170)] = 255
            tmp2[np.where(tmp==255)] = 255
            y, x = np.where(tmp2==255)
          
        else:
            y, x = np.where(tmp==255)
        return y, x
        
    def poly_fit(self,x,y,degree=3,it=3,farDiff=5,s_ratio = 1):
        """
        Fit a polynomial to the given layer as x and y locations. Use the given
        input degree, iterations to fit a polynomial and farDiff to ignore pixels
        that are too far away while polynomial fitting on the layer.
        """
        ignoreFarPoints=False
        tmpx = np.copy(x)
        tmpy = np.copy(y)
        origy = np.copy(y)
        origx = np.copy(x)
        finalx = np.copy(tmpx)
        finaly = tmpy
        
        for i in range(it):
            if( s_ratio > 1 ):
                s_rate = len(tmpx)/s_ratio
                rand   = np.random.rand(s_rate) * len(tmpx)
                rand   = rand.astype('int')            
                
                sx = tmpx[rand]
                sy = tmpy[rand]
                if(self.polyFitType=='None'):
                    z = np.polyfit(sx, sy, deg = degree)
                else:
                    z = self.compute_reguilarized_fit(sx, sy, deg = degree)
            else:
                if(self.polyFitType=='None'):
                    z = np.polyfit(tmpx, tmpy, deg = degree)
                else:
                    z = self.compute_reguilarized_fit(tmpx, tmpy, deg = degree)
            p = np.poly1d(z)
            
            new_y = p(finalx).astype('int')
            if( ignoreFarPoints ):
                tmpx = []
                tmpy = []
                for i in range(0,len(origx)):
                  diff=new_y[i]-origy[i]
                  if diff<farDiff:
                      tmpx.append(origx[i])
                      tmpy.append(origy[i])
            else:
                tmpy = np.maximum(new_y, tmpy)
            finaly = new_y
               
        return finaly, finalx
        
    def compute_reguilarized_fit(self,x,y,deg):
        resMat=np.zeros((deg+1,deg+1))
        for d in range(deg+1):
            z=np.polyfit(x, y, deg = d)            
            for i in range(len(z)):
                resMat[d,-1-i]=z[-1-i]
#        print "======="
#        print resMat
        weightedAvg=np.average(resMat,axis=0,weights=self.lambdaVec)
#        print weightedAvg
        return weightedAvg
        
    def normal_RPE_estimation(self,layer,degree = 3,it = 3,s_ratio = 1, \
                            farDiff = 5, ignoreFarPoints=True, returnImg=False,\
                            useBM = False,useWarping=True,xloc=[],yloc=[]):   
        """
        Given the RPE layer, estimate the normal RPE. By first warping the RPE
        layer with respect to the BM layer. Then fit a third degree polynomial
        on the RPE layer, and warp the resulting curve back.
        """                       
        if(useWarping):
            y, x = self.get_RPE_location(layer)
            
            yn, xn = self.warp_BM(layer)
            return yn, xn
            
        if( useBM ):
            y_b, x_b = self.get_BM_location( layer ) 
            y_r, x_r = self.get_RPE_location( layer )  
            
            if(self.polyFitType=='None'):
                z = np.polyfit(x_b, y_b, deg = degree)        
            else:
                z = self.compute_reguilarized_fit(x_b, y_b, deg = degree)            
                
            p = np.poly1d(z)        
            y_b = p(x_r).astype('int')
            
            prev_dist = np.inf
            offset = 0
            for i in range(50):
                 newyb = y_b - i 
                 diff  = np.sum(np.abs(newyb-y_r))
                 if( diff < prev_dist ):
                      prev_dist = diff
                      continue
                 offset = i
                 break
            if( returnImg ):
                img = np.zeros(layer.shape)
                img[y_b-offset, x_r] = 255.0
                return y_b-offset, x_r, img
            return y_b-offset, x_r
            
        tmp = np.copy(layer)
        y = []
        x = []
        if(xloc==[] or yloc==[]):
            if( np.sum(layer)==0.0):
                return y, x
            if( len(np.unique(tmp)) == 4 ):
                tmp2 = np.zeros(tmp.shape)
                tmp2[np.where(tmp==170)] = 255
                tmp2[np.where(tmp==255)] = 255
                y, x = np.where(tmp2==255)
              
            else:
                y, x = np.where(tmp==255)
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
            if( s_ratio > 1 ):
                s_rate = len(tmpx)/s_ratio
                rand   = np.random.rand(s_rate) * len(tmpx)
                rand   = rand.astype('int')            
                
                sx = tmpx[rand]
                sy = tmpy[rand]
                if(self.polyFitType=='None'):
                    z = np.polyfit(sx, sy, deg = degree)
                else:
                    z = self.compute_reguilarized_fit(sx, sy, deg = degree)
            else:
                if(self.polyFitType=='None'):
                    z = np.polyfit(tmpx, tmpy, deg = degree)
                else:
                    z = self.compute_reguilarized_fit(tmpx, tmpy, deg = degree)
            p = np.poly1d(z)
            
            new_y = p(finalx).astype('int')
            if( ignoreFarPoints ):
                tmpx = []
                tmpy = []
                for i in range(0,len(origx)):
                  diff=new_y[i]-origy[i]
                  if diff<farDiff:
                      tmpx.append(origx[i])
                      tmpy.append(origy[i])
            else:
                tmpy = np.maximum(new_y, tmpy)
            finaly = new_y
        if( returnImg ):
            return finaly, finalx, tmp
        
        return finaly, finalx
        
    def find_area_btw_RPE_normal_RPE(self, mask):
        """
        Find the area between two given layers in the mask image.
        """
        area_mask = np.zeros(mask.shape)
        for i in range( mask.shape[1] ):
            col = mask[:,i]
            v1  = np.where(col==1.0)
            v2  = np.where(col==2.0)
            v3  = np.where(col==3.0)
           
            v1 = np.min(v1[0]) if len(v1[0]) > 0  else -1
            v2 = np.max(v2[0]) if len(v2[0]) > 0  else -1
            v3 = np.min(v3[0]) if len(v3[0]) > 0  else -1
            
            if( v1 >= 0 and v2 >= 0 ):
                area_mask[v1:v2,i] = 1
        return area_mask
        
    def warp_BM(self,seg_img, returnWarpedImg=False ):
        """
        Warp seg_img with respect to the BM layer in the segmentation image. 
        Return the location of the warped RPE layer in the end. If 
        returnWarpedImg is set to True, return the whole warped seg_img.
        """
        h, w = seg_img.shape
        yr, xr = self.get_RPE_location( seg_img )
        yb, xb = self.get_BM_location( seg_img )
        rmask  = np.zeros((h, w), dtype='int')
        bmask  = np.zeros((h, w), dtype='int')
      
        rmask[yr, xr] = 255
        bmask[yb, xb] = 255
    
        vis_img = np.copy(seg_img)
        shifted = np.zeros(vis_img.shape)
        wvector = np.empty((w), dtype='int')
        wvector.fill(h-(h/2))
        nrmask = np.zeros((h,w), dtype='int')
        nbmask = np.zeros((h,w), dtype='int')
        
        zero_x =[]
        zero_part = False  
        last_nonzero_diff = 0
        for i in range(w):
            bcol = np.where(bmask[:,i]>0)[0]
            wvector[i] = wvector[i]-np.max(bcol) if len(bcol) > 0 else 0
            if( len(bcol) == 0  ):
                zero_part = True
                zero_x.append(i)
            if( len(bcol)>0 and zero_part ):
                diff = wvector[i]
                zero_part = False
                wvector[zero_x]=diff
                zero_x=[]
            if( len(bcol)>0):
                last_nonzero_diff = wvector[i]
            if( i == w-1 and zero_part):
                wvector[zero_x]=last_nonzero_diff

        # Where wvector is zero, set the displacing to the displaceing of a non zero
        # neighbour
        for i in range(w):
            nrmask[:, i] = np.roll(rmask[:,i], wvector[i])
            nbmask[:, i] = np.roll(bmask[:,i], wvector[i])
            shifted[:, i] = np.roll(vis_img[:,i], wvector[i])
        shifted_yr =[]   
        for i in range(len(xr)):
            shifted_yr.append(yr[i] + wvector[xr[i]])
        yn, xn = self.normal_RPE_estimation( rmask,it=5,useWarping=False,\
                    xloc=xr, yloc=shifted_yr )

        for i in range(len(xn)):
            yn[i] = yn[i] - wvector[xn[i]]

        if(returnWarpedImg):
            return shifted
        return yn,xn 
        
    def draw_layers_in_image(self,height,width,xrpe,yrpe,xbm,ybm):  
        """
        Given the RPE and BM layer locations, paint them in one single 
        segmentation map.
        """
        img=np.zeros((height,width))
        tmp=np.copy(img)
        img[yrpe,xrpe]=1.
        tmp[ybm,xbm]=1.
        join=img*tmp
        if(np.sum(join)>0.):
            yover,xover=np.where(join>0.)
            img=img*255.
            img[ybm,xbm]=127.
            img[yover,xover]=170.
        else:
            img=img*255.
            img[ybm,xbm]=127.
        return img
        
    def fill_inner_gaps(self,layer ):
        """
        If the segmentation layer is fragmented, connect the disconnected line
        segments.
        """
        prev=-1
        for j in range(layer.shape[1]):
            i=np.where(layer[:,j]>0)[0]   
            if(len(i)>0):
                if(prev!=-1 and abs(prev-i[0])>1):
                    layer[prev:i[0],j]=1
                    layer[i[0]:prev,j]=1
              
                prev=i[0]
            
        return layer
        
    def interpolate_layer_in_region(self,reg,reg2,by,ty,bx,tx,polyDegree,layerName):
        """
        In a given region, fit a polynomial on the given layer. 
        """
        finImg=np.copy(reg)
        if(layerName=='RPE'):
            yr,xr=self.get_RPE_location(reg)
            borderHint=np.zeros(reg.shape)
            borderHint[yr,xr]=1
            yr,xr=np.where(borderHint[1:-1,1:-1]>0)
            if(len(yr)>0):
                yn,xn=self.poly_fit(xr,yr,polyDegree)
                tmp=np.empty(reg.shape)
                tmp.fill(0)
                tmp[yn+1,xn+1]=1
                tmp[0,:]=borderHint[0,:]
                tmp[-1,:]=borderHint[-1,:]
                tmp[:,0]=borderHint[:,0]
                tmp[:,-1]=borderHint[:,-1]
                tmp=self.fill_inner_gaps(tmp)
                yn,xn=np.where(tmp>0)
                yb,xb=self.get_BM_location(reg)
                finImg=self.draw_layers_in_image(reg.shape[0],reg.shape[1],xn,yn,xb,yb)
            
        elif(layerName=='BM'):
            yb,xb=self.get_BM_location(reg)
            if(len(yb)>0):
                yn,xn=self.poly_fit(xb,yb,polyDegree)
                yr,xr=self.get_RPE_location(reg)
                finImg=self.draw_layers_in_image(reg.shape[0],reg.shape[1],xr,yr,xn,yn)
        return finImg
