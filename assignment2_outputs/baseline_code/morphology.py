"""morphology.py — AgriTech Assignment 1"""
import cv2, numpy as np
class MorphologyPipeline:
    def __init__(self, config):
        self.config=config; sz=config["morphological"]["kernel_size"]
        m={"ellipse":cv2.MORPH_ELLIPSE,"cross":cv2.MORPH_CROSS,"rect":cv2.MORPH_RECT}
        self.kernel=cv2.getStructuringElement(m.get(config["morphological"]["kernel_shape"],cv2.MORPH_ELLIPSE),(sz,sz))
    def apply_erosion(self,img,it=1): return cv2.erode(img,self.kernel,iterations=it)
    def apply_dilation(self,img,it=1): return cv2.dilate(img,self.kernel,iterations=it)
    def apply_opening(self,img,it=1):
        r=img.copy()
        for _ in range(it): r=cv2.morphologyEx(r,cv2.MORPH_OPEN,self.kernel)
        return r
    def apply_closing(self,img,it=1):
        r=img.copy()
        for _ in range(it): r=cv2.morphologyEx(r,cv2.MORPH_CLOSE,self.kernel)
        return r
    def remove_small_objects(self,img,min_size):
        nl,lbl,stats,_=cv2.connectedComponentsWithStats(img,8,cv2.CV_32S)
        m=np.zeros_like(img,dtype=np.uint8)
        for i in range(1,nl):
            if stats[i,cv2.CC_STAT_AREA]>=min_size: m[lbl==i]=255
        return m
    def watershed_segmentation(self,img,thresh=0.30):
        im=(img>0).astype(np.uint8)*255
        dist=cv2.distanceTransform(im,cv2.DIST_L2,5)
        cv2.normalize(dist,dist,0,1.0,cv2.NORM_MINMAX)
        _,mb=cv2.threshold(dist,thresh,1.0,cv2.THRESH_BINARY)
        _,markers=cv2.connectedComponents(mb.astype(np.uint8))
        markers=markers+1; markers[im==0]=0
        markers=cv2.watershed(cv2.cvtColor(im,cv2.COLOR_GRAY2BGR),markers)
        r=np.zeros_like(im); r[markers>1]=255; return r
    def refine_mask(self,image):
        c=self.config; ms=max(500,c["object_counting"]["min_seed_area"]//4)
        ws=c.get("watershed_thresh",0.30)
        s=self.remove_small_objects(image,ms)
        s=self.apply_closing(s,it=c["morphological"]["closing_iterations"])
        s=self.apply_opening(s,it=c["morphological"]["opening_iterations"])
        s=self.remove_small_objects(s,ms)
        if c["object_counting"]["use_watershed"]:
            nl,_,stats,_=cv2.connectedComponentsWithStats(s,8,cv2.CV_32S)
            if any(stats[i,cv2.CC_STAT_AREA]>c["object_counting"]["max_seed_area"] for i in range(1,nl)):
                s=self.watershed_segmentation(s,thresh=ws)
        return s
