"""counting.py — AgriTech Assignment 1"""
import cv2, numpy as np
class RegionProp:
    def __init__(self,l,a,c,b,p):
        self.label=l; self.area=a; self.centroid=(c[1],c[0])
        self.bbox=(b[1],b[0],b[1]+b[3],b[0]+b[2]); self.perimeter=p
class CountingPipeline:
    def __init__(self, config):
        self.config=config; oc=config["object_counting"]
        self.min_area=oc["min_seed_area"]; self.max_area=oc["max_seed_area"]
        self.min_circ=oc["min_circularity"]; self.max_ar=oc["max_aspect_ratio"]
        self.edge_buf=oc["edge_buffer"]
    def get_region_props(self,mask):
        nl,lbl,stats,cens=cv2.connectedComponentsWithStats(mask,8,cv2.CV_32S)
        props=[]
        for i in range(1,nl):
            cm=(lbl==i).astype(np.uint8); cts,_=cv2.findContours(cm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            p=cv2.arcLength(cts[0],True) if cts else 0
            props.append(RegionProp(i,stats[i,cv2.CC_STAT_AREA],cens[i],stats[i,:4],p))
        return props,lbl
    def count_seeds(self,mask,img_shape=None):
        regions,labeled=self.get_region_props(mask)
        valid,partial,oversized=[],[],[]
        for r in regions:
            ok,reason=self._valid(r,img_shape)
            if ok: valid.append(r)
            elif reason=="edge": partial.append(r)
            elif reason=="oversized": oversized.append(r)
        count=len(valid)+len(partial)//2
        ref=float(np.median([r.area for r in valid])) if valid else float(self.max_area)
        for r in oversized: count+=max(1,round(r.area/ref))
        return count,valid,labeled,partial,oversized
    def _valid(self,r,sh):
        if r.area<self.min_area: return False,"small"
        if r.area>self.max_area*2.0: return False,"oversized"
        if r.perimeter>0:
            if 4*np.pi*r.area/(r.perimeter**2)<self.min_circ: return False,"non_circular"
        mr,mc,xr,xc=r.bbox; ar=max(xr-mr,xc-mc)/(min(xr-mr,xc-mc)+1e-6)
        if ar>self.max_ar: return False,"elongated"
        if sh:
            h,w=sh[:2]; b=self.edge_buf
            if mr<=b or mc<=b or xr>=h-b or xc>=w-b: return False,"edge"
        return True,"valid"
    def visualize_counting(self,image,mask,regions,partial_regions=None,oversized_regions=None):
        viz=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB) if image.ndim==2 else image.copy()
        n=1
        for r in (regions or []):
            mr,mc,xr,xc=r.bbox; cv2.rectangle(viz,(mc,mr),(xc,xr),(50,100,255),3)
            y,x=r.centroid; cv2.circle(viz,(int(x),int(y)),6,(50,100,255),-1)
            cv2.putText(viz,str(n),(int(x)-10,int(y)+8),cv2.FONT_HERSHEY_SIMPLEX,0.9,(50,100,255),2); n+=1
        for r in (partial_regions or []):
            mr,mc,xr,xc=r.bbox; cv2.rectangle(viz,(mc,mr),(xc,xr),(255,165,0),2)
            y,x=r.centroid; cv2.putText(viz,"E",(int(x)-8,int(y)+6),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,165,0),2)
        ref=float(np.median([r.area for r in regions])) if regions else float(self.max_area)
        for r in (oversized_regions or []):
            mr,mc,xr,xc=r.bbox; cv2.rectangle(viz,(mc,mr),(xc,xr),(255,50,50),3)
            y,x=r.centroid; est=max(1,round(r.area/ref))
            cv2.putText(viz,f"~{est}",(int(x)-15,int(y)+8),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,50,50),2)
        return viz
