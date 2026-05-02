"""clustering.py — KMeans via cv2.kmeans; DBSCAN vectorised NumPy. NO sklearn."""
import cv2, numpy as np
from scipy.spatial import cKDTree
class ClusteringPipeline:
    REF_AREA = 25000.0
    def __init__(self, config): self.config = config
    def kmeans_segmentation(self, image):
        n,mi,ep = (self.config["clustering"]["kmeans"][k] for k in ["n_clusters","max_iter","epsilon"])
        pixels = image.reshape(-1,1).astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, mi, ep)
        _,labels,centers = cv2.kmeans(pixels,n,None,crit,10,cv2.KMEANS_RANDOM_CENTERS)
        labels = labels.reshape(image.shape); centers = centers.flatten()
        return np.uint8((labels==int(np.argmin(centers)))*255), labels
    def dbscan_segmentation(self, image):
        eps,ms = self.config["clustering"]["dbscan"]["eps"], self.config["clustering"]["dbscan"]["min_samples"]
        scale = 4; sh,sw = image.shape[0]//scale, image.shape[1]//scale
        small = cv2.resize(image,(sw,sh),interpolation=cv2.INTER_AREA)
        dy,dx = np.where(small < small.mean()-1.0*small.std())
        if len(dy)<ms: return np.zeros_like(image,dtype=np.uint8)
        coords = np.column_stack([dy,dx]).astype(np.float32)
        tree = cKDTree(coords); nbl = tree.query_ball_tree(tree,r=eps/scale)
        nc = np.array([len(x) for x in nbl]); is_core = nc>=ms
        labels = -np.ones(len(coords),dtype=np.int32); visited = np.zeros(len(coords),dtype=bool)
        cid = 0
        for si in np.where(is_core)[0]:
            if visited[si]: continue
            front = np.array([si])
            while front.size:
                visited[front]=True; labels[front]=cid
                cf = front[is_core[front]]
                new = (np.unique(np.concatenate([nbl[i] for i in cf])).astype(int) if len(cf)>0 else np.array([],dtype=int))
                front = new[~visited[new]]
            cid += 1
        mask = np.zeros_like(small,dtype=np.uint8); nn = labels>=0
        mask[coords[nn,0].astype(int),coords[nn,1].astype(int)] = 255
        return cv2.resize(mask,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)
    def compare_clustering(self, image):
        km,_ = self.kmeans_segmentation(image); db = self.dbscan_segmentation(image)
        return {"kmeans":{"mask":km,"score":self._score(km)},"dbscan":{"mask":db,"score":self._score(db)}}
    def _score(self,mask):
        if mask.sum()==0: return 0.
        nl,_,stats,_=cv2.connectedComponentsWithStats(mask,8,cv2.CV_32S)
        nc=nl-1; 
        if nc<=0: return 0.
        avg=np.mean(stats[1:,cv2.CC_STAT_AREA])
        return float(nc*min(avg/self.REF_AREA,1.))
