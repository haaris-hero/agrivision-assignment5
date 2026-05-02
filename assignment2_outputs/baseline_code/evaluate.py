"""evaluate.py — AgriTech Assignment 1. Import in Assignments 2-5."""
import numpy as np, json
class EvaluationPipeline:
    def compute_metrics(self, pred, gt):
        p=np.array(pred,dtype=float); t=np.array(gt,dtype=float); ts=np.where(t==0,1,t)
        return {"mae":float(np.mean(np.abs(p-t))),"rmse":float(np.sqrt(np.mean((p-t)**2))),
                "mape":float(np.mean(np.abs(p-t)/ts*100)),
                "accuracy_percent":float(100*np.mean(1-np.abs(p-t)/ts)),
                "perfect_count_accuracy":float(np.mean(p==t)),
                "error_std":float(np.std(p-t)),
                "max_underestimate":float(np.min(p-t)),"max_overestimate":float(np.max(p-t)),
                "per_image":{"predicted":p.tolist(),"true":t.tolist(),"errors":(p-t).tolist()}}
    def identify_failure_cases(self, pred, gt, threshold=10):
        p=np.array(pred,float); t=np.array(gt,float); ts=np.where(t==0,1,t)
        pct=100*np.abs(p-t)/ts
        return [{"image_index":int(i),"true_count":int(t[i]),"predicted_count":int(p[i]),
                 "error":int(p[i]-t[i]),"percent_error":float(pct[i])}
                for i in np.where(pct>threshold)[0]]
    def generate_report(self, metrics, path):
        r={"summary":{k:metrics[k] for k in ["mae","rmse","mape","accuracy_percent","perfect_count_accuracy"]},
           "detailed":{k:metrics[k] for k in ["error_std","max_underestimate","max_overestimate"]},
           "per_image":metrics["per_image"]}
        with open(path,"w") as f: json.dump(r,f,indent=2)
        return r
