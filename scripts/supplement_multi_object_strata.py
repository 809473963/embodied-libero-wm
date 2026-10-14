#!/usr/bin/env python3
"""Descriptive spatial and action strata; thresholds fixed, no training/selection."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.diagnose_multi_object_perception import inputs,read_npz,CV,NAMES,stats,save

def main():
 d,n,_=inputs();r=n['roots'];t=n['positions'];results={}
 for family in ['shared','independent']:
  p=read_npz(CV/f'oof_{family}_ensemble.npz')['nodes'];err=np.linalg.norm(p-t,axis=2)*1000;objects={}
  for obj in range(5):
   bins={}
   for axis in range(3):
    category=np.zeros(len(r),int);cutoffs={}
    for f in range(3):
     q=np.quantile(t[d['fold']!=f,obj,axis],[1/3,2/3]);category[d['fold']==f]=np.searchsorted(q,t[d['fold']==f,obj,axis],side='right');cutoffs[str(f)]=q.tolist()
    bins['world_'+('xyz'[axis])]=dict(fit_only_cutoffs_m=cutoffs,strata={str(k):stats(err[category==k,obj],r[category==k]) for k in range(3)})
   dist=np.linalg.norm(t[:,obj]-t[:,2],axis=1)
   bins['distance_to_plate_m']={label:stats(err[mask,obj],r[mask]) for label,mask in [('under_0.1',dist<.1),('0.1_to_0.3',(dist>=.1)&(dist<.3)),('over_0.3',dist>=.3)]}
   action=d['obs'][:,12:19];translation=np.linalg.norm(action[:,:3],axis=1)
   bins['previous_action']={label:stats(err[mask,obj],r[mask]) for label,mask in [('translation_over_0.02',translation>.02),('translation_at_most_0.02',translation<=.02),('z_positive',action[:,2]>.02),('z_negative',action[:,2]<-.02)]}
   objects[NAMES[obj]]=bins
  results[family]=objects
 save('spatial_phase_analysis.json',dict(models=results,interpretation='GT-position-conditioned descriptive strata, not deployable routing; previous action coordinates are controller commands, not measured displacement or semantic manipulation phase; thresholds not selected from performance.'))
if __name__=='__main__':main()
