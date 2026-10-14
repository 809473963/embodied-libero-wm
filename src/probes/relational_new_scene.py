"""Scene fingerprints and fixed diagnostic actions, independent of outcomes."""
import hashlib
import numpy as np

def canonical_layout(pose):
    x=np.asarray(pose[:5,:7],np.float64).copy()
    assert x.shape==(5,7) and np.isfinite(x).all()
    q=x[:,3:];norm=np.linalg.norm(q,axis=1);assert (norm>0).all();q/=norm[:,None]
    for i in range(5):
        nonzero=np.flatnonzero(abs(q[i])>1e-12)
        if q[i,nonzero[0]]<0:q[i]*=-1
    x[x==0]=0
    return x

def fingerprint(pose):
    return hashlib.sha256(np.round(canonical_layout(pose),8).tobytes()).hexdigest()

def distance(a,b):
    x,y=canonical_layout(a),canonical_layout(b)
    return dict(max_position_m=float(np.linalg.norm(x[:,:3]-y[:,:3],axis=1).max()),
        max_orientation_deg=float(np.degrees(2*np.arccos(np.clip(abs((x[:,3:]*y[:,3:]).sum(1)),0,1))).max()))

def near(a,b):
    d=distance(a,b);return d['max_position_m']<=.001 and d['max_orientation_deg']<=1.

def probe_actions():
    a=np.zeros((24,7),np.float32);a[:8,6]=-1;a[8:,6]=1;a[8:16,2]=.1
    return a
