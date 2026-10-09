"""Explicit linear measurement and finite-sample uncertainty operations."""
import math
import numpy as np

def endpoint_weights(times, window, baseline):
    w=np.zeros(len(times),float)
    signal=(times>=window[0])&(times<window[1])
    base=(times>=baseline[0])&(times<baseline[1])
    if not signal.any() or not base.any():raise ValueError('Empty measurement window')
    w[signal]=1/signal.sum();w[base]-=1/base.sum()
    return w

def preserve_measurement(parent, candidate, operator):
    """Closest candidate satisfying L output = L parent, in each leading row."""
    l=np.atleast_2d(np.asarray(operator,float))
    residual=np.einsum('...t,kt->...k',parent-candidate,l)
    return candidate+np.einsum('...k,tk->...t',residual,np.linalg.pinv(l))

def conformal_quantile(scores,alpha):
    s=np.asarray(scores,float)
    if s.ndim!=1 or not len(s) or not np.isfinite(s).all():raise ValueError('Invalid calibration scores')
    k=math.ceil((len(s)+1)*(1-alpha))
    return float(np.sort(s)[k-1]) if k<=len(s) else math.inf

def constrained_rank_radius(direction, operator, endpoint_bias, residual_radius):
    """Center/halfwidth of 2<v,Delta> for L Delta=b and null-L radius r."""
    v=np.asarray(direction,float);l=np.atleast_2d(operator)
    center=np.linalg.pinv(l)@np.asarray(endpoint_bias,float)
    if not np.allclose(l@center,endpoint_bias):raise ValueError('Inconsistent endpoint constraints')
    null_v=v-np.linalg.pinv(l)@(l@v)
    return 2*float(v@center),2*residual_radius*float(np.linalg.norm(null_v))
