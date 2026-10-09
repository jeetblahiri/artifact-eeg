"""Smooth, exactly endpoint-preserving correction in a declared frequency space."""
import numpy as np

def bandlimited_guard_matrix(operator,sampling_hz=256,cutoff_hz=30):
    l=np.atleast_2d(np.asarray(operator,float));n=l.shape[1]
    frequencies=np.fft.rfftfreq(n,1/sampling_hz)
    pl=np.fft.irfft(np.fft.rfft(l,axis=1)*(frequencies<=cutoff_hz),n=n,axis=1)
    gram=l@pl.T
    if np.linalg.matrix_rank(gram)!=len(l):raise ValueError('Constraints are not feasible in the declared frequency space')
    return pl.T@np.linalg.inv(gram)

def smooth_guard(parent,candidate,operator,sampling_hz=256,cutoff_hz=30):
    l=np.atleast_2d(np.asarray(operator,float));q=bandlimited_guard_matrix(l,sampling_hz,cutoff_hz)
    residual=np.einsum('...t,kt->...k',parent-candidate,l)
    return candidate+np.einsum('...k,tk->...t',residual,q)
