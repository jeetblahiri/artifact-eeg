"""Utilities shared by the numerical studies; no hidden acquisition or fallbacks."""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(ROOT / ".cache" / "mne"))
import numpy as np

def config():
    c = json.loads((ROOT / "config.json").read_text())
    for key in ["source_dir", "ecg_dir", "bci_dir"]:
        c[key] = str((ROOT / c[key]).resolve())
    return c

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")

def normalize(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("Expected a finite 2D waveform bank")
    a = a - a.mean(axis=1, keepdims=True)
    scale = a.std(axis=1, keepdims=True)
    if np.any(scale <= 1e-12):
        raise ValueError("Degenerate source waveform")
    return a / scale

def content_ids(a, decimals=8):
    return [hashlib.sha256(np.round(row, decimals).astype('<f8').tobytes()).hexdigest()
            for row in normalize(a)]

def unique_bank(a):
    ids = content_ids(a)
    seen, keep = set(), []
    for i, h in enumerate(ids):
        if h not in seen:
            keep.append(i)
            seen.add(h)
    return normalize(a[keep]).astype(np.float32), np.asarray(keep), ids

def split_ids(n, rng, fractions=(.6, .2, .2)):
    p = rng.permutation(n)
    a, b = int(n * fractions[0]), int(n * (fractions[0] + fractions[1]))
    return dict(train=p[:a], val=p[a:b], test=p[b:])

def make_mixtures(s_bank, a_bank, s_ids, a_ids, n, snrs, rng):
    si = rng.choice(s_ids, n)
    ai = rng.choice(a_ids, n)
    db = rng.choice(snrs, n)
    s = s_bank[si].astype(np.float64)
    a = a_bank[ai].astype(np.float64)
    coeff = np.sqrt(np.mean(s*s, axis=1) / np.mean(a*a, axis=1)) * 10.**(-db/20)
    x = s + coeff[:, None]*a
    scale = x.std(axis=1)
    return dict(x=(x/scale[:, None]).astype(np.float32),
                s=(s/scale[:, None]).astype(np.float32),
                eeg_id=si, artifact_id=ai, snr_db=db, scale=scale, coefficient=coeff)

def per_row_metrics(pred, target, fs=256):
    pred, target = np.asarray(pred, float), np.asarray(target, float)
    err = pred-target
    mse = np.mean(err*err, axis=1)
    rrmse = np.sqrt(mse / np.mean(target*target, axis=1))
    pc, tc = pred-pred.mean(1, keepdims=True), target-target.mean(1, keepdims=True)
    denom = np.linalg.norm(pc, axis=1)*np.linalg.norm(tc, axis=1)
    corr = np.divide(np.sum(pc*tc, axis=1), denom, out=np.full(len(pc), np.nan), where=denom>1e-12)
    out = dict(mse=mse, rrmse=rrmse, corr=corr)
    ef, sf = np.fft.rfft(err), np.fft.rfft(target)
    freq = np.fft.rfftfreq(target.shape[1], 1/fs)
    for name, lo, hi in [('delta',1,4),('theta',4,8),('alpha',8,13),('beta',13,30),('high',30,80)]:
        mask = (freq >= lo) & (freq < hi)
        out[name+'_relative_error_power'] = np.sum(abs(ef[:,mask])**2,1) / np.maximum(np.sum(abs(sf[:,mask])**2,1),1e-12)
    return out

def multiway_ci(diff, eeg, artifact, rng, n_boot=1000):
    """Two-source Poisson product multiplier; conditional source-bank uncertainty."""
    _, es = np.unique(eeg, return_inverse=True)
    _, aa = np.unique(artifact, return_inverse=True)
    vals = []
    for _ in range(n_boot):
        weights = rng.poisson(1, es.max()+1)[es]*rng.poisson(1, aa.max()+1)[aa]
        if weights.sum():
            vals.append(float(np.sum(weights*diff)/weights.sum()))
    return [float(v) for v in np.quantile(vals,[.025,.975])]
