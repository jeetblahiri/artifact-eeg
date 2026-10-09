"""Checks for claims whose failure would invalidate the numerical study."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import make_mixtures, normalize, unique_bank, split_ids, multiway_ci
from mixture_study import TinyCNN,LargeMLP,fit_linear,predict_linear
from real_task import reference_regression,decode

class ScientificContracts(unittest.TestCase):
    def test_true_power_snr_and_shared_scale(self):
        rng=np.random.default_rng(1)
        s=normalize(rng.normal(size=(12,512)));a=normalize(rng.normal(size=(14,512)))
        d=make_mixtures(s,a,np.arange(12),np.arange(14),20,[-7,2],rng)
        actual=10*np.log10(np.sum(d['s']**2,1)/np.sum((d['x']-d['s'])**2,1))
        np.testing.assert_allclose(actual,d['snr_db'],atol=1e-5)
        np.testing.assert_allclose(d['s']*d['scale'][:,None],s[d['eeg_id']],atol=1e-6)

    def test_content_dedup_and_disjointness(self):
        rng=np.random.default_rng(2);a=rng.normal(size=(20,512))
        a=np.r_[a,3*a[0:1]+2]
        bank,keep,_=unique_bank(a)
        self.assertEqual(len(bank),20)
        split=split_ids(len(bank),rng)
        self.assertFalse(set(split['train'])&set(split['test']))
        self.assertFalse(set(split['val'])&set(split['test']))

    def test_model_capacity(self):
        self.assertEqual(sum(p.numel() for p in TinyCNN().parameters()),1905)
        self.assertEqual(sum(p.numel() for p in LargeMLP().parameters()),1050624)

    def test_linear_rules_preserve_nonzero_mean(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(600,512))
        s=.4*x+np.sin(np.arange(512))[None,:]
        d=dict(x=x,s=s);fit=fit_linear(d,d);pred=predict_linear(fit,x)
        self.assertLess(np.mean((pred['affine']-s)**2),1e-5)
        self.assertLess(np.mean((pred['fourier']-s)**2),1e-10)

    def test_reference_regression_removes_known_reference_component(self):
        rng=np.random.default_rng(1);e=rng.normal(size=(8,3,512));beta=rng.normal(size=(3,22))
        x=np.einsum('nkt,kc->nct',e,beta)
        a,b,_=reference_regression(x,e,x,e)
        self.assertLess(np.mean((a-a.mean(axis=(0,2),keepdims=True))**2),1e-6)

    def test_decoder_uses_features_and_all_classes(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(32,22,512));y=np.tile(np.arange(4),8)
        pred,prob,*_=decode(x,y,x[:8])
        self.assertEqual(pred.shape,(8,));self.assertEqual(prob.shape,(8,4))
        np.testing.assert_allclose(prob.sum(1),1)

    def test_multisource_bootstrap_constant_contrast(self):
        diff=np.full(16,.12);ci=multiway_ci(diff,np.repeat(np.arange(4),4),np.tile(np.arange(4),4),np.random.default_rng(1),100)
        np.testing.assert_allclose(ci,[.12,.12])

if __name__=='__main__':
    unittest.main()
