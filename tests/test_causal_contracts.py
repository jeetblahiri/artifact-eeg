"""Tests of the additional mathematical identification and rank claims."""
import sys,unittest
from pathlib import Path
import numpy as np
from scipy.stats import betabinom,hypergeom
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from benchmark_models import BenchmarkCNN
import torch

class CausalContracts(unittest.TestCase):
    def test_ridge_target_intervention_is_exact(self):
        rng=np.random.default_rng(8);x=rng.normal(size=(80,12));t=rng.normal(size=(80,4));d=rng.normal(size=(80,4));z=rng.normal(size=(9,12));eta=.07
        xc=x-x.mean(0);a=xc.T@xc+eta*np.eye(12)
        def fit(y):
            w=np.linalg.solve(a,xc.T@(y-y.mean(0)));return (z-x.mean(0))@w+y.mean(0)
        expected=(z-x.mean(0))@np.linalg.solve(a,xc.T@(d-d.mean(0)))+d.mean(0)
        np.testing.assert_allclose(fit(t+d)-fit(t),expected,atol=1e-12)
    def test_random_calibration_threshold_uses_rank_count(self):
        # Count <=5 test scores below calibration rank19: among the first24
        # ranks at least19 must be calibration observations.
        rank=float(hypergeom.sf(18,30,20,24));beta=float(betabinom.cdf(5,10,19,2))
        self.assertAlmostEqual(rank,beta,places=14);self.assertAlmostEqual(rank,.008841732979664,places=14)
    def test_published_architecture_shapes_and_parameters(self):
        # Verify the architectures independently against layer counts from
        # pinned source: simple four64-channel convs; complex six residuals.
        for architecture,count in [('simple_cnn',16815552),('complex_cnn',8455424)]:
            model=BenchmarkCNN(architecture).eval()
            with torch.no_grad():y=model(torch.zeros(2,1,512))
            self.assertEqual(tuple(y.shape),(2,512));self.assertTrue(torch.isfinite(y).all())
            if count is not None:self.assertEqual(sum(p.numel() for p in model.parameters()),count)

if __name__=='__main__':unittest.main()
