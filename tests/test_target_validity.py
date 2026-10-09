"""Scientific contracts for new target-uncertainty and teacher controls."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from target_validity import risk_bounds,ranking_radius
from teacher_intervention import teacher_projection

class TargetValidityContracts(unittest.TestCase):
    def test_risk_identity_and_contrast_hold_without_independence(self):
        rng=np.random.default_rng(21)
        n=rng.normal(size=(200,32));t=.7*n+rng.normal(size=n.shape)*.2
        f=.8*t+.1*n;g=.4*t
        delta=t-n;e=f-t
        true=np.mean((f-n)**2)
        self.assertAlmostEqual(true,np.mean(e**2)+np.mean(delta**2)+2*np.mean(e*delta),places=12)
        self.assertAlmostEqual(true-np.mean((g-n)**2),
            np.mean((f-t)**2)-np.mean((g-t)**2)+2*np.mean((f-g)*delta),places=12)

    def test_sharp_bounds_include_cancellation_and_endpoints(self):
        t=np.zeros((4,8));f=np.ones_like(t)*2
        self.assertEqual(risk_bounds(4,1),(1,9))
        for delta,expected in [(np.ones_like(t),9),(-np.ones_like(t),1)]:
            self.assertEqual(float(np.mean((f-t+delta)**2)),expected)
        self.assertEqual(risk_bounds(4,3)[0],0)
        self.assertEqual(float(np.mean((f-t-f)**2)),0)

    def test_family_bounds_retain_seed_spread(self):
        # Two seed outputs have mean 2 and immutable spread 1. A common
        # target shift can cancel their mean error but cannot cancel spread.
        f=np.asarray([np.ones((3,8)),np.ones((3,8))*3])
        self.assertEqual(risk_bounds(5,2,family_variance=1),(1,17))
        self.assertEqual(float(np.mean((f-2)**2)),1)
        self.assertEqual(float(np.mean((f+2)**2)),17)

    def test_ranking_radius_uses_family_losses_not_ensemble_loss(self):
        rng=np.random.default_rng(22);t=rng.normal(size=(40,16))
        f=np.asarray([t+.1,t+.6]);g=np.asarray([t+.2])
        gap,kappa,radius=ranking_radius(f,g,t)
        direction=f.mean(0)-g.mean(0)
        delta=-np.sign(gap)*radius*direction/kappa
        neural=t-delta
        self.assertAlmostEqual(float(np.mean((f-neural)**2)-np.mean((g-neural)**2)),0,places=12)
        self.assertNotAlmostEqual(gap,float(np.mean((f.mean(0)-t)**2)-np.mean((g.mean(0)-t)**2)),places=8)

    def test_teacher_is_orthogonal_and_idempotent(self):
        rng=np.random.default_rng(23);x=rng.normal(size=(5,3,512)).astype(np.float32)
        for variant in ['remove_8_30','retain_1_8','remove_1_4']:
            p=teacher_projection(x,variant);r=x-p
            # The delivered teacher returns float32. Two FFT passes therefore
            # agree up to roundoff rather than bitwise; retain a dtype-scaled
            # tolerance while testing the underlying projection identity.
            tolerance=10*np.finfo(np.float32).eps*max(1,float(np.max(np.abs(p))))
            np.testing.assert_allclose(teacher_projection(p,variant),p,rtol=0,atol=tolerance)
            self.assertLess(abs(float(np.mean(p*r))),1e-7)
            self.assertAlmostEqual(float(np.mean(x*x)),float(np.mean(p*p)+np.mean(r*r)),places=6)

if __name__=='__main__':
    unittest.main()
