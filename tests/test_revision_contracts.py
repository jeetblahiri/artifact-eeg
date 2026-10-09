"""Scientific failure modes of the additional revision controls."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from revision_math import smooth_guard
from measurement_math import conformal_quantile

class RevisionContracts(unittest.TestCase):
    def test_bandlimited_guard_preserves_constraints_without_window_steps(self):
        t=np.arange(512)/256-.5
        b=((t>=-.2)&(t<0)).astype(float);b/=b.sum()
        w=((t>=.3)&(t<.6)).astype(float);w/=w.sum();l=np.array([w-b,b])
        rng=np.random.default_rng(51);parent=rng.normal(size=(3,512));candidate=rng.normal(size=(3,512))
        guarded=smooth_guard(parent,candidate,l);correction=guarded-candidate
        np.testing.assert_allclose(guarded@l.T,parent@l.T,atol=1e-12)
        freq=np.fft.rfftfreq(512,1/256)
        self.assertLess(np.max(abs(np.fft.rfft(correction)[:,freq>30])),1e-12)
        # Feasible bandlimited null-constraint changes can only increase norm.
        trial=np.sin(2*np.pi*5*t)
        # Project an independently chosen bandlimited direction off the
        # correction row-space, then verify it is an admissible alternative.
        from revision_math import bandlimited_guard_matrix
        null=trial-bandlimited_guard_matrix(l)@(l@trial)
        np.testing.assert_allclose(l@null,0,atol=1e-12)
        for a in [-3.,-1.,1.,3.]:
            self.assertGreaterEqual(np.linalg.norm(correction+a*null)+1e-12,np.linalg.norm(correction))

    def test_twenty_calibration_participants_allow_finite_95_percent_rank(self):
        values=np.arange(21,dtype=float);covered=0
        for i in range(21):
            q=conformal_quantile(np.delete(values,i),.05)
            self.assertTrue(np.isfinite(q));covered+=int(values[i]<=q)
        self.assertEqual(covered,20)

if __name__=='__main__':unittest.main()
