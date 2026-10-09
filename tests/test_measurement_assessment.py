"""Checks for substantive failures of proposed measurement certificates."""
import sys,unittest,itertools
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from measurement_math import preserve_measurement,conformal_quantile,constrained_rank_radius

class MeasurementAssessmentContracts(unittest.TestCase):
    def test_guard_is_closest_feasible_candidate_with_redundant_endpoints(self):
        l=np.array([[1.,-1.,0.],[2.,-2.,0.]])
        parent=np.array([3.,1.,8.]);candidate=np.array([0.,2.,4.])
        g=preserve_measurement(parent,candidate,l)
        np.testing.assert_allclose(l@g,l@parent,atol=1e-12)
        # All feasible alternatives differ by null-space directions.
        for a,b in itertools.product([-3,0,2],repeat=2):
            h=g+np.array([a,a,b]);self.assertGreaterEqual(np.linalg.norm(h-candidate)+1e-12,np.linalg.norm(g-candidate))

    def test_preserving_endpoints_does_not_bound_waveform_or_rank_in_null_space(self):
        l=np.array([[1.,0.,0.]])
        for magnitude in [1,100,10000]:
            n=np.array([1.,magnitude,0.]);t=np.array([1.,0.,0.])
            self.assertEqual(float(np.linalg.norm(l@(n-t))),0)
            self.assertEqual(float(np.linalg.norm(n-t)),magnitude)
        center,width=constrained_rank_radius(np.array([0.,1.,0.]),l,[0.],2)
        self.assertEqual(center,0);self.assertEqual(width,4)
        self.assertEqual(constrained_rank_radius(np.array([1.,0.,0.]),l,[.2],100),(0.4,0.0))

    def test_quantile_has_exact_rank_coverage_and_detects_small_sample_vacuity(self):
        # Every rank of a new observation is equally likely under exchangeability.
        vals=np.arange(11,dtype=float);covered=0
        for i in range(11):
            q=conformal_quantile(np.delete(vals,i),.1);covered+=int(vals[i]<=q)
        self.assertEqual(covered,10)
        self.assertTrue(np.isinf(conformal_quantile(np.arange(10),.05)))

    def test_joint_score_is_needed_for_selection_safe_coverage(self):
        # Componentwise 90% envelopes can each fail on a different new person.
        a=np.r_[np.zeros(9),1.,0.];b=np.r_[np.zeros(10),1.]
        failures=np.sum((a>0)|(b>0));self.assertEqual(failures,2)
        self.assertGreater(failures/11,.1)

if __name__=='__main__':unittest.main()
