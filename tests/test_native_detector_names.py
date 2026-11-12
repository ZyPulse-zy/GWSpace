"""Standalone upstream-style tests for native detector naming and short records."""
import unittest
import numpy as np
from gwspace import libFastGB
from gwspace.Orbit import get_pos, TaijiOrbit
from gwspace.Waveform import GCBWaveform


class NativeDetectorTests(unittest.TestCase):
    def wave(self,days=90):
        return GCBWaveform(.6,.4,days*86400.,phi0=.4,f0=.003000047,
            fdot=0.,fddot=0.,DL=.005,Lambda=2.1,Beta=-.08,iota=.68,psi=.2)

    def test_taiji_aliases_generate_same_finite_signal(self):
        w=self.wave()
        a=w.get_fastgb_fd_single(10.,8,detector='Taiji')
        b=w.get_fastgb_fd_single(10.,8,detector='TaiJi')
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        self.assertTrue(np.isfinite(a[1:]).all())
        self.assertGreater(np.linalg.norm(a[1:]),0.)

    def test_native_taiji_orbit_matches_python(self):
        t=np.linspace(0.,90*86400.,101)
        x,y,z,L=get_pos(t,'Taiji')
        native=np.array([x,y,z]).transpose(1,0,2)
        expected=TaijiOrbit(t)
        np.testing.assert_allclose(native,expected.orbits,rtol=0.,atol=1e-8)
        self.assertAlmostEqual(L,expected.L_T,delta=1e-12)

    def test_single_native_position_has_same_alias(self):
        t=1234.
        a=[np.zeros(3) for _ in range(3)];b=[np.zeros(3) for _ in range(3)]
        libFastGB.Compute_position(t,*a,'Taiji')
        libFastGB.Compute_position(t,*b,'TaiJi')
        np.testing.assert_array_equal(a,b)

    def test_unknown_detector_is_rejected(self):
        t=np.array([0.]);xyz=[np.zeros(3) for _ in range(3)]
        with self.assertRaises(ValueError):get_pos(t,'unknown')
        with self.assertRaises(ValueError):libFastGB.Compute_position(0.,*xyz,'unknown')


if __name__ == "__main__":
    unittest.main()
