"""Regress optional native cubic-phase support and legacy defaults."""
import unittest
import numpy as np
from gwspace import libFastGB
from gwspace.Waveform import GCBWaveform


class OptionalFddotTests(unittest.TestCase):
    def waveform(self,fddot):
        T=30*86400.
        return GCBWaveform(.6,.4,T,phi0=.4,f0=.003000047,fdot=0.,fddot=fddot,
            DL=.005,Lambda=2.1,Beta=-.08,iota=.68,psi=.2)

    def test_zero_matches_legacy_exactly(self):
        w=self.waveform(0.)
        a=w.get_fastgb_fd_single(10.,8)
        b=w.get_fastgb_fd_single(10.,8,include_fddot=True)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)

    def test_default_preserves_legacy_with_nonzero_fddot(self):
        w=self.waveform(1e-20);z=self.waveform(0.)
        for x,y in zip(w.get_fastgb_fd_single(10.,8),z.get_fastgb_fd_single(10.,8)):
            np.testing.assert_array_equal(x,y)
        a=w.get_fastgb_fd_single(10.,8)[1]
        b=w.get_fastgb_fd_single(10.,8,include_fddot=True)[1]
        self.assertGreater(np.linalg.norm(a-b)/np.linalg.norm(a),.01)


    def test_full_buffer_propagates_option(self):
        w=self.waveform(1e-20)
        f,*local=w.get_fastgb_fd_single(10.,8,include_fddot=True)
        _,*full=w.get_fastgb_fd(10.,8,include_fddot=True)
        bins=np.rint(f*w.T_obs).astype(int)
        for a,b in zip(local,full):np.testing.assert_array_equal(a,b[bins])

    def test_bad_parameter_count_raises(self):
        n=64;out=[np.zeros(2*n) for _ in range(6)]
        for length,count in [(7,7),(8,9),(10,10)]:
            with self.subTest(length=length,count=count),self.assertRaises(ValueError):
                libFastGB.ComputeXYZ_FD(np.ones(length),n,86400.,10.,*out,count,'TianQin')

    def test_larger_parameter_buffer_keeps_eight_parameter_behavior(self):
        values=np.array([.003,0.,-.08,2.1,1e-22,.68,.2,.4])
        short=[np.zeros(128) for _ in range(6)];long=[np.zeros(128) for _ in range(6)]
        libFastGB.ComputeXYZ_FD(values,64,86400.,10.,*short,8,'TianQin')
        libFastGB.ComputeXYZ_FD(np.append(values,123.),64,86400.,10.,*long,8,'TianQin')
        for a,b in zip(short,long):np.testing.assert_array_equal(a,b)


if __name__=='__main__':unittest.main()
