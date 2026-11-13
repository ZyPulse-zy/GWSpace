"""Short records still need a valid radix-two slow FFT length."""
import unittest
import numpy as np
from gwspace.Waveform import GCBWaveform


def waveform(days):
    return GCBWaveform(.6,.4,days*86400.,phi0=.4,f0=.003000047,
        fdot=0.,fddot=0.,DL=.005,Lambda=2.1,Beta=-.08,iota=.68,psi=.2)


class ShortObservationTests(unittest.TestCase):
    def test_positive_short_record_has_radix_two_buffer(self):
        for days in [.01,.4,1,7,30]:
            for detector in ['TianQin','LISA','Taiji']:
                with self.subTest(days=days,detector=detector):
                    n=waveform(days)._buffer_size(detector)
                    self.assertGreaterEqual(n,2)
                    self.assertEqual(n&(n-1),0)

    def test_short_native_tianqin_and_lisa_are_finite(self):
        for days in [.01,.4,1,7,30,45,60]:
            for detector in ['TianQin','LISA']:
                for oversample in [1,8]:
                    with self.subTest(days=days,detector=detector,oversample=oversample):
                        _,*response=waveform(days).get_fastgb_fd_single(
                            10.,oversample,detector=detector)
                        self.assertTrue(np.isfinite(response).all())

    def test_already_valid_oversampled_buffers_are_unchanged(self):
        # A one-point base estimate can still yield a valid oversampled FFT.
        for days,detector in [(.4,'TianQin'),(45,'LISA'),(45,'Taiji')]:
            for oversample in [2,8]:
                with self.subTest(days=days,detector=detector,oversample=oversample):
                    self.assertEqual(waveform(days)._buffer_size(detector,oversample),
                                     oversample)

    def test_long_record_buffers_are_unchanged(self):
        self.assertEqual(waveform(90)._buffer_size('TianQin',8),2048)
        self.assertEqual(waveform(90)._buffer_size('LISA',8),16)


if __name__=='__main__':unittest.main()
