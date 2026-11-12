"""Regress optional finite-interval slow-envelope quadrature using public APIs."""
import unittest

import numpy as np

from gwspace import libFastGB
from gwspace.Waveform import GCBWaveform
from gwspace.constants import EarthOrbitOmega_SI
from gwspace.response import tdi_XYZ2AET


def waveform(frequency=.003000047, duration=90*86400., fdot=0.):
    return GCBWaveform(.6, .4, duration, phi0=.4, f0=frequency, fdot=fdot,
                       fddot=0., DL=.005, Lambda=2.1, Beta=-.08, iota=.68, psi=.2)


def native(wave, count, detector='TianQin', correction=None):
    params = np.array([wave.f0, wave.fdot, wave.Beta, wave.Lambda,
                       wave.amplitude, wave.iota, wave.psi, wave.phi0])
    output = [np.zeros(2*count) for _ in range(6)]
    options = {} if correction is None else {'endpoint_correction': correction}
    libFastGB.ComputeXYZ_FD(params, count, wave.T_obs, 10., *output, 8,
                          detector=detector, **options)
    return np.asarray([v.view(np.complex128) for v in output])


def ae(raw):
    return np.asarray(tdi_XYZ2AET(*raw[3:])[:2])


def crop(values, count):
    # Fixed integer-bin offsets about the same native carrier, independent of N.
    start = (values.shape[-1]-count)//2
    return values[..., start:start+count]


def finite_reference(wave, count, detector='TianQin'):
    # Extrapolate the uncorrected left sums, independently of the new formula.
    coarse = crop(ae(native(wave, 32768, detector, False)), count)
    fine = crop(ae(native(wave, 65536, detector, False)), count)
    return 2*fine-coarse


class EndpointCorrectionTests(unittest.TestCase):
    def test_default_and_explicit_false_are_bitwise_identical(self):
        for detector in ('TianQin', 'LISA', 'TaiJi'):
            with self.subTest(detector=detector):
                wave = waveform()
                self.assertEqual(native(wave, 512, detector).tobytes(),
                                 native(wave, 512, detector, False).tobytes())

    def test_same_model_finite_integral_improves_at_fixed_bins(self):
        for frequency in (.000500047, .003000047, .009999973):
            with self.subTest(frequency=frequency):
                wave = waveform(frequency)
                reference = finite_reference(wave, 2048)
                original = ae(native(wave, 2048, correction=False))
                corrected = ae(native(wave, 2048, correction=True))
                scale = np.linalg.norm(reference)
                before = np.linalg.norm(original-reference)/scale
                after = np.linalg.norm(corrected-reference)/scale
                self.assertLess(after, .001)
                self.assertLess(after, before/10)

    def test_native_detector_paths_are_finite_and_improve_quadrature(self):
        for detector in ('LISA', 'TaiJi'):
            with self.subTest(detector=detector):
                wave = waveform(duration=180*86400.)
                reference = finite_reference(wave, 512, detector)
                before = ae(native(wave, 512, detector, False))
                after = ae(native(wave, 512, detector, True))
                self.assertTrue(np.isfinite(after).all())
                self.assertLess(np.linalg.norm(after-reference),
                                np.linalg.norm(before-reference))

    def test_zero_and_near_zero_slow_bins(self):
        wave = waveform()
        count = 16384
        for detector in ('TianQin', 'LISA'):
            with self.subTest(detector=detector):
                reference = finite_reference(wave, count, detector)
                corrected = ae(native(wave, count, detector, True))
                selected = slice(count//2-1, count//2+2)
                self.assertTrue(np.isfinite(corrected).all())
                error = np.linalg.norm((corrected-reference)[:, selected])
                # Keep the same quadrature budget, including weak carrier bins.
                self.assertLess(error/np.linalg.norm(reference[:, selected]), .001)

    def test_periodic_envelope_changes_only_by_roundoff(self):
        duration = 2*np.pi/EarthOrbitOmega_SI
        wave = waveform(100003/duration, duration)
        original = native(wave, 512, 'LISA', False)
        corrected = native(wave, 512, 'LISA', True)
        self.assertLess(np.linalg.norm(corrected-original)/np.linalg.norm(original), 1e-8)

    def test_physical_dc_and_small_fft_sizes(self):
        wave = waveform(3/86400., 86400.)
        raw = native(wave, 128, correction=True)
        self.assertTrue(np.isfinite(raw).all())
        np.testing.assert_array_equal(raw[3:, 128//2-3], 0.)
        for count in (2, 4):
            self.assertTrue(np.isfinite(native(wave, count, correction=True)).all())
        for count in (1, 3):
            with self.subTest(count=count), self.assertRaises(ValueError):
                native(wave, count, correction=True)

    def test_high_level_grid_buffer_and_td_forwarding(self):
        wave = waveform()
        dt = 60.
        for corrected in (False, True):
            frequency, *local = wave.get_fastgb_fd_single(dt, 8, endpoint_correction=corrected)
            full_frequency, *full = wave.get_fastgb_fd(dt, 8, endpoint_correction=corrected)
            bins = np.rint(frequency*wave.T_obs).astype(int)
            for expected, actual in zip(local, full):
                np.testing.assert_array_equal(actual[bins], expected)
            times, *samples = wave.get_fastgb_td(dt, 8, endpoint_correction=corrected)
            for spectrum, actual in zip(full, samples):
                expected = np.fft.irfft(spectrum)*len(times)
                # FFT scaling order can differ by roundoff near time-series zeros.
                error = np.linalg.norm(actual-expected)/np.linalg.norm(expected)
                self.assertLess(error, 1e-14)
        original_grid = wave.get_fastgb_fd_single(dt, 8)[0]
        np.testing.assert_array_equal(frequency, original_grid)


if __name__ == '__main__':
    unittest.main()
