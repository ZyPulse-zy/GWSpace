"""Compare FastGB endpoint quadrature on fixed bins using only public GWSpace.

Run after installing this branch: python examples/fastgb_endpoint_correction.py
The reference extrapolates two fine uncorrected left sums of the same model;
it is not an independent moving-spacecraft or time-domain response reference.
"""
import numpy as np

from gwspace.Noise import TianQinNoise
from gwspace.Waveform import GCBWaveform
from gwspace.response import tdi_XYZ2AET


def response(wave, oversample, correction=False):
    frequency, *xyz = wave.get_fastgb_fd_single(
        10., oversample, endpoint_correction=correction)
    return frequency, 10*np.asarray(tdi_XYZ2AET(*xyz)[:2])


def same_bins(values, count):
    # All calls use the same native carrier; only their slow FFT length differs.
    start = (values.shape[-1]-count)//2
    return values[:, start:start+count]


def relative_error(value, reference, psd):
    return np.sqrt(np.sum(abs(value-reference)**2/psd)
                   / np.sum(abs(reference)**2/psd))


def main():
    print('90 days, dt=10 s, fixed 2048-bin TianQin A/E support, instrument PSD')
    print('f0 [mHz]    FFT error [%]    corrected error [%]')
    for frequency in (.000500047, .003000047, .009999973):
        wave = GCBWaveform(.6, .4, 90*86400., phi0=.4, f0=frequency,
                           fddot=0., DL=.005, Lambda=2.1, Beta=-.08,
                           iota=.68, psi=.2)
        bins, original = response(wave, 8)
        _, corrected = response(wave, 8, True)
        _, fine = response(wave, 128)
        _, finer = response(wave, 256)
        reference = 2*same_bins(finer, len(bins))-same_bins(fine, len(bins))
        psd, _ = TianQinNoise().noise_AET(
            bins, unit='relative_frequency', TDIgen=1, wd_foreground=0.)
        print(f'{1000*frequency:8.6f}    '
              f'{100*relative_error(original, reference, psd):13.6f}    '
              f'{100*relative_error(corrected, reference, psd):19.6f}')


if __name__ == '__main__':
    main()
