import logging

import pandas as pd

import scipy.signal as scsig
import scipy.fft as scfft
import numpy as np

__all__ = ['butterworth_bandpass', 'butterworth_bandpass_array', 'fft_and_psd']

logger = logging.getLogger(__name__)


def butterworth_bandpass(data: pd.DataFrame, fs: float, lowcut: float, highcut: float, degree=5, axis=0):
    """ Apply a Butterworth bandpass, highpass, or lowpass filter to the x, y, and z components of the input DataFrame.

    Parameters:
        :param data: (DataFrame)  input DataFrame containing x, y, and z components of the signal.
        :param fs: (int) sampling frequency in Hz.
        :param lowcut: (float) lower cutoff frequency (Hz) for the bandpass or highpass filter.
        :param highcut: (float) upper cutoff frequency (Hz) for the bandpass or highpass filter.
        :param degree: (int, Optional)degree of the Butterworth filter (default is 5).
        :param axis: (int, Optional) axis along which to apply the filter (default is 0).

    Returns:
        :return: (DataFrame) with same dimensions as 'data', containing the filtered data. """

    info = {}

    signal = data[['x', 'y', 'z']].to_numpy()
    f_nyq = fs / 2.0

    if lowcut is not None:
        if not (0 < lowcut < f_nyq):
            info.update({'ok': 0,'status': f"lowcut_out_of_range: {lowcut} not in (0, {f_nyq})"})
            return data, info

    if highcut is not None:
        if not (0 < highcut < f_nyq):
            info.update({'ok': 0, 'status': f"highcut_out_of_range: {highcut} not in (0, {f_nyq})"})
            return data, info

    if lowcut is not None and highcut is not None:
        if not (lowcut < highcut):
            info.update({'ok': 0, 'status': "lowcut_not_smaller_than_highcut"})
            return data, info
        _btype = 'bandpass'
        _wn = [lowcut / f_nyq, highcut / f_nyq]

    elif lowcut is not None:
        _btype = 'highpass'
        _wn = lowcut / f_nyq
    elif highcut is not None:
        _btype = 'lowpass'
        _wn = highcut / f_nyq
    else:
        info.update({'ok': 0, 'status': "no_cutoff_provided"})
        return data, info

    try:
        sos = scsig.butter(degree, _wn, btype=_btype, analog=False, output='sos')
        signal_filtered = scsig.sosfiltfilt(sos, signal, axis=axis)
        data[['x', 'y', 'z']] = signal_filtered.astype(data.dtypes.iloc[0], copy=False)
        info.update({'ok': 1,'status': 1})

    except Exception as e:
        info.update({'ok': 0,'status': f"{type(e).__name__}: {e}"})

    return data, info


def butterworth_bandpass_array(signal: np.ndarray, fs: float, lowcut: float, highcut: float, degree=5, axis=0):
    f_nyq = fs / 2.0

    if lowcut is not None and f_nyq <= lowcut:
        logger.warning(f"Skipping lowpass filter: sample rate {fs} too low for cutoff rate {lowcut}")

    if lowcut is not None and highcut is not None:
        _btype = 'bandpass'
        _wn = [lowcut / f_nyq, highcut / f_nyq]
        logger.debug(f"applying bandpass with f_low={lowcut:.3f}Hz={_wn[0]:.3f}*f_nyq "
                     f"and f_high={highcut:.3f}Hz={_wn[1]:.3f}*f_nyq")
    elif lowcut is not None:
        _btype = 'highpass'
        _wn = lowcut / f_nyq
        logger.debug(f"applying highpass with f_low={lowcut:.2f}Hz={_wn:.2f}*f_nyq")
    elif highcut is not None:
        _btype = 'lowpass'
        _wn = highcut / f_nyq
        logger.debug(f"applying lowpass with f_high={highcut:.2f}Hz={_wn:.2f}*f_nyq")
    else:
        raise ValueError("At least one of lowcut and highcut must be provided.")

    sos = scsig.butter(degree, _wn, btype=_btype, analog=False, output='sos')
    return scsig.sosfiltfilt(sos, signal, axis=axis)


def fft_and_psd(data: np.ndarray, fs=100, nperseg=1024):
    # get the FFT spectrum of the data, helps to identify cutoffs
    fft = scfft.fft(data)
    # _freq_fft = np.fft.fftfreq(len(fft)) * fs
    _freq_fft = scfft.fftfreq(len(fft), 1 / fs)
    freq_psd, psd = scsig.welch(data, fs=fs, nperseg=nperseg)
    freq_fft = _freq_fft[:len(_freq_fft) // 2]
    fft = np.abs(fft)[:len(_freq_fft) // 2]
    return freq_fft, fft, freq_psd, psd
