""" Contains functions to calculate local features, i.e. features describing a singular movement bout.
 Inputs are a pd.Series of the movement segment containing columns 'x', 'y', 'z' and 'mag' and a datetime index.
Returns are dicts containing the feature name and value. """

import gc
import logging
import warnings

from nolds import sampen, hurst_rs
import numpy as np
import pandas as pd
import scipy
import statsmodels.tsa.stattools as stattools
import matplotlib.pyplot as plt
from pyhrv.nonlinear import poincare
from sklearn.exceptions import UndefinedMetricWarning

logger = logging.getLogger(__name__)

__all__ = [
    'moment_features',
    'quantile_features',
    'energy_features',
    'spectral_features',
    'autocorr_features',
    'peak_features',
    'non_linear_dynamic_features',
    'movement_duration',
    'poincare_features']

NP_FLOAT_PRECISION = np.float64
MIN_TIMESERIES_SAMPLES = 20


def moment_features(magnitude, x, y, z):
    """ Compute moment features of acceleration signal treated as distribution.

    Parameters:
        :param magnitude: (array-like) The vector magnitude of the acceleration signal.
        :param x: (array-like) The x component.
        :param y: (array-like) The y component.
        :param z: (array-like) The z component.

    Returns:
        :return: (dict) Dictionary containing computed moment features.

    Moment Features:
        - mean_<mag/x/y/z>: The mean value of the respective property.
        - std_<mag/x/y/z>: Standard deviation.
        - skew_<mag/x/y/z>: Skewness.
        - kurt_<mag/x/y/z>: Kurtosis. """
    moments = {}
    for name, data in zip(['mag', 'x', 'y', 'z'], [magnitude, x, y, z]):
        moments.update({
            f"mean_{name}": np.mean(data),
            f"std_{name}": np.std(data),
            f"skew_{name}": scipy.stats.skew(data),
            f"kurt_{name}": scipy.stats.kurtosis(data),
        })
    return moments


def quantile_features(magnitude, x, y, z):
    """ Compute quantile features of the acceleration signal.

    Parameters:
        :param magnitude: (array-like) The vector magnitude of the acceleration signal.
        :param x: (array-like) The x component.
        :param y: (array-like) The y component.
        :param z: (array-like) The z component.

    Returns:
        :return: (dict) Dictionary containing computed quantile features.

    Quantile Features:
        - min_<mag/x/y/z>: The 0% quantile or minimum value of the respective property.
        - q25_<mag/x/y/z>: 25% quantile.
        - med_<mag/x/y/z>: 50% quantile or median.
        - q75_<mag/x/y/z>: 75% quantile.
        - max_<mag/x/y/z>: 100% quantile or maximum. """
    quantiles = {}
    for name, data in zip(['mag', 'x', 'y', 'z'], [magnitude, x, y, z]):
        quantiles.update({
            name: value for name, value in zip(
                [f"min_{name}", f"q25_{name}", f"med_{name}", f"q75_{name}", f"max_{name}"],
                np.quantile(data, (0, .25, .5, .75, 1)))
        })
        quantiles.update({f"iqr_{name}": scipy.stats.iqr(data)})

    return quantiles


def energy_features(magnitude, x, y, z):
    """ Compute energy features of the acceleration signal.

    Parameters:
        :param magnitude: (array-like) The vector magnitude of the acceleration signal.
        :param x: (array-like) The x component.
        :param y: (array-like) The y component.
        :param z: (array-like) The z component.

    Returns:
        :return: (dict) Dictionary containing computed energy features.

    Energy Features:
        - sma_mag: Signal Magnitude Area (SMA) of the acceleration signal.
        - power: Average power of the acceleration signal.
        - rms_mag: Root Mean Square (RMS) of the magnitude of the acceleration signal. """
    return {
        'sma_mag': np.sum(np.abs(x)) + np.sum(np.abs(y)) + np.sum(np.abs(z)),
        'power': np.mean(x ** 2 + y ** 2 + z ** 2),
        'rms_mag': np.sqrt(np.mean(np.square(magnitude))),
    }


def spectral_features(magnitude, n_dom_freqs: int, target_frequencies_hz: list, sample_rate: int, debug: bool = False):
    """ Compute spectral features from the acceleration vector magnitude.

    Parameters:
        :param magnitude: (array-like) The vector magnitude of the acceleration signal.
        :param n_dom_freqs: (int) Number of dominant frequencies to extract.
        :param target_frequencies_hz: (list) List of target frequencies in Hz.
        :param sample_rate: (int) Sampling rate of the signal.
        :param debug: (bool, optional) If True, return additional debug information. Default is False.

    Returns:
    :   return: (dict) Dictionary containing computed spectral features.

    Spectral Features:
        - asd_<target_freq>: Amplitude spectral density at each target frequency.
        - f<i>: Frequency of the <i>-th dominant peak.
        - p<i>: Power of the <i>-th dominant peak.
        - sp_entropy: Spectral entropy.
        - avg_sp_power: Average spectral power."""
    spectral_feats = {}
    _nperseg = sample_rate if len(magnitude) >= sample_rate else len(magnitude)

    try:
        freqs, psd = scipy.signal.welch(
            magnitude,
            fs=sample_rate,
            nperseg=_nperseg,
            noverlap=_nperseg // 2,
            detrend='constant',
            scaling='density',
            average='median'
        )
        asd = np.sqrt(psd)

        # extract the ASD at the target frequencies:
        max_available_freq = float(freqs[-1])
        asd_interp = scipy.interpolate.interp1d(freqs, asd, kind='cubic')

        for target_freq in target_frequencies_hz:
            effective_freq = min(float(target_freq), max_available_freq)

            if target_freq > max_available_freq:
                logger.warning(
                    "Requested ASD frequency %.2f Hz exceeds the maximum measurable frequency %.2f Hz "
                    "(sample_rate=%s Hz, Nyquist=%.2f Hz). Using the ASD at %.2f Hz instead.",
                    target_freq,
                    max_available_freq,
                    sample_rate,
                    sample_rate / 2,
                    effective_freq,
                )

            if effective_freq in freqs:
                value = asd[np.where(freqs == effective_freq)[0][0]]
            else:
                value = asd_interp(effective_freq)

            spectral_feats[f"asd_{target_freq}"] = float(value)

        # find the top n dominant freqs:
        peaks, _ = scipy.signal.find_peaks(asd)
        peak_powers = asd[peaks]
        peak_freqs = freqs[peaks]
        peak_ranks = np.argsort(peak_powers)[::-1]

        spectral_feats.update({f"f{i + 1}": 0 for i in range(n_dom_freqs)})
        spectral_feats.update({f"p{i + 1}": 0 for i in range(n_dom_freqs)})

        for i, j in enumerate(peak_ranks[:n_dom_freqs]):
            spectral_feats[f"f{i + 1}"] = peak_freqs[j]
            spectral_feats[f"p{i + 1}"] = peak_powers[j]

        # average power and spectral entropy:
        spectral_feats.update({
            'sp_entropy': scipy.stats.entropy(asd[asd > 0]),
            'avg_sp_power': np.sum(asd)
        })

    except ValueError as value_error:
        logger.warning(f"Spectral feature computation failed. (ValueError: {value_error} in spectral_features.)")
        spectral_feats.update({f"asd_{f}": np.NaN for f in target_frequencies_hz})
        spectral_feats.update({f"f{i + 1}": np.NaN for i in range(n_dom_freqs)})
        spectral_feats.update({f"p{i + 1}": np.NaN for i in range(n_dom_freqs)})

        if 'asd' in locals():
            spectral_feats.update({
                'sp_entropy': scipy.stats.entropy(asd[asd > 0]),
                'avg_sp_power': np.sum(asd)
            })
        else:
            spectral_feats.update({
                'sp_entropy': np.NaN,
                'avg_sp_power': np.NaN
            })

    if debug:
        return spectral_feats, freqs, asd
    else:
        return spectral_feats


def autocorr_features(magnitude, x, y, z, sample_rate: int, peak_prom_threshold: float, ignore_warnings: bool = False,
                      debug: bool = False):
    """ Compute auto-correlation features from the acceleration signal.
    Uses cross-correlation of signal with itself and extracts features.

    Parameters:
        :param magnitude: (array-like) The vector magnitude of the acceleration signal.
        :param x: (array-like) The x component.
        :param y: (array-like) The y component.
        :param z: (array-like) The z component.
        :param peak_prom_threshold: (float) Prominence threshold for peak detection.
        :param sample_rate: (int) Sampling rate of the signal.
        :param ignore_warnings: (bool, optional) If True, mute warnings for invalid division and map NaNs.
            Default is False.
        :param debug: (bool, optional) If True, return additional debug information. Default is False.

    Returns:
        :return: (dict) Dictionary containing computed auto-correlation features.

    Auto-Correlation Features:
        - ac_first_max_<mag/x/y/z>: Auto-correlation at first maximum.
        - loc_first_max_<mag/x/y/z>: Location (in s) of first maximum.
        - ac_first_min_<mag/x/y/z>: Auto-correlation at first minimum.
        - loc_first_min_<mag/x/y/z>: Location (in s) of first minimum.
        - zeros_<mag/x/y/z>: Number of zero-crossings. """

    autocorr_feats = {}
    autocorrs = []
    for name, data in zip(('mag', 'x', 'y', 'z'), [magnitude, x, y, z]):
        if ignore_warnings:
            with np.errstate(divide='ignore', invalid='ignore'):  # ignore invalid div warnings
                acf = np.nan_to_num(stattools.acf(data, nlags=2 * sample_rate))
        else:
            acf = stattools.acf(data, nlags=2 * sample_rate)

        if debug:
            autocorrs.append(acf)

        # find location and auto-correlation of first maximum:
        peaks, _ = scipy.signal.find_peaks(acf, prominence=peak_prom_threshold)
        if len(peaks) > 0:
            loc_first_max = peaks[0]
            ac_fist_max = acf[loc_first_max]
            loc_first_max /= sample_rate  # in secs
        else:
            ac_fist_max = loc_first_max = 0.0

        # find location and auto-correlation of first minimum:
        valleys, _ = scipy.signal.find_peaks(-acf, prominence=peak_prom_threshold)
        if len(valleys) > 0:
            loc_first_min = valleys[0]
            ac_first_min = acf[loc_first_min]
            loc_first_min /= sample_rate  # convert to seconds
        else:
            ac_first_min = loc_first_min = 0.0

        acf_zeros = np.sum(np.diff(np.signbit(acf)))  # count the zero crossings:

        autocorr_feats.update({
            f"ac_first_max_{name}": ac_fist_max,
            f"loc_first_max_{name}": loc_first_max,
            f"ac_first_min_{name}": ac_first_min,
            f"loc_first_min_{name}": loc_first_min,
            f"zeros_{name}": acf_zeros,
        })

    if debug:
        return autocorr_feats, np.array(autocorrs)
    else:
        return autocorr_feats


def peak_features(magnitude, sample_rate: int, peak_prom_threshold: float, min_peak_distance: float):
    """ Compute peak property features from the acceleration vector magnitude.

    Parameters:
        :param magnitude: (array-like) The vector magnitude of the acceleration signal.
        :param sample_rate: (int) Sampling rate of the signal.
        :param peak_prom_threshold: (float) Prominence threshold for peak detection.
        :param min_peak_distance: (float) Prominence threshold for peak detection.

    Returns:
        :return: (dict) Dictionary containing computed peak features.

    Peak Features:
        - peaks_per_sec: Number of peaks per second of signal.
        - avg_peak_prom: Average peak prominence.
        - max_peak_prom: Maximum peak prominence.
        - min_peak_prom: Minimum peak prominence. """
    peak_feats = {}

    data = magnitude.copy().to_numpy()

    peaks, peak_props = scipy.signal.find_peaks(
        data, distance=min_peak_distance * sample_rate, prominence=peak_prom_threshold)

    peak_feats['peaks_per_sec'] = len(peaks) / (len(magnitude) / sample_rate)  # number of peaks per second

    # peak prominence stats:
    if len(peak_props['prominences']) > 0:
        peak_feats['avg_peak_prom'] = np.mean(peak_props['prominences'])
        peak_feats['max_peak_prom'] = np.max(peak_props['prominences'])
        peak_feats['min_peak_prom'] = np.min(peak_props['prominences'])
    else:
        peak_feats['avg_peak_prom'] = peak_feats['max_peak_prom'] = peak_feats['min_peak_prom'] = 0

    return peak_feats


def non_linear_dynamic_features(magnitude, x, y, z, emb_dim: int):
    """ Compute Non-linear dynamic (NOLD) features from time-series.

    Parameters:
        :param magnitude: (array-like) The vector magnitude of the acceleration signal.
        :param x: (array-like) The x component.
        :param y: (array-like) The y component.
        :param z: (array-like) The z component.
        :param emb_dim: (int) The embedding dimension used in the calculation of nolds.corr_dim. It determines the
            number of lagged copies of the time series used to reconstruct the phase space
            for computing correlation dimension.

    Returns:
        :return: (dict) Dictionary containing computed nold features.

    Nold Features:
        - SampEn_<mag/x/y/z>: Sample entropy. Measures complexity based on approx. entropy.
        - corr_dim_<mag/x/y/z>: Measure of fractal dimension, also related to complexity.
        - hurst_rs_<mag/x/y/z>: Measure of 'long-term memory', i.e. whether a time-series is more, less, or equally
            likely to increase if it has in previous steps.
        - lyap_r_<mag/x/y/z>: Lyapunov exponent. Positive indicates chaos and unpredictability,
            nolds.lyap_r is largest, use nolds.lyap_e for whole spectrum. """

    non_lin_feats = {}

    for name, data in zip(['mag', 'x', 'y', 'z'], [magnitude, x, y, z]):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore',
                    category=UndefinedMetricWarning,
                    message='R\^2 score is not well-defined with less than two samples'
                )

                non_lin_feats.update({
                    f"SampEn_{name}": sampen(data, emb_dim, debug_plot=False),
                    # f"corr_dim_{name}": nolds.corr_dim(data, emb_dim, debug_plot=False),
                    f"hurst_rs_{name}": hurst_rs(data),
                    # f"lyap_r_{name}": nolds.lyap_r(data),
                    # f"lyap_mean_{name}": np.mean(nolds.lyap_e(data)),
                    # f"lyap_std_{name}": np.std(nolds.lyap_e(data)),
                })

        except ValueError as value_error:  # timeseries too short
            logger.warning(f"Timeseries too short. (ValueError: {value_error} in nold.)")
            non_lin_feats.update({
                f"SampEn_{name}": np.NaN,
                # f"corr_dim_{name}": np.NaN,
                f"hurst_rs_{name}": np.NaN,
                # f"lyap_r_{name}": nolds.lyap_r(data),
                # f"lyap_mean_{name}": np.NaN,
                # f"lyap_std_{name}": np.NaN,
            })

    return non_lin_feats


def poincare_features(magnitude, show_plot: bool = False):
    """ Compute Poincaré plot features to analyze long and short-term dynamics and variability of a time-series.

    Parameters:
        :param magnitude: (array-like) The vector magnitude of the acceleration signal.
        :param show_plot: (bool) Whether to show the plot or not.

    Returns:
        :return: (dict) Dictionary containing computed Poincaré plot features.

    Poincaré Features:
        - sd1: Standard deviation of points perpendicular to the line of identity, indicating short-term variability.
        - sd2: Standard deviation along the line of identity, indicating long-term variability.
        - sd_ratio:
            Ratio of sd2 to sd1, providing insight into the balance between long-term and short-term variability.
        - ellipse_area:
            Area of the ellipse formed by the points on the Poincaré plot,
            representing the total variability in the time series.
    """
    poincare_feats = {}
    metrics = poincare(magnitude, show=show_plot)
    poincare_feats.update({
        f"sd1": metrics['sd1'],
        f"sd2": metrics['sd2'],
        f"sd_ratio": metrics['sd_ratio'],
        f"ellipse_area": metrics['ellipse_area'],
    })
    del metrics
    plt.close('all')  # 'show' arg in poincare seems to be buggy and will open fig anyway, so avoid mem. pile-up
    gc.collect()
    return poincare_feats


def movement_duration(index: pd.DatetimeIndex):
    """ The movement duration in seconds.
    Parameters:
        :param index: (pd.DatetimeIndex) The datetime index of the movement episode dataframe.
    Returns:
        :return: (dict) Containing 'duration_sec'. """

    assert isinstance(index, pd.DatetimeIndex), \
        f"argument 'index' must be instance of 'pd.DatetimeIndex', got {type(index)}"
    return {'duration_sec': (index.max() - index.min()).total_seconds()}


def differential_features():
    """ - velocity features: dv/dt = a
    - jerk features:     da/dt
    keep in mind that it's inertial sensor not real accelerometer...
    how would I aggregate features? (because I could simply run each function again for the new timeseries)
    how much gain of information? i.e. information that are not already present in a? """
    raise NotImplementedError
