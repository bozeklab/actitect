import logging
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from typing import Optional

from .settings import ResolveNwSleepParams
from .. import processing
from .. import utils

__all__ = ['BaseDevice']
logger = logging.getLogger(__name__)


def _processing_step(step_name: str):
    """ Decorator factory to handle error management, logging, and timing for the processing steps
    of the BaseDevice class. All functions decorated with this should raise an Exception if the subprocess fails.
    Parameters:
        step_name (str): The name of the processing step, used for logging and updating processing_info."""

    def decorator(func):
        def wrapper(self, *args, **kwargs):
            processing_info = self.processing_info.get(step_name, {})
            try:
                with utils.Timer() as timer:
                    result = func(self, *args, **kwargs)
                processing_info['time(s)'] = timer()
                processing_info['status'] = 1  # Indicate success
                logger.info(f"({step_name}: {self.meta['patient_id']}) successful. ({timer():.2f}s)")
            except Exception as e:
                error_message = f"{type(e).__name__}: {e}"
                processing_info['status'] = error_message
                logger.error(f"({step_name}: {self.meta['patient_id']}) {error_message}")
                traceback.print_exc()
                result = args[0]  # return the original dataframe on failure
            self.processing_info[step_name] = processing_info
            return result

        return wrapper

    return decorator


class BaseDevice(ABC):
    """Abstract class representing different actimeter devices such as Axivity Ax6, GENEActive etc. Is used to load
    and optionally pre-process data from a given recording of the specified devices. Provides the public methods
    'load_raw_data' and 'process' for this purpose. Subclasses like e.g. Axivity or GENEActive must implement
    the logic to parse the binary data in the abstract methods '__str__' and '_parse_binary_to_df'."""

    _NYQUIST_CUTOFF_FRACTION = 0.95

    def __init__(
            self,
            filepath: Optional[Path],
            patient_id: str,
            resolve_nw_params: ResolveNwSleepParams = None,
            raw_df: Optional[pd.DataFrame] = None,
            header: Optional[dict] = None):

        self.raw_df: pd.DataFrame = None
        self.meta = {'patient_id': patient_id, 'raw_data_loaded': False, 'processed_data': False}
        self.binary_header = {}
        self.processing_info = {'all_steps_successful': None, 'loading': {'filepath': filepath}, 'resampling': {},
                                'filter': {}, 'calibration': {}, 'non-wear': {}, 'sleep_segmentation': {},
                                'sleep_nonwear_ambiguities': {}, 'sleep_movements': {}}
        self.resolve_nw_params = resolve_nw_params if resolve_nw_params is not None else ResolveNwSleepParams()

        if raw_df is not None:
            assert filepath is None, f"if 'raw_df' is provided, the 'filepath' must be None."
            self.set_raw_data(raw_df, header=header)

    def load_raw_data(self, resolve_duplicates: bool = True, header_only: bool = False, *,
                      max_days: Optional[float] = None):
        """ Wrapper for the binary parser. Loads the raw data into time-indexed pd.DataFrame.
        Parameters:
            :param resolve_duplicates: (bool) whether to check and resolve duplicate time indices.
            :param header_only: (bool) if True, will only load the header from the binary, not the data.
            :param max_days: (float, Optional) if set, truncate the data to first 'max_days' days.
        Returns:
            :return:  (pd.Dataframe): the raw data with columns 'x,'y', and 'z' (in units of g) """

        fp = self.processing_info['loading'].get('filepath')
        if self.meta['raw_data_loaded'] and not header_only:  # already loaded or preset, just return it
            logger.info(f"(io: {self.meta['patient_id']}) using preloaded raw data.")
            raw_df, trunc_info = self._truncate_if_too_long(
                self.raw_df, max_days=max_days, log_id=self.meta['patient_id'])
            self.processing_info.setdefault("loading", {}).update(trunc_info)
            return raw_df
        if header_only and self.binary_header:
            logger.info(f"(io: {self.meta['patient_id']}) header already available (preloaded).")
            return self.binary_header

        # otherwise, we need a file
        assert fp is not None and Path(fp).is_file(), f"(io: {self.meta['patient_id']}): File '{fp}' not found."

        try:
            with utils.Timer() as load_timer:
                logger.info(
                    f"(io: {self.meta['patient_id']}) parsing data from"
                    f" '{self.processing_info['loading']['filepath'].name}'.")
                raw_df, header = self._parse_binary_to_df(header_only=header_only)
                raw_df, trunc_info = self._truncate_if_too_long(
                    raw_df, max_days=max_days, log_id=self.meta['patient_id'])
                self.processing_info.setdefault('loading', {}).update(trunc_info)

                if not header_only:
                    utils.assert_valid_df(raw_df)
                    if resolve_duplicates:
                        _info_for_print = {'patient_id': self.meta['patient_id'], 'sample_rate': header['sample_rate']}
                        raw_df = utils.handle_duplicate_timestamps(raw_df, remove_dupes=True, **_info_for_print)
                    _is_uniform, _mean_fs, _std_fs = utils.infer_mean_sample_rate(raw_df)

            if not header_only:
                self.meta['raw_data_loaded'] = True
                self.raw_df, self.binary_header = raw_df, header
                self.processing_info['loading']['time(s)'] = load_timer()
                self.processing_info['resampling'].update({'raw_fs_is_uniform': _is_uniform, 'raw_fs_mean': _mean_fs,
                                                           'raw_fs_std': _std_fs, 'raw_num_ticks': raw_df.shape[0]})

                self._set_logging_start_end_from_raw_df(raw_df, fs_hint=_mean_fs)

                logger.info(f"(io: {self.meta['patient_id']}) successfully loaded raw data. ({load_timer()}s)")
                return raw_df
            else:
                logger.info(f"(io: {self.meta['patient_id']}) successfully loaded data header (raw data not loaded!)")
                return header
        except Exception as e:
            error_message = f"{type(e).__name__}: {e}"
            raise UserWarning(f"(io: {self.meta['patient_id']}) Failed to load raw data: {error_message}")

    def set_raw_data(self, raw_df: pd.DataFrame, header: Optional[dict] = None, resolve_duplicates: bool = True):
        """Inject a preloaded raw DataFrame (bypasses on-disk loading)."""
        utils.assert_valid_df(raw_df)
        if resolve_duplicates:
            info_for_print = {'patient_id': self.meta['patient_id'], 'sample_rate': (header or {}).get('sample_rate')}
            raw_df = utils.handle_duplicate_timestamps(raw_df, remove_dupes=True, **info_for_print)

        is_uniform, mean_fs, std_fs = utils.infer_mean_sample_rate(raw_df)

        self.raw_df = raw_df
        self.binary_header = header or {}
        if 'sample_rate' not in self.binary_header and mean_fs is not None:  # keep downstream logic happy
            try:
                self.binary_header['sample_rate'] = float(mean_fs)
            except Exception:
                pass

        self.meta['raw_data_loaded'] = True
        self.processing_info['resampling'].update({
            'raw_fs_is_uniform': is_uniform,
            'raw_fs_mean': mean_fs,
            'raw_fs_std': std_fs,
            'raw_num_ticks': raw_df.shape[0]})

        self._set_logging_start_end_from_raw_df(raw_df, fs_hint=mean_fs)

    def process(self, resample_rate: Union[int, str] = 'infer', lowpass_hz: float = None, highpass_hz: float = None,
                calibrate: bool = False, detect_nonwear: bool = False, detect_sleep: bool = False,
                max_days: int = None) -> pd.DataFrame:
        """ Run the pipeline to pre-process the raw recorded data based on the specified settings.
        Parameters:
            :param resample_rate: (int or 'infer'') If int, the target sample rate for resampling, if 'infer',the
            resample rate is inferred from the raw data. If None, resampling is skipped.
            :param lowpass_hz: (float) the upper cut-off frequency of the Butterworth filter in Hz. Set None to skip.
            :param highpass_hz: (float) the lower cut-off frequency of the Butterworth filter in Hz. Set None to skip.
            :param calibrate: (bool, Optional) If True, skip the auto-calibration.
            :param detect_nonwear: (bool, Optional) If True, perform non-wear segmentation.
            :param detect_sleep: (bool, Optional) If True, perform sleep segmentation.
            :param max_days: (float, Optional) If set, truncate raw recording to 'max_days' days.
        Returns:
            :return: (pd.DataFrame) the pre-processed data as pd.Dataframe with time-index and
                columns 'x','y','z', 'wear' and 'sleep'. """

        raw_df = self._check_load_raw_data(max_days=max_days)

        # 1. resample to uniform sample rate if necessary and apply lowpass filter:
        if resample_rate is not None:
            y_df = self._resample_uniform(raw_df, resample_rate)
        else:
            y_df = raw_df.copy()
            logger.warning(f"(resample: {self.meta['patient_id']}) skipping resampling as "
                           f"resample_rate is set {resample_rate}")

        if lowpass_hz:
            y_df = self._apply_butterworth_filter(y_df, {'lowcut': None, 'highcut': lowpass_hz}, 'lowpass')

        # 2. perform VanHees2014 'sphere' gravity calibration:
        if calibrate:
            y_df = self._auto_calibrate(y_df)

        # 4. segment non-wear episodes and sleep windows:
        if detect_nonwear:
            y_df = self._infer_nonwear_segments(y_df)
        if detect_sleep:
            y_df = self._segment_sleep_windows(y_df)
        # 5. applying a Butterworth highpass filter to remove gravity component:
        # (sleep/non-wear detection performs better this way around since
        # it requires axis-alignment information for z-angle)
        if highpass_hz:
            y_df = self._apply_butterworth_filter(y_df, {'lowcut': highpass_hz, 'highcut': None}, 'highpass')

        self.meta['processed_data'] = True

        # noinspection PyTypedDict
        self.processing_info['all_steps_successful'] = \
            all(info.get('status') == 1 for key, info in self.processing_info.items()
                if key in ['resampling', 'filter', 'calibration', 'non-wear', 'sleep_segmentation'])

        return y_df

    @_processing_step('resampling')
    def _resample_uniform(self, x_df: pd.DataFrame, _resample_rate: Union['str', 'int'] = 'infer'):
        """ Resamples the data to a specified sampling frequency if needed, e.g. if the data sampling is non-uniform
        due to internal clock drifts of the device and correct frequency preservation is critical.
        Parameters:
            :param x_df: (pd.DataFrame) the data with the original sampling.
            :_resample_rate: ('infer' or int): see 'resample_rate' in BaseDevice.process().
        Returns:
            :return: the data with resampled time-index."""
        _raw_fs_is_uniform = self.processing_info['resampling']['raw_fs_is_uniform']
        _raw_fs = self.processing_info['resampling']['raw_fs_mean']
        target_fs = (np.floor(_raw_fs / 10) * 10 if _raw_fs % 10 < 5 else np.ceil(_raw_fs / 10) * 10) \
            if _resample_rate == 'infer' else _resample_rate  # if 'infer', take closest fs%10=0
        header_fs = self.binary_header.get('sample_rate')
        if isinstance(header_fs, str):
            try:
                header_fs = float(header_fs)
            except ValueError:
                logger.warning(f"(resampling: {self.meta['patient_id']}) '"
                               f"header_fs' is stored as str but cannot be converted to float: '{header_fs}'")
                target_fs = None

        if header_fs is not None and isinstance(target_fs, (float, int)):
            threshold = .1 * header_fs  # threshold as 10% of header sample rate
            if abs(header_fs - target_fs) > threshold:
                logger.warning(f"(resampling: {self.meta['patient_id']}) Device header sample_rate={header_fs:.2f} "
                               f"Hz differs significantly from target_fs={target_fs:.2f} Hz. Likely due to significant "
                               f"non-wear periods in data. Falling back to header sample rate.")
                target_fs = header_fs

        _apply_resampling = False
        if not _raw_fs_is_uniform:  # check if raw data has uniform fs
            logger.info(f"(resampling: {self.meta['patient_id']}) non-uniform sampling rate in raw data detected: "
                        f"fs = {_raw_fs} ± {self.processing_info['resampling']['raw_fs_std']} Hz")
            if not _resample_rate:
                logger.info(f"(resampling: {self.meta['patient_id']}) uniform resampling is recommended."
                            f"Please specify an int target rate or use 'infer'.")
            else:  # uniform resampling rate -> perform resampling
                _apply_resampling = True

        else:  # raw data already has uniform fs
            logger.info(f"(resampling: {self.meta['patient_id']}) raw data already has uniform sampling rate:"
                        f"fs = {self.processing_info['resampling']['raw_fs_mean']} "
                        f"± {self.processing_info['resampling']['raw_fs_std']} Hz")

            if not (_resample_rate is None or _resample_rate == 'infer'):  # check if target rate is specified
                logger.info(f"(resampling: {self.meta['patient_id']}) raw data already has uniform sampling rate:"
                            f"fs = {_raw_fs_is_uniform} ± {self.processing_info['resampling']['raw_fs_std']} Hz.")

                if np.isclose(_raw_fs, _resample_rate, rtol=1e-5, atol=1e-8):  # raw fs matches target
                    logger.info(f"(resampling: {self.meta['patient_id']}) raw sample rate already matches specified"
                                f"target ({_resample_rate} Hz). Skipping resampling.")
                else:  # data is uniform but should be resampled to different frequency
                    _apply_resampling = True

        if _apply_resampling:
            x_df = processing.resample_df_uniform(x_df, target_fs)
            utils.assert_valid_df(x_df)
            _is_uniform, _mean_fs, _std_fs = utils.infer_mean_sample_rate(x_df)
            if not (np.isclose(_mean_fs, target_fs, rtol=1e-5, atol=1e-8) or _is_uniform):
                _status = (f"error: resampling rate ({_mean_fs:.2f} Hz does not match target ({target_fs:.2f} Hz))"
                           f"or is not uniform (std = {_std_fs} Hz).")
                raise RuntimeError(f"Resampling failed: {_status}")

            self.processing_info['resampling'].update(
                {'resample_fs_is_uniform': _is_uniform, 'resample_fs_mean': _mean_fs,
                 'resample_fs_std': _std_fs, 'resample_num_ticks': x_df.shape[0]})
        return x_df

    @_processing_step('filter')
    def _apply_butterworth_filter(self, x_df: pd.DataFrame, kwargs, filter_name: str):
        """Apply a Butterworth low- or high-pass filter to the data.

        For a requested low-pass cutoff at or above Nyquist, cap the effective
        cutoff at a small margin below Nyquist instead of skipping the filter.
        The requested and effective cutoffs are both recorded for provenance.
        """
        kwargs = dict(kwargs)

        resampled_fs = self.processing_info['resampling'].get('resample_fs_mean')
        raw_fs = self.processing_info['resampling'].get('raw_fs_mean')
        fs = resampled_fs if resampled_fs is not None else raw_fs
        if fs is None or not np.isfinite(fs) or fs <= 0:
            raise RuntimeError(f"{filter_name} failed: invalid sampling rate '{fs}'")

        fs = float(fs)
        kwargs['fs'] = fs

        requested_highcut = kwargs.get('highcut')
        effective_highcut = requested_highcut
        nyquist_hz = fs / 2.0
        cutoff_adjusted = False

        if filter_name == 'lowpass' and requested_highcut is not None and requested_highcut >= nyquist_hz:
            effective_highcut = self._NYQUIST_CUTOFF_FRACTION * nyquist_hz
            kwargs['highcut'] = effective_highcut
            cutoff_adjusted = True
            logger.warning(
                f"(filter: {self.meta['patient_id']}) requested low-pass cutoff "
                f"{requested_highcut:.3f} Hz is not below Nyquist ({nyquist_hz:.3f} Hz at fs={fs:.3f} Hz). "
                f"Using Nyquist-safe cutoff {effective_highcut:.3f} Hz "
                f"({self._NYQUIST_CUTOFF_FRACTION:.0%} of Nyquist)."
            )

        x_df, info = processing.butterworth_bandpass(x_df, **kwargs)
        info.update({
            'kwargs': kwargs.copy(),
            'requested_highcut_hz': requested_highcut,
            'effective_highcut_hz': effective_highcut,
            'nyquist_hz': nyquist_hz,
            'cutoff_adjusted_to_nyquist': cutoff_adjusted,
        })
        self.processing_info['filter'].update({filter_name: info})

        ok = info.get('ok', None)
        if ok in (0, False):
            status = info.get('status', 'filter_failed')
            msg = info.get('status_msg', '')
            raise RuntimeError(f"{filter_name} failed: {status}" + (f" ({msg})" if msg else ""))

        utils.assert_valid_df(x_df)
        return x_df

    @_processing_step('calibration')
    def _auto_calibrate(self, x_df: pd.DataFrame):
        """ Perform auto-calibration to align axis."""
        x_df, _info = processing.van_hees_sphere_calibration(x_df)
        utils.assert_valid_df(x_df)
        calib_ok = _info.get('calib_ok', 0)
        if calib_ok == 0:
            reason = _info.get('calib_status', 'unknown_failure')
            raise RuntimeError(f'Calibration failed: {reason}')

        logger.info(f"(calibration: {self.meta['patient_id']}) calibration error summary: "
                    f"before={_info['calib_error_before(mg)']:.2f}mg after={_info['calib_error_after(mg)']:.2f}mg."
                    f"({_info['calib_num_iter']} iterations)")
        self.processing_info['calibration'].update(_info)
        return x_df

    @_processing_step('non-wear')
    def _infer_nonwear_segments(self, x_df: pd.DataFrame):
        """Find non-wear segments in the data. Will add a boolean 'wear' column to the returned DataFrame."""
        x_df, _info = processing.segment_non_wear_episodes(x_df)
        utils.assert_valid_df(x_df)

        logger.info(f"(non-wear: {self.meta['patient_id']}) wear-time-segmentation summary:"
                    f"wear={_info['total_wear_time(d)']:.2f}d, non-wear={_info['non_wear_time(d)']:.2f}d"
                    f"({_info['num_non_wear_episodes']} episodes)")
        self.processing_info['non-wear'].update(_info)

        return x_df

    @_processing_step('sleep_segmentation')
    def _segment_sleep_windows(self, x_df: pd.DataFrame):
        """ Detect sleep segments in data, updates 'x_df' with boolean 'sleep' column and returns it.
        NOTE: Overwrites info['sleep_segmentation']['sptws'] with SPTWs re-derived from the FINAL
        boolean masks in x_df (after ambiguity resolution), using utils.extract_segments. """
        _fs = (self.processing_info['resampling'].get('resample_fs_mean')
               or np.ceil(self.processing_info['resampling'].get('raw_fs_mean'))).astype('int')

        detector = processing.SleepDetector()
        x_df, _info, _sptw = detector.fit(x_df, sample_rate=_fs)

        _durations_h = np.array([sptw.get('duration(h)', np.nan) for sptw in _info['sptws'].values()])
        _durations_above_5h = _durations_h[np.where(_durations_h > 5)]
        logger.info(f"(sleep-segmentation: {self.meta['patient_id']}) sleep-segmentation summary:"
                    f"found n={len(_durations_h):.0f} sptws with n={len(_durations_above_5h):.0f} above 5h with "
                    f"durations {np.nanmean(_durations_above_5h):.1f}±{np.nanstd(_durations_above_5h):.1f}h")
        utils.assert_valid_df(x_df)
        if 'wear' in x_df.columns:
            x_df, (_init_num_amb, _final_num_amb) = self._resolve_nonwear_sleep(x_df, _sptw, self.resolve_nw_params)
            if _init_num_amb:
                logger.info(f"(sleep: {self.meta['patient_id']}) found non-wear/sleep ambiguities,"
                            f"resolved {_init_num_amb} -> {_final_num_amb}")
                self.processing_info['sleep_nonwear_ambiguities'].update({'num_init_segments': _init_num_amb,
                                                                          'num_final_segments': _final_num_amb})
        else:
            logger.warning(f"(sleep: {self.meta['patient_id']}) it is recommended to perform "
                           f"non-wear segmentation before sleep analysis.")

        try:  # overwrite the sptws in info with updated/resolved sptws, treating final df as source of truth
            seg_sptw = utils.extract_segments(  # get sptw segments: (start, end, length)
                x_df, column='sptw', condition=True, add_during_night_indicator=True,
                night_start=self.resolve_nw_params.night_start, night_end=self.resolve_nw_params.night_end,
                during_night_h_thres=self.resolve_nw_params.night_sptw_duration_during_night_h)

            if seg_sptw is None or seg_sptw.empty:
                sptws_new = {}
                num_sptws_new = 0
            else:  # normalize expected columns from helper
                len_col = 'length(h)'
                if len_col not in seg_sptw.columns:  # be defensive if helper changes
                    for c in seg_sptw.columns:
                        if c.startswith('length(') and c.endswith(')'):
                            len_col = c
                            break

                sptws_new = {}
                for i, row in seg_sptw.reset_index(drop=True).iterrows():
                    st = pd.Timestamp(row['start_time'])
                    et = pd.Timestamp(row['end_time'])
                    dur_h = float(row.get(len_col, np.nan))

                    # simple overnight heuristic: crosses date OR end earlier-than-start clock time
                    overnight = (st.normalize() != et.normalize()) or (et.hour < st.hour)

                    sptws_new[int(i)] = dict(
                        start_time=st,
                        end_time=et,
                        overnight=bool(overnight),
                        **{"duration(h)": dur_h},
                    )

                num_sptws_new = len(sptws_new)

            num_sleep_bouts_new = None  # optionally also refresh sleep-bout count from final mask
            if 'sleep_bout' in x_df.columns:
                seg_bouts = utils.extract_segments(
                    x_df, column='sleep_bout', condition=True, add_during_night_indicator=False)
                num_sleep_bouts_new = 0 if (seg_bouts is None or seg_bouts.empty) else int(len(seg_bouts))

            # overwrite / update info
            _info = dict(_info)  # ensure mutable copy
            _info['sptws'] = sptws_new
            _info['num_sptws'] = int(num_sptws_new)
            if num_sleep_bouts_new is not None:
                _info['num_sleep_bouts'] = int(num_sleep_bouts_new)

        except Exception as e:
            logger.warning(
                "(sleep-segmentation: %s) Failed to overwrite SPTWs from final mask (%s: %s). "
                "Keeping detector SPTWs in info.",
                self.meta['patient_id'], type(e).__name__, e
            )

        del _sptw

        self.processing_info['sleep_segmentation'].update(_info)
        return x_df

    @staticmethod
    def _resolve_nonwear_sleep(x_df: pd.DataFrame, sptw: pd.DataFrame, params: ResolveNwSleepParams):
        """ Resolves ambiguities between non-wear periods and sleep detection (i.e. 'non_wear' and 'sleep' are
        'True') by applying heuristic rules. If none of these rules applies, the ambiguity remains and is marked

        Parameters:
            :param x_df: (pd.DataFrame): The accelerometer data with a datetime index, containing
                boolean 'wear' and 'sptw' columns indicating wear status and sleep periods.
            :param sptw:  (pd.DataFrame): Sleep period time windows with 'start_time' and 'end_time' columns.
            :param params: (ResolveNwSleepParams): Parameters and thresholds defining the heuristic rules.

        Returns:
            :return: Tuple[pd.DataFrame, Tuple[int, int]]: A tuple containing:
                - Updated `x_df` DataFrame with corrected 'wear' and 'sptw' columns.
                - A tuple with two integers:
                    - `num_init_ambiguities`: The number of initial ambiguous segments detected.
                    - `num_remaining_ambiguities`: The number of ambiguities remaining after resolution.

        Notes:
            - The function works by:
                1. Identifying ambiguous segments where both non-wear and sleep are detected.
                2. Grouping these segments and calculating their duration and whether they occur during the night.
                3. Iterating over each ambiguous segment to determine the appropriate correction based on:
                    - Whether the ambiguity is at the start or end of the recording.
                    - The duration of the ambiguity.
                    - Whether the ambiguity occurs during the night or day.
                    - Correspondence with underlying sleep periods (`sptw`).
                4. Adjusting the 'wear' and 'sptw' columns in `x_df` accordingly to resolve the ambiguities. """
        mask = (x_df['wear'] == False) & (x_df['sptw'] == True)  # find ambig. segments, i.e. non-wear & sleep detected
        num_init_ambiguities, num_remaining_ambiguities = None, None
        if mask.any():
            ambiguity_ids = (mask != mask.shift()).cumsum()
            ambiguities = pd.DataFrame({
                'start_time': x_df[mask].groupby(ambiguity_ids).apply(lambda x: x.index.min()),
                'end_time': x_df[mask].groupby(ambiguity_ids).apply(lambda x: x.index.max())
            }).reset_index(drop=True)
            ambiguities['length(h)'] = (ambiguities['end_time'] - ambiguities['start_time']) / pd.Timedelta(hours=1)
            ambiguities['during_night'] = (  # check if ambiguity is completely within night window
                    ((params.night_start <= ambiguities['start_time'].dt.hour) & (
                            ambiguities['start_time'].dt.hour < 24) |
                     (0 <= ambiguities['start_time'].dt.hour) & (ambiguities['start_time'].dt.hour < params.night_end))
                    &
                    ((params.night_start <= ambiguities['end_time'].dt.hour) & (ambiguities['end_time'].dt.hour < 24) |
                     (0 <= ambiguities['end_time'].dt.hour) & (ambiguities['end_time'].dt.hour < params.night_end))
            )

            num_init_ambiguities = ambiguities.shape[0]
            num_remaining_ambiguities = num_init_ambiguities
            for index, ambiguity_row in ambiguities.iterrows():  # loop over segments and match to underlying sptw:
                corresponding_sptw = sptw[(sptw['start_time'] <= ambiguity_row['start_time'])
                                          & (sptw['end_time'] >= ambiguity_row['end_time'])]

                if not corresponding_sptw.empty:  # loop over the segments and match to the underlying sptw:
                    if ambiguity_row['start_time'] < (x_df.index[0] + pd.Timedelta(params.thres_onset, unit='h')):
                        '''very likely non-wear mistaken for sleep (e.g. before patients wear device,
                        patients are awake when device is put on,
                        and it is likely that sleep only occurs after certain threshold)
                        so set total sleep before end of ambiguity false.'''
                        x_df.loc[(x_df.index <= ambiguity_row['end_time']), ['sptw', 'sleep_bout']] = False
                        num_remaining_ambiguities -= 1

                    # cases where ambiguity is at end of whole recording:
                    elif ambiguity_row['end_time'] > (x_df.index[-1] - pd.Timedelta(params.thres_offset, unit='h')):
                        # likely non-wear mistaken for sleep, e.g. patient stops wearing device before it's turned off
                        if (corresponding_sptw['overnight'].any()
                                and ambiguity_row['length(h)'] > params.thres_non_wear_is_short):
                            # amb. is long & sptw overnight, e.g. whole night misclassified -> set whole sptw False
                            x_df.loc[(x_df.index >= corresponding_sptw['start_time'].iloc[0])
                                     & (x_df.index <= corresponding_sptw['end_time'].iloc[0]),
                            ['sptw', 'sleep_bout']] = False
                            num_remaining_ambiguities -= 1

                        else:
                            '''amb. is rel. short, e.g. only at morning -> set amb. sleep false
                            x_df.loc[(x_df.index >= ambiguity_row['start_time']), ['sptw', 'sleep_bout']] = False
                            or set whole sleep on last day after 8am false'''
                            x_df.loc[(x_df.index >= x_df.index[-1].replace(hour=8, minute=0, second=0)),
                            ['sptw', 'sleep_bout']] = False
                            num_remaining_ambiguities -= 1

                    else:  # ambiguity not at on/offset of recording
                        if (ambiguity_row['during_night'] and
                                ambiguity_row['length(h)'] < params.thres_non_wear_is_short):
                            # cases where amb. is during night and relatively short
                            # very likely sleep mistaken for non-wear so correct the wear column
                            x_df.loc[(x_df.index >= ambiguity_row['start_time'])
                                     & (x_df.index <= ambiguity_row['end_time']), 'wear'] = True
                            num_remaining_ambiguities -= 1

                        elif not ambiguity_row['during_night'] and (
                                ambiguity_row['start_time'].hour < params.night_end  # not <= since .hour rounds value
                                or ambiguity_row['end_time'].hour > params.night_start):

                            # amb. starts/ends before night window ends/starts -> likely non-wear after/before sleep
                            x_df.loc[(x_df.index >= ambiguity_row['start_time'])
                                     & (x_df.index <= ambiguity_row['end_time']), ['sptw', 'sleep_bout']] = False
                            num_remaining_ambiguities -= 1

                        elif not ambiguity_row['during_night']:  # all cases where amb. is completely during day window
                            if ((corresponding_sptw['overnight'].all()
                                 and ambiguity_row['length(h)'] > params.thres_non_wear_is_short)
                                    or not corresponding_sptw['overnight'].all()):
                                # sptw overnight and amb. is long or sptw not overnight
                                # whole sptw probably wrongly classified
                                x_df.loc[(x_df.index >= corresponding_sptw['start_time'].iloc[0])
                                         & (x_df.index <= corresponding_sptw['end_time'].iloc[0]),
                                ['sptw', 'sleep_bout']] = False
                                num_remaining_ambiguities -= 1

                            elif (corresponding_sptw['overnight'].all()  # sptw overnight but amb. is rel. short
                                  and ambiguity_row['length(h)'] < params.thres_non_wear_is_short):
                                x_df.loc[  # mis-classified sleep non-wear transition -> set amb. sleep false
                                    (x_df.index >= ambiguity_row['start_time'])
                                    & (x_df.index <= ambiguity_row['end_time']),
                                    ['sptw', 'sleep_bout']
                                ] = False
                                num_remaining_ambiguities -= 1

            del ambiguities
            del corresponding_sptw
            del mask

        return x_df, (num_init_ambiguities, num_remaining_ambiguities)

    def _check_load_raw_data(self, *, max_days: Optional[float] = None):
        if not self.meta['raw_data_loaded']:
            return self.load_raw_data(max_days=max_days)
        else:
            raw_df, trunc_info = self._truncate_if_too_long(
                self.raw_df, max_days=max_days, log_id=self.meta['patient_id'])
            self.processing_info.setdefault("loading", {}).update(trunc_info)
            return raw_df

    def _set_logging_start_end_from_raw_df(
            self, raw_df: pd.DataFrame, *,
            fs_hint: Optional[float] = None, store_under: str = 'processing.loading') -> None:
        """Derive logging start/end from the raw_df datetime index and store it in processing_info.
            - loggingStartTime: first timestamp in raw_df
            - loggingEndTime:   end-exclusive timestamp (last timestamp + one sample interval if fs is known)"""
        if raw_df is None or raw_df.empty:
            logger.warning("(io: %s) raw_df empty; cannot derive logging start/end.", self.meta["patient_id"])
            return

        if not isinstance(raw_df.index, pd.DatetimeIndex):
            logger.warning(
                "(io: %s) raw_df index is not a DatetimeIndex (%s); cannot derive logging start/end.",
                self.meta["patient_id"], type(raw_df.index))
            return

        start, last = raw_df.index.min(), raw_df.index.max()

        # determine sample interval for end-exclusive timestamp
        fs = None
        # i. header fs if usable
        header_fs = self.binary_header.get("sample_rate")
        if isinstance(header_fs, (int, float)) and np.isfinite(header_fs) and header_fs > 0:
            fs = float(header_fs)

        # ii. explicit hint if provided
        if fs is None and fs_hint is not None and np.isfinite(fs_hint) and fs_hint > 0:
            fs = float(fs_hint)

        # iii. compute from index diffs (fallback)
        if fs is None:
            try:
                diffs = raw_df.index.to_series().diff().dropna().dt.total_seconds().to_numpy()
                if diffs.size:
                    dt_s = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else float(np.median(diffs))
                    if np.isfinite(dt_s) and dt_s > 0:
                        fs = 1.0 / dt_s
            except Exception:
                fs = None

        if fs is not None and fs > 0:
            end = last + pd.to_timedelta(1.0 / fs, unit="s")  # end-exclusive
        else:  # no fs → fall back to inclusive last timestamp (still useful)
            end = last

        # store, default: processing_info['loading'][...]
        if store_under == 'processing.loading':
            tgt = self.processing_info.setdefault('loading', {})
        else:
            # optionally allow other nesting later
            tgt = self.processing_info.setdefault('loading', {})

        tgt['loggingStartTime'], tgt['loggingEndTime'] = pd.Timestamp(start), pd.Timestamp(end)
        try:
            tgt['loggingDuration(s)'] = float((pd.Timestamp(end) - pd.Timestamp(start)).total_seconds())
            tgt['loggingDuration(h)'] = float(tgt['loggingDuration(s)'] / 3600.0)
        except Exception:
            pass

    def get_info(self):
        return {'meta': self.meta, 'header': self.binary_header, 'processing': self.processing_info}

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def _parse_binary_to_df(self):
        """Abstract method to parse the binary data from different devices and return it as a standardized DataFrame."""
        raise NotImplementedError

    @classmethod
    def from_dataframe(cls, raw_df: pd.DataFrame, patient_id: str, header: Optional[dict] = None,
                       resolve_nw_params: ResolveNwSleepParams = None):
        inst = cls(filepath=None, patient_id=patient_id, resolve_nw_params=resolve_nw_params)
        inst.set_raw_data(raw_df, header=header)
        return inst

    @staticmethod
    def _span_days(x_df: Optional[pd.DataFrame]) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], float]:
        if x_df is None or x_df.empty:
            return None, None, float("nan")
        t0, t1 = x_df.index.min(), x_df.index.max()
        return pd.Timestamp(t0), pd.Timestamp(t1), float((t1 - t0).total_seconds() / 86_400)

    def _truncate_if_too_long(
            self, x_df: pd.DataFrame, *, max_days: Optional[float], log_id: str) -> tuple[pd.DataFrame, dict]:
        """Truncates a given dataframe to 'max_days'. Used in cases where device was not turned off after wear."""
        t0, t1, span_days = self._span_days(x_df)
        info = {
            'raw_span_days_pre_trunc': span_days,
            'raw_start_pre_trunc': t0,
            'raw_end_pre_trunc': t1,
            'raw_trunc_max_days': max_days,
            'raw_was_truncated': False,
        }

        if not max_days or x_df is None or x_df.empty or not np.isfinite(span_days):
            info.update({
                'raw_span_days_post_trunc': span_days,
                'raw_start_post_trunc': t0,
                'raw_end_post_trunc': t1,
            })
            return x_df, info

        if span_days <= max_days:
            info.update({
                'raw_span_days_post_trunc': span_days,
                'raw_start_post_trunc': t0,
                'raw_end_post_trunc': t1,
            })
            return x_df, info

        cutoff = t0 + pd.Timedelta(days=max_days)
        logger.warning(f"(io: {log_id}) recording spans {span_days:.2f} days. Truncating to first {max_days} days.")

        y_df = x_df.loc[x_df.index <= cutoff]
        t0b, t1b, span_days_b = self._span_days(y_df)

        info.update({
            'raw_was_truncated': True,
            'raw_trunc_cutoff': pd.Timestamp(cutoff),
            'raw_span_days_post_trunc': span_days_b,
            'raw_start_post_trunc': t0b,
            'raw_end_post_trunc': t1b,
            'raw_num_ticks_pre_trunc': int(x_df.shape[0]),
            'raw_num_ticks_post_trunc': int(y_df.shape[0]),
        })
        return y_df, info

