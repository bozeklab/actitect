""" Functions to generate descriptive motion features from time-series data."""
import gc
from pathlib import Path
import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import List, Any, Optional, Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from . import global_ as global_ff
from . import local_ as local_ff
from .. import utils

logger = logging.getLogger(__name__)

__all__ = ['compute_sleep_features', 'compute_per_night_sleep_features', 'build_aggregated_feature_list']


@dataclass
class FeatureSetup:
    LOG_TIME_INFO_THRESHOLD_SEC: int = 3
    SPECTRAL_TARGET_FREQS: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    SPECTRAL_N_DOM_FREQS: int = 3
    PEAKS_PROM_THRESHOLD: float = .25
    PEAKS_MIN_DISTANCE: float = .2
    AC_PEAK_PROM_THRESHOLD_AUTO_CORR: float = .1
    NOLD_EMB_DIM: int = 3
    CLUSTER_KDE_KWARGS: Dict[str, Any] = field(default_factory=lambda: {'kernel': 'gaussian', 'bandwidth': .04})
    CLUSTER_N_KDE_SAMPLES: int = 400
    CLUSTER_PEAK_KWARGS_KDE: Dict[str, Any] = field(init=False)
    CLUSTER_SLEEP_WAKE_TOLERANCE: float = .075

    def __post_init__(self):
        self.CLUSTER_PEAK_KWARGS_KDE = {
            'prominence': .15, 'distance': int(.1 * (self.CLUSTER_N_KDE_SAMPLES - 1)), 'height': .5, }


class CalcLocalMoveFeatures:
    """ Calculate local, i.e. per movement bout, features from the data."""

    def __init__(self, mode: str, sample_rate: float):
        self.mode = mode
        self.sample_rate = sample_rate
        self.setup = FeatureSetup()
        self.sample_rate = sample_rate
        assert self.mode in ('max_version', 'max_version_with_processing', 'per_night'), \
            f"'mode' must be in ('max_version', 'max_version_with_processing', 'per_night'), got {self.mode}"

    def get_feature_names(self):
        """Helper function to get a complete list of the feature name strings."""
        _feature_func = None
        if self.mode == 'per_night':
            self.sample_rate = self.sample_rate or 100  # set a dummy sample rate so downstream code won't break
            _feature_func = self.per_night_version
        else:
            raise ValueError(f"invalid 'mode' option: '{self.mode}'.")
        return list(_feature_func(*[pd.Series(np.random.normal(0, 1, 100)) for _ in range(4)]).keys())

    def calc_features(self, _movement_episode_df):
        """ Entry point to call the """
        _feature_func = None
        if self.mode == 'per_night':
            _feature_func = self.per_night_version
        else:
            raise ValueError(f"invalid 'mode' option: '{self.mode}'.")
        return _feature_func(
            _movement_episode_df.mag, _movement_episode_df.x, _movement_episode_df.y, _movement_episode_df.z)

    @staticmethod
    def _run_feature_func(feature_func, feature_name: str, warn_threshold: float, *args, **kwargs):
        """ Runs the given feature function, measures execution time, and returns a dict of features. """
        with utils.Timer() as timer:
            features = feature_func(*args, **kwargs)
        elapsed = timer()
        if elapsed > warn_threshold:
            size_str = f"({args[0].shape[0]},)" if args else ""
            logger.warning(f"{feature_name}: {size_str} -> {elapsed:.2f}s")
        return features

    def per_night_version(self, acc_mag, acc_x, acc_y, acc_z):
        for prop in (acc_mag, acc_x, acc_y, acc_z):
            assert isinstance(prop, pd.Series) or isinstance(prop, np.ndarray), \
                f"input must be of type pd.Series or np.array, not {type(prop)}"

        feature_calls = {
            'moments': dict(func=local_ff.moment_features, args=(acc_mag, acc_x, acc_y, acc_z), kwargs={}),
            'quantiles': dict(func=local_ff.quantile_features, args=(acc_mag, acc_x, acc_y, acc_z), kwargs={}),
            'energy': dict(func=local_ff.energy_features, args=(acc_mag, acc_x, acc_y, acc_z), kwargs={}),
            'spectral': dict(func=local_ff.spectral_features, args=(acc_mag,), kwargs=dict(
                n_dom_freqs=self.setup.SPECTRAL_N_DOM_FREQS,
                target_frequencies_hz=self.setup.SPECTRAL_TARGET_FREQS, sample_rate=self.sample_rate)),
            'auto_corr': dict(func=local_ff.autocorr_features, args=(acc_mag, acc_x, acc_y, acc_z), kwargs=dict(
                sample_rate=self.sample_rate, peak_prom_threshold=self.setup.AC_PEAK_PROM_THRESHOLD_AUTO_CORR)),
            'peaks': dict(func=local_ff.peak_features, args=(acc_mag,), kwargs=dict(
                sample_rate=self.sample_rate, peak_prom_threshold=self.setup.PEAKS_PROM_THRESHOLD,
                min_peak_distance=self.setup.PEAKS_MIN_DISTANCE)),
            'nold': dict(func=local_ff.non_linear_dynamic_features, args=(acc_mag, acc_x, acc_y, acc_z),
                         kwargs=dict(emb_dim=self.setup.NOLD_EMB_DIM)),
            'poincare': dict(func=local_ff.poincare_features, args=(acc_mag,), kwargs={})}

        local_features = {}
        for feature_name, spec in feature_calls.items():
            local_features.update(self._run_feature_func(
                spec['func'], feature_name, self.setup.LOG_TIME_INFO_THRESHOLD_SEC, *spec['args'], **spec['kwargs']))
        return local_features


class CalcGlobalMoveFeatures:
    def __init__(self, global_move_segments_df: pd.DataFrame, mode: str, create_cluster_plots: bool,
                 names_only: bool = False):
        self.mode = mode
        self.create_cluster_plots = create_cluster_plots
        self.cluster_fig = None
        self.setup = FeatureSetup()
        if self.mode == 'per_night':
            if not names_only:
                self.features = self.per_night_version(global_move_segments_df)
        else:
            raise ValueError(f"The feature mode {self.mode} is unknown. Currently available are: 'per_night'")

    def get_feature_names(self):
        """Return global (per-night) feature names without requiring a prior compute.

        Mirrors CalcLocalMoveFeatures.get_feature_names by generating a tiny synthetic
        'movement' series, extracting segments via utils.extract_segments, and running
        the same per-night pipeline to obtain keys.
        """
        # Try fast path: probe with a tiny synthetic night that yields a few segments
        try:
            # Build a minimal dummy night with a DateTimeIndex and a 'movement' boolean
            idx = pd.date_range("2000-01-01", periods=600, freq="S")  # 10 minutes @ 1 Hz
            movement = np.zeros(len(idx), dtype=bool)
            # Create a few short "movement bouts"
            movement[50:65] = True
            movement[140:170] = True
            movement[300:315] = True
            movement[480:520] = True

            _dummy = pd.DataFrame({"movement": movement}, index=idx)
            # Use the same extractor as production to match expected schema
            move_segments = utils.extract_segments(_dummy, 'movement', True, time_unit='s')

            # Run the same global feature function to get the output dict keys
            feats = self.per_night_version(move_segments)
            return list(feats.keys())

        except Exception as e:
            # Fallbacks: if we already have computed features, use their keys;
            # otherwise surface a clear error.
            if hasattr(self, "features") and isinstance(self.features, dict) and self.features:
                logger.warning("get_feature_names(): probe failed, falling back to existing features; "
                               f"reason: {e}")
                return list(self.features.keys())
            raise RuntimeError(f"CalcGlobalMoveFeatures.get_feature_names() probe failed and no "
                               f"features available to fall back on: {e}")

    def per_night_version(self, move_segments: pd.DataFrame):
        global_features = {}

        # cluster features:
        _debug = True if self.create_cluster_plots else False
        cluster_feats, debug_info = global_ff.cluster_features(
            move_segments,
            kde_kwargs=self.setup.CLUSTER_KDE_KWARGS,
            peak_kwargs=self.setup.CLUSTER_PEAK_KWARGS_KDE,
            n_samples=self.setup.CLUSTER_N_KDE_SAMPLES,
            sleep_wake_tolerance=self.setup.CLUSTER_SLEEP_WAKE_TOLERANCE,
            debug=_debug
        )
        global_features.update(cluster_feats)

        if self.create_cluster_plots:
            kde, mmp_scaled, density_x, density_y, peaks = debug_info
            x_peaks, y_peaks = density_x[peaks], density_y[peaks]

            plt.rcParams['font.weight'] = 'bold'
            plt.rcParams['axes.labelweight'] = 'bold'
            fig, (lgd_ax, ax) = plt.subplots(2, 1, sharex=True, figsize=(9, 6), height_ratios=(.1, 1))
            plt.subplots_adjust(hspace=0)
            min_, max_ = np.max(density_y) / 40, np.max(density_y) / 5
            ax.scatter(mmp_scaled,
                       MinMaxScaler((min_, max_)).fit_transform(np.random.normal(0, 1, len(mmp_scaled)).reshape(-1, 1)),
                       c='c', edgecolor='k', marker='o', s=15, alpha=.5, label=f'move episodes (midpoint)')
            ax.plot(density_x, density_y, c='k', label=f"kde: {self.setup.CLUSTER_KDE_KWARGS['kernel']}"
                                                       f"(bw={self.setup.CLUSTER_KDE_KWARGS['bandwidth']})")
            # ax.scatter(x_peaks, y_peaks, marker='*', c='r', s=50, zorder=200)
            ax.axvspan(0, self.setup.CLUSTER_SLEEP_WAKE_TOLERANCE, color='r', alpha=.2, zorder=0)
            ax.axvspan(1 - self.setup.CLUSTER_SLEEP_WAKE_TOLERANCE, 1, color='r', alpha=.2, zorder=0,
                       label='excluded sleep on/offset')
            ax.plot(-1, 0, c='b', lw=2, label=f"n_peaks={cluster_feats['kde_n_peaks']}"
                                              f"(prom. > {self.setup.CLUSTER_PEAK_KWARGS_KDE['prominence']})")
            for _xp in x_peaks:
                ax.axvline(_xp, c='b', lw=.7, alpha=.5)
            ax.fill_between(density_x, density_y, color='gray', alpha=.2)
            for key in ('kde_avg_prom', 'kde_min_prom', 'kde_max_prom', 'hopkins_score'):
                ax.plot(0, 0, visible=False, label=f"{key}: {cluster_feats[key]:.3f}")
            ax.set_ylim(0)
            ax.set_xlim(0, 1)
            handles, labels = ax.get_legend_handles_labels()
            lgd_ax.legend(handles=handles, labels=labels, ncols=3, loc='lower left', bbox_to_anchor=(0, 0), fontsize=9)
            lgd_ax.axis('off')
            ax.set_xlabel('normalized sleep-period', labelpad=15, fontsize=13)
            ax.set_ylabel('probability density', labelpad=15, fontsize=13)
            ax.tick_params(which='major', tickdir='in', size=6, width=1.5, labelsize=11, right=True, top=True)
            ax.minorticks_on()
            ax.tick_params(which='minor', tickdir='in', size=3, width=.6, right=True, top=True)
            self.cluster_fig = fig
            plt.close()

        # moves per time:
        global_features.update(global_ff.n_moves_per_time(move_segments))

        # move durations:
        global_features.update(global_ff.movement_durations(move_segments))

        return global_features


def compute_sleep_features(
        processed_df: pd.DataFrame,
        selected_nights: pd.DataFrame,
        move_bout_mask: pd.DataFrame,
        move_bout_ids: pd.DataFrame,
        *,
        sample_rate: Optional[int] = None,
        pbar: Optional = None,
        info: Optional[Dict] = None,
        sample_id: Optional[str] = None,
        movement_duration_filters: Optional[Dict[str, Optional[Tuple[float, float]]]] = None,
        draw_movement_cluster_plot: Optional[bool] = False,
        subject_meta_dict: Optional[Dict] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[int, plt.Figure]]:
    """Compute local and global sleep-related features from a processed actigraphy DataFrame.

    Iterates over all selected nights (SPTWs), computes per-night (global) and per-movement
    (local) features, optionally updates a progress bar, and returns the resulting feature
    tables along with updated processing info and optional cluster plots.

    Parameters;
        :param processed_df: (pandas.DataFrame)
            Processed actigraphy data containing columns such as 'wear', 'sptw', and 'movement'.
        :param selected_nights: (pandas.DataFrame)
            DataFrame of selected SPTWs (one row per night) as returned by `select_night_sptws`.
        :param move_bout_mask: (pandas.DataFrame)
            Mask labeling individual movement bouts across the recording.
        :param move_bout_ids: (pandas.DataFrame)
            Unique movement-bout identifiers corresponding to the mask.
        :param sample_rate: (float, optional)
            Sampling rate in Hz; inferred if None.
        :param pbar: (tqdm object, optional)
            Progress bar instance for progress tracking.
        :param info: (dict, optional)
            Metadata dictionary; updated in place with per-night movement counts.
        :param sample_id: (str, optional)
            Identifier used for logging and progress display.
        :param movement_duration_filters: (Dict[local/global: tuple[float, float]], optional)
            Duration filter for valid movement bouts in seconds. Contains a 'local' and 'global', key value pair,
            where the value is either None (to skip the filter) or a tuple of floats, that represent the filter
             boundaries in seconds.
        :param draw_movement_cluster_plot: (bool, optional)
            If True, generates and returns movement clustering plots per night.
        :param subject_meta_dict: (dict, optional)
            Subject-level metadata (keys: 'patient_id', 'record_id', 'diagnosis').

    Returns:
        :return: (tuple)
            (1) local_feat_df (pandas.DataFrame): per-movement feature table,
            (2) global_feat_df (pandas.DataFrame): per-night feature table,
            (3) info (dict): updated processing information,
            (4) cluster_plot_dict (dict[int, matplotlib.figure.Figure]):optional cluster plots, a dict with one
                entry per night.
    """
    if movement_duration_filters is None:
        movement_duration_filters = {'local': (.5, 50), 'global': None}
    if not sample_rate:
        is_uniform, sample_rate, _ = utils.infer_mean_sample_rate(processed_df)
        if not is_uniform:
            logger.warning(
                "'compute_sleep_features' did not receive 'sample_rate' and inferred"
                " sample rate from 'processed_df' is non uniform.")

    cluster_plot_dict = {}
    create_own_pbar = pbar is None
    optional_pbar = utils.custom_tqdm(total=len(selected_nights)) if create_own_pbar else nullcontext(pbar)

    with optional_pbar as pbar:
        local_feat_df, global_feat_df = pd.DataFrame(), pd.DataFrame()
        if create_own_pbar:
            pbar.set_description(f"[PROGRESS]: Computing sleep features")
        for night_idx, night_sptw in selected_nights.iterrows():  # loop over nights

            # set pbar postfix
            _postfix_dict = {'sample': f"{sample_id}"} if sample_id else {}
            _postfix_dict.update({'night': f"{night_idx + 1}/{selected_nights.shape[0]}"})
            pbar.set_postfix(_postfix_dict)

            # select the subset of the data corresponding to the night
            night_df = processed_df[
                (processed_df.index >= night_sptw['start_time']) & (processed_df.index <= night_sptw['end_time'])]
            move_bout_mask_night = move_bout_mask[
                (move_bout_mask.index >= night_sptw['start_time']) & (move_bout_mask.index <= night_sptw['end_time'])]
            move_bout_ids_night = move_bout_ids[
                (move_bout_ids >= move_bout_mask_night.min()) & (move_bout_ids <= move_bout_mask_night.max())]
            if info is not None:
                info['processing']['sleep_movements'].update({f"n_moves_night_{night_idx}": len(move_bout_ids_night)})

            if not len(move_bout_ids_night) > 10:
                logger.warning(f"not enough movement episodes found in night {night_idx} from sample_id"
                               f" {sample_id or 'none'} (n={len(move_bout_ids_night)}). Check for non-wear.")

            else:
                global_feats, cluster_plot = _compute_global_features_per_night(  # calc. global (per-night) features
                    night_df, night_sptw, night_idx, filter_durations=movement_duration_filters['global'],
                    create_cluster_plots=draw_movement_cluster_plot, subject_meta=subject_meta_dict)
                cluster_plot_dict.update({night_idx: cluster_plot})

                global_feat_df = pd.concat([global_feat_df, pd.DataFrame(global_feats, index=[night_idx])])
                del global_feats
                gc.collect()

                # calculate local (per-move) features:
                feat_names = CalcLocalMoveFeatures(mode='per_night', sample_rate=sample_rate).get_feature_names()
                for idx, relevant_idx in enumerate(move_bout_ids_night):  # loop over movement bouts

                    move_bout_df = night_df[move_bout_mask_night == relevant_idx]

                    if move_bout_df.empty:
                        logger.warning(f"move bouts not found in night_idx: {night_idx}, rel_idx: {relevant_idx}"
                                       f"\n - ids: {move_bout_ids_night}"
                                       f"\n - mask: {move_bout_mask_night}"
                                       f"\n - episode: {move_bout_df}")

                    local_feats = _compute_local_features_per_night(
                        move_bout_df, relevant_idx, night_sptw, night_idx,
                        filter_durations=movement_duration_filters['local'], feature_names=feat_names,
                        sample_rate=sample_rate, subject_meta=subject_meta_dict)

                    chunk = pd.DataFrame(local_feats, index=[idx])
                    local_feat_df = pd.concat([local_feat_df, chunk], axis=0)

                    _postfix_dict = {'sample': f"{sample_id}"} if sample_id else {}
                    _postfix_dict.update({
                        "night": f"{night_idx + 1}/{selected_nights.shape[0]} ",
                        "features": f"{np.round((idx / len(move_bout_ids_night)) * 100, 1)}%"})
                    pbar.set_postfix(_postfix_dict)

                    del local_feats
                    del move_bout_df
                    gc.collect()

            del night_df
            del move_bout_ids_night
            del move_bout_mask_night
            gc.collect()

            if create_own_pbar:
                pbar.update(1)

    return local_feat_df, global_feat_df, info or {}, cluster_plot_dict


def compute_per_night_sleep_features(
        processed_df: pd.DataFrame,
        selected_nights: pd.DataFrame,
        move_bout_mask: pd.DataFrame,
        move_bout_ids: pd.DataFrame,
        *,
        sample_id: Optional[str] = None):
    """Low-level helper to get merged global feature DataFrame."""
    # calculate the features
    local_feat_df, global_feat_df, _, _ = compute_sleep_features(
        processed_df, selected_nights, move_bout_mask, move_bout_ids, sample_id=sample_id)

    local_feat_df = utils.handle_problematic_values_in_feature_df(
        local_feat_df, drop=True, replace=None, df_log_name='local_feat_df', excluded_cols=[
            'id', 'record_id', 'diagnosis', 'time_start', 'time_end', 'time_diff', 'sptw_start',
            'sptw_end', 'sptw_idx', 'night', 'runtime', 'ident'])

    # aggregate local features to night-level
    local_feat_names = CalcLocalMoveFeatures(mode='per_night', sample_rate=None).get_feature_names()
    local_agg_to_global = utils.aggregate_local_feat_df_to_global(
        local_feat_df, local_feature_base_names=local_feat_names)

    # choose the best available join keys; fall back to 'night' if IDs are absent
    candidate_keys = ['id', 'record_id', 'night']
    on_keys = [k for k in candidate_keys if k in local_agg_to_global.columns and k in global_feat_df.columns]
    if not on_keys:
        on_keys = ['night']

    # merge aggregated local with global per-night features
    merged = pd.merge(
        global_feat_df, local_agg_to_global,
        on=on_keys, how='left', suffixes=('', '_dup')
    )

    # if any dup columns snuck in (rare), prefer non-NaN originals
    for col in list(merged.columns):
        if col.endswith('_dup'):
            base = col[:-4]
            if base in merged.columns:
                merged[base] = merged[base].fillna(merged[col])
            merged.drop(columns=[col], inplace=True)

    front = [c for c in ['id', 'record_id', 'night', 'start_time', 'end_time'] if c in merged.columns]
    merged = merged[front + [c for c in merged.columns if c not in front]]

    return merged


def _compute_global_features_per_night(
        night_df: pd.DataFrame,
        night_sptw: pd.DataFrame,
        night_idx: int,
        *,
        filter_durations: Tuple[float, float],
        create_cluster_plots: bool,
        subject_meta: Optional[Dict]) -> pd.DataFrame:
    subject_meta = subject_meta or {}
    # get a representation as start, end, length where each row corresponds to one movement
    night_move_segments = utils.extract_segments(night_df, 'movement', True, time_unit='s')

    if isinstance(filter_durations, tuple):  # apply a duration filter to movement bouts if specified:
        _low, _high = filter_durations
        assert _low <= _high
        logger.info(f"filtering movement durations: {_low}s <= t <= {_high}s")
        night_move_segments = night_move_segments[
            (night_move_segments['length(s)'] > _low) & (night_move_segments['length(s)'] < _high)]

    with utils.Timer() as global_timer:  # calculate global features:
        calc_global_move_features = CalcGlobalMoveFeatures(
            night_move_segments, mode='per_night', create_cluster_plots=create_cluster_plots)
        _global_feats = calc_global_move_features.features

    sptw_duration = night_sptw['length(h)']
    sleep_duration = (night_df.index.to_series().diff()[night_df['sleep_bout']] / pd.Timedelta(hours=1)).sum()
    _global_feats['sleep_eff'] = sleep_duration / sptw_duration
    _global_feats['waso'] = sptw_duration / sleep_duration
    _global_feats['id'] = subject_meta.get('patient_id', 'none')
    _global_feats['record_id'] = subject_meta.get('record_id', 'none')
    _global_feats['diagnosis'] = subject_meta.get('diagnosis', 'none')
    _global_feats['start_time'] = night_sptw['start_time']
    _global_feats['end_time'] = night_sptw['end_time']
    _global_feats['length(h)'] = night_sptw['length(h)']
    _global_feats['sptw_idx'] = night_sptw['sptw_idx']
    _global_feats['night'] = night_idx
    _global_feats['runtime'] = global_timer()
    del night_move_segments

    return _global_feats, calc_global_move_features.cluster_fig or None


def _compute_local_features_per_night(
        move_bout_df: pd.DataFrame,
        relevant_idx: int,
        night_sptw: pd.DataFrame,
        night_idx: int,
        *,
        filter_durations: Tuple[float, float],
        feature_names: np.ndarray,
        sample_rate: float,
        subject_meta: Optional[Dict]) -> pd.DataFrame:
    subject_meta = subject_meta or {}
    with utils.Timer() as local_timer:
        if not (int(filter_durations[0] * sample_rate)
                <= move_bout_df.shape[0]
                <= int(filter_durations[1] * sample_rate)):
            logger.debug(f"timeseries too short/long: {relevant_idx} "
                         f"({(move_bout_df.index[-1] - move_bout_df.index[0]).total_seconds():.3f}s)")
            _local_feats = {_feat_name: np.NaN for _feat_name in feature_names}
        else:  # calculate the movement features of each episode
            _feature_generator = CalcLocalMoveFeatures('per_night', sample_rate)
            _local_feats = _feature_generator.calc_features(move_bout_df)

    _local_feats['id'] = subject_meta.get('patient_id', 'none')
    _local_feats['record_id'] = subject_meta.get('record_id', 'none')
    _local_feats['diagnosis'] = subject_meta.get('diagnosis', 'none')
    _local_feats['time_start'] = move_bout_df.index[0]
    _local_feats['time_end'] = move_bout_df.index[-1]
    _local_feats['time_diff'] = (move_bout_df.index[-1] - move_bout_df.index[0]).total_seconds()
    _local_feats['sptw_start'] = move_bout_df.sptw.iloc[0]
    _local_feats['sptw_end'] = move_bout_df.sptw.iloc[0]
    _local_feats['sptw_idx'] = night_sptw['sptw_idx']
    _local_feats['night'] = night_idx
    _local_feats['runtime'] = local_timer()
    return _local_feats


def build_aggregated_feature_list(
        from_yaml: bool,
        *,
        default_aggregation: List[str] = None,
) -> np.ndarray:
    """Return exhaustive feature names: all GLOBAL names + aggregated LOCAL names.
    Parameters:
        :param from_yaml: (bool): whether to return feature names from YAML config, otherwise build full list.
        :param default_aggregation: (list[str]) of aggregation suffixes applied to each local base feature.
    Returns:
        :return: (np.ndarray) of feature names."""
    default_aggregation = default_aggregation or (
        'mean', 'std', 'skew', 'kurt', 'mad', 'iqr', '10th_percentile', '90th_percentile')
    if from_yaml:
        from actitect.config import ExternalTestConfig
        cfg = ExternalTestConfig.from_yaml(which='external_test')
        local_base = cfg.data.loader.included_local_features
        aggregation = cfg.data.loader.aggregation
        global_names = cfg.data.loader.included_global_features

    else:
        local_base = CalcLocalMoveFeatures(mode='per_night', sample_rate=None).get_feature_names()
        aggregation = default_aggregation
        global_names: List[str] = CalcGlobalMoveFeatures(
            mode='per_night', global_move_segments_df=None, names_only=True, create_cluster_plots=False).get_feature_names()

    local_agg = [f"{name}_{agg}" for name in local_base for agg in aggregation]
    return np.array(list(global_names) + local_agg)


