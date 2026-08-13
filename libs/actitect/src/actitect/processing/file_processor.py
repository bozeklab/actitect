import gc
import json
import logging
from argparse import Namespace
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import resource
import platform

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow
import pyarrow.parquet as parquet

from .movements import segment_nocturnal_movements
from .. import utils
from ..features import compute_sleep_features
from ..processing.sleep import select_night_sptws
from ..vis import draw_actigraphy_data

if TYPE_CHECKING:
    from ..actimeter.settings import ResolveNwSleepParams

MIN_LOCAL_SAMPLE_LENGTH_SECONDS = .5
MAX_LOCAL_SAMPLE_LENGTH_SECONDS = 50

logger = logging.getLogger(__name__)

__all__ = ['FileProcessor']


class FileProcessor:
    # TODO: parallelize on file level
    _delete_confirmation_given = False  # only relevant for 'delete_processed_files' operational modes

    def __init__(self, patient_id: str, record_id: str, label: str, acti_file_path: Path, save_dir: Path,
                 save_processed_data: bool, sleep_log: pd.DataFrame = None, ax6_legacy_mode: bool = False):

        self.patient_id = patient_id
        self.record_id = record_id if record_id and record_id != 'none' else None
        self.saving_suffix = self.patient_id if not self.record_id else f"{self.patient_id}_{self.record_id}"
        self.label = label
        self.acti_file_path = acti_file_path
        self.save_processed_data = save_processed_data
        self.sleep_log = sleep_log
        self.ax6_legacy_mode = ax6_legacy_mode

        self.recording_save_dir = save_dir.joinpath(patient_id) if not self.record_id \
            else save_dir.joinpath(patient_id, self.record_id)
        self.parquet_path = self.recording_save_dir.joinpath(f"df-{self.saving_suffix}.parquet")
        self.info, self.process_kwargs, self.feat_kwargs, self.pbar = None, None, None, None

    def process(self, feat_kwargs: dict, process_kwargs: dict, operational_kwargs: Namespace, pbar=None):
        """ Process the raw data and calculate features based on the specified mode.
        Parameters:
            :param feat_kwargs: (dict)
                Dictionary containing parameters related to the feature calculation process.
            :param process_kwargs: (dict)
                Dictionary containing parameters related to the data processing steps, such as filtering and resampling.
            :param pbar: (tqdm object, optional)
                Progress bar instance for displaying progress. Default is None.
            :param operational_kwargs: (argparse.Namespace) The command line arguments defining the operational mode of
                the processing function. Refer to  main README.md or process_dataset.py for details. """

        self.pbar = pbar
        self.feat_kwargs = feat_kwargs
        self.process_kwargs = process_kwargs
        self._log_memory('start', self.saving_suffix)

        if self.pbar:
            self.pbar.set_description(f"[PROGRESS]: Processing files")
            self.pbar.set_postfix({
                "file": f"{self.saving_suffix}",
                # "save_processed": operational_kwargs.save_processed,
                # "create_plots": operational_kwargs.create_plots,
                # "redo_processing": operational_kwargs.redo_processing,
                # "skip_feature_calc": operational_kwargs.skip_feature_calc,
                # "delete_processed_files": operational_kwargs.delete_processed_files
            })

        # 0.1 handle the deletion case (cleanup mode)
        if operational_kwargs.delete_processed_files:
            self._validate_and_perform_deletion()
            return  # exit process method

        # 0.2 resume behavior (default on): skip work if outputs already exist
        if getattr(operational_kwargs, "resume", True):
            if self._is_complete(operational_kwargs):
                logger.info(f"(io: {self.saving_suffix}): resume=True and outputs exist -> skipping.")
                return

        # 1. apply processing if needed
        processed_df, self.info = self._load_or_process_data(operational_kwargs.redo_processing)
        self._log_memory('after load/preprocessing', self.saving_suffix)

        _processing_is_complete = bool(self.info.get('processing', {}).get('all_steps_successful') is True)
        if not _processing_is_complete:
            msg = f'Incomplete pre-processing detected for {self.saving_suffix}'
            if getattr(operational_kwargs, 'allow_incomplete_preprocessing', False):
                logger.warning(f"{msg} — continuing because --allow_incomplete_preprocessing was set.")
                # leave a breadcrumb in the info dict
                self.info.setdefault('processing', {})['incomplete_preprocessing_allowed'] = True
            else:
                raise UserWarning(f"{msg}, excluding ...")

        # 2. feature calculation (if specified)
        if not operational_kwargs.skip_feature_calc:
            if self.pbar:
                self.pbar.set_postfix({"file": f"{self.saving_suffix}", "status": "extr. movements"})
            selected_nights, move_bout_mask, move_bout_ids = self._segment_nocturnal_movements(processed_df)
            self._log_memory('after movement segmentation', self.saving_suffix)

            local_feat_df, global_feat_df = self._calculate_features(
                processed_df, selected_nights, move_bout_mask, move_bout_ids)
            self._log_memory('after feature calculation', self.saving_suffix)

            _feat_dir = utils.check_make_dir(self.recording_save_dir.joinpath(f"features/{self.feat_kwargs['mode']}/"))
            global_feat_df.to_csv(_feat_dir.joinpath(f"global-features-{self.saving_suffix}.csv"))
            local_feat_df.to_csv(_feat_dir.joinpath(f"local-features-{self.saving_suffix}.csv"))
            utils.dump_to_json(self.info, self.recording_save_dir.joinpath(f"info-{self.saving_suffix}.json"))
            logger.info(f"(io: {self.saving_suffix}): features successfully saved")
            self._log_memory('after feature saving', self.saving_suffix)

            # cleanup memory
            del local_feat_df, global_feat_df, selected_nights
            self._log_memory('after feature cleanup', self.saving_suffix)
        else:
            logger.info(f"(io: {self.saving_suffix}): Skipping feature extraction."
                        f"('args.skip_feature_calc'={operational_kwargs.skip_feature_calc})")

        # 3. plotting (if specified)
        if operational_kwargs.create_plots:
            self._log_memory('before plotting', self.saving_suffix)
            self._plot_data(processed_df)
            self._log_memory('after plotting', self.saving_suffix)

        del processed_df
        gc.collect()
        self._log_memory('after final cleanup', self.saving_suffix)

    def _validate_and_perform_deletion(self):
        """ Delete the processed .parquet file if all derived files exist. prompts for confirmation once."""
        if not FileProcessor._delete_confirmation_given:  # prompt for confirmation on first class instance
            logger.warning("You are about to delete processed .parquet files. This action cannot be undone.")
            confirm = input("Continue? [y/n]: ")
            if confirm.lower() != 'y':
                raise SystemExit("Deletion operation cancelled by the user, terminating...")
            # set global class variable True for all future class instances in this runtime:
            FileProcessor._delete_confirmation_given = True

        if not self.has_been_processed:  # check if files exists in the first place
            logger.warning(f"Processed file for patient {self.saving_suffix} does not exist. Skipping deletion.")
            return  # Nothing to delete

        # define and scan for derived files that should exist(feature files and plots)
        feature_dir = self.recording_save_dir.joinpath("features")
        raw_plot_path = self.parquet_path.parent.joinpath(f"{self.saving_suffix}-raw.png")
        processed_plot_path = self.parquet_path.parent.joinpath(f"{self.saving_suffix}-processed.png")
        # search for feature files within all subdirectories of 'features/'
        global_feature_files = list(feature_dir.rglob(f"global-features-{self.saving_suffix}.csv"))
        local_feature_files = list(feature_dir.rglob(f"local-features-{self.saving_suffix}.csv"))
        required_files = global_feature_files + local_feature_files + [raw_plot_path, processed_plot_path]

        missing_files = [str(file) for file in required_files if not file.exists()]
        if not missing_files:
            try:
                self.parquet_path.unlink()
                logger.warning(f"({self.saving_suffix}) deleted processed data .parquet file {self.parquet_path}.")
            except Exception as e:
                logger.error(f"({self.saving_suffix}) failed to delete {self.parquet_path}: {e}")
        else:
            logger.warning(f"cannot delete {self.parquet_path} as the following derived files are missing:"
                           f"{missing_files}. Skipping deletion.")

    @property
    def has_been_processed(self):
        return self.parquet_path.exists()  # simply checks if the processed file exists

    def _info_path(self) -> Path:
        return self.recording_save_dir.joinpath(f"info-{self.saving_suffix}.json")

    def _feature_dir(self) -> Path:
        return self.recording_save_dir.joinpath(f"features/{self.feat_kwargs['mode']}/")

    def _global_feat_path(self) -> Path:
        return self._feature_dir().joinpath(f"global-features-{self.saving_suffix}.csv")

    def _local_feat_path(self) -> Path:
        return self._feature_dir().joinpath(f"local-features-{self.saving_suffix}.csv")

    def _is_complete(self, operational_kwargs: Namespace) -> bool:
        """Defines what 'done' means for resume mode.
        - If skip_feature_calc is False (default): require info.json + global/local feature CSVs
        - If skip_feature_calc is True: require at least the processed parquet + info.json
          (since features are intentionally not produced)"""
        info_ok = self._info_path().exists()
        if operational_kwargs.skip_feature_calc:
            return info_ok and self.parquet_path.exists()
        # normal full pipeline
        return info_ok and self._global_feat_path().exists() and self._local_feat_path().exists()

    def _load_info(self):
        with open(self.recording_save_dir.joinpath(f"info-{self.saving_suffix}.json"), 'r') as f:
            return json.load(f)

    def _load_or_process_data(self, redo_processing: bool):
        from ..actimeter import ActimeterFactory
        if self.has_been_processed and not redo_processing:  # file exist and we do not want to force re-processing
            processed_df = parquet.read_table(self.parquet_path).to_pandas()
            with open(self.recording_save_dir.joinpath(f"info-{self.saving_suffix}.json"), 'r') as f:
                info = json.load(f)
            logger.info(f"(io: {self.saving_suffix}): using archived pre-processed .parquet file.")

        else:  # otherwise, run processing from raw data
            if self.has_been_processed and redo_processing:
                logger.warning(f"(io: {self.saving_suffix}): processed data exists as .parquet file but will"
                               f"re-run processing due to operational args. Data will be overwritten "
                               f"if 'save_processed_data' is True ({self.save_processed_data}).")
            if self.ax6_legacy_mode:  # factory handles kwargs but this will mute the warning for other devices
                actimeter = ActimeterFactory(self.acti_file_path, self.saving_suffix, legacy_mode=self.ax6_legacy_mode)
            else:
                actimeter = ActimeterFactory(self.acti_file_path, self.saving_suffix)
            utils.check_make_dir(self.recording_save_dir, use_existing=True)
            processed_df = actimeter.process(**self.process_kwargs)

            info = actimeter.get_info()
            if self.save_processed_data:
                parquet.write_table(pyarrow.Table.from_pandas(processed_df), self.parquet_path, compression='snappy')
                utils.dump_to_json(info, self.recording_save_dir.joinpath(f"info-{self.saving_suffix}.json"))
                logger.info(f"(io: {self.saving_suffix}): pre-processed DataFrame saved to .parquet file")
            del actimeter

        return processed_df, info

    def _segment_nocturnal_movements(self, processed_df: pd.DataFrame, params: Optional['ResolveNwSleepParams'] = None):
        from ..actimeter.settings import ResolveNwSleepParams

        params = params if params else ResolveNwSleepParams()

        # get non-wear segments: (start, end, length)
        non_wears = utils.extract_segments(
            processed_df, column='wear', condition=False, add_during_night_indicator=True,
            night_start=params.night_start, night_end=params.night_end, during_night_h_thres=.1)

        selected_sptws = select_night_sptws(processed_df, params=params)
        logger.info(f"({self.saving_suffix}) found {selected_sptws.shape[0]} full nights of sleep in data.")

        # mask the movement bouts:
        move_segment_mask, move_segment_ids, move_stats = segment_nocturnal_movements(processed_df, selected_sptws)

        # update some infos about selected nights and number of movements:
        logger.info(
            f"({self.saving_suffix}) movement segmentation done: n_moves={len(move_segment_ids)} across all nights.")

        self.info['processing']['sleep_movements'].update({f"n_moves_total": len(move_segment_ids)})
        self.info['processing']['sleep_movements'].update(move_stats)
        selected_sptw_dict = selected_sptws.assign(**{'length(h)': lambda x: x['length(h)'].round(4)}).to_dict()
        self.info['processing']['sleep_segmentation'].update({'selected_sptw_nights': {
            index: {key: value_list[index] for key, value_list in selected_sptw_dict.items()}
            for index in range(len(next(iter(selected_sptw_dict.values()))))}})
        self.info['processing']['non-wear'].update({'final_segments': non_wears.to_dict()})

        del non_wears
        gc.collect()
        return selected_sptws, move_segment_mask, move_segment_ids

    def _calculate_features(self, _processed_df: pd.DataFrame, _selected_nights: pd.DataFrame,
                            _move_bout_mask: pd.DataFrame, _move_bout_ids: pd.DataFrame):

        sample_rate = self._get_final_sample_rate()
        meta_dict = {'patient_id': self.patient_id, 'record_id': self.record_id if self.record_id else 'none',
                     'diagnosis': self.label}

        move_duration_filters = {
            'local': (MIN_LOCAL_SAMPLE_LENGTH_SECONDS, MAX_LOCAL_SAMPLE_LENGTH_SECONDS),  # always apply to locals
            'global': self.feat_kwargs.get('filter_move_durations', None)  # restore conditional legacy behavior
        }

        local_feat_df, global_feat_df, updated_info, cluster_plots = compute_sleep_features(
            _processed_df, _selected_nights, _move_bout_mask, _move_bout_ids,
            sample_rate=sample_rate, pbar=self.pbar, info=self.info, sample_id=self.saving_suffix,
            movement_duration_filters=move_duration_filters,
            draw_movement_cluster_plot=self.feat_kwargs['create_cluster_plots'], subject_meta_dict=meta_dict)

        if updated_info is not None and updated_info is not self.info:  # safely update info
            self.info.clear()
            self.info.update(updated_info)

        if cluster_plots:  # none empty dict
            _cluster_dir = utils.check_make_dir(
                self.recording_save_dir.joinpath(f"move_clusters/"), True, verbose=False)
            for night_idx, fig in cluster_plots.items():
                fig.savefig(
                    _cluster_dir.joinpath(f"clusters-{self.saving_suffix}-night{night_idx}.png"), bbox_inches='tight')
                plt.close(fig)
            cluster_plots.clear()
            gc.collect()

        return local_feat_df, global_feat_df

    def _get_final_sample_rate(self) -> float:
        """ Retrieves the final sample rate from either the device header or the processing info.
        Raises a ValueError if neither provides a valid numeric sample rate.
        If both are available and differ, a warning is logged.
        Assumes that self.info has already been populated.

        Returns:
            A numeric sample rate (float).
        """
        assert self.info is not None, \
            "self.info is not defined. Ensure data has been processed before calling this method."

        # try to access header sample rate if defined
        header_fs = self.info.get('header', {}).get('sample_rate')
        header_fs = self._validate_sample_rate_type(header_fs, self.saving_suffix) if header_fs else None

        # try to access the data sampling rate (either with or without resampling)
        data_fs = self.info['processing']['resampling'].get(
            'resample_fs_mean', self.info['processing']['resampling'].get('raw_fs_mean'))
        data_fs = self._validate_sample_rate_type(data_fs, self.saving_suffix) if data_fs else None

        # check if both are undefined
        if header_fs is None and data_fs is None:
            raise ValueError(f"(io: {self.saving_suffix}) No valid sample rate found in header or processing info.")
        # if one is undefined, use the defined one
        if header_fs is None:
            return data_fs
        elif data_fs is None:
            return header_fs
        else:  # both defined: warn if they differ and use data_fs
            if header_fs != data_fs:
                logger.warning(f"(io: {self.saving_suffix}) Device header sample rate ({header_fs} Hz) "
                               f"differs from data sample rate ({data_fs} Hz)."
                               f"Consider activating resampling in the processing pipeline.")
            return data_fs

    @staticmethod
    def _validate_sample_rate_type(sample_rate, log_suffix: str):
        if isinstance(sample_rate, str):
            try:
                sample_rate = float(sample_rate)
            except ValueError:
                logger.warning(f"(io: {log_suffix}) sample rate '{sample_rate}' cannot be converted to float.")
                return None
            return sample_rate
        elif isinstance(sample_rate, (float, int)):
            return sample_rate
        else:
            logger.warning(
                f"(io: {log_suffix}) sample rate '{sample_rate}' has invalid dtype: {type(sample_rate).__name__}")
            return None

    def _plot_data(self, processed_df: pd.DataFrame):
        """Plot raw and processed data, saving the plots to disk."""
        from ..actimeter import ActimeterFactory

        def _save_plot(fig, path, data_type):
            if path.exists():
                logger.info(f"(io: {self.saving_suffix}): {data_type} plot exists and will be overwritten.")

            self._log_memory(f'before {data_type} savefig', self.saving_suffix)
            fig.savefig(path, bbox_inches='tight')
            self._log_memory(f'after {data_type} savefig', self.saving_suffix)

            plt.close(fig)
            del fig
            gc.collect()

            self._log_memory(f'after {data_type} close', self.saving_suffix)

        raw_plot_path = self.parquet_path.parent / f'{self.saving_suffix}-raw.png'
        processed_plot_path = self.parquet_path.parent / f'{self.saving_suffix}-processed.png'

        self._log_memory('before raw loading', self.saving_suffix)
        raw_df = ActimeterFactory(self.acti_file_path, self.saving_suffix).load_raw_data()
        self._log_memory('after raw loading', self.saving_suffix)

        logger.info(
            f'[plot-data] {self.saving_suffix}: raw rows={len(raw_df):,}, '
            f'span={(raw_df.index.max() - raw_df.index.min()) / pd.Timedelta(days=1):.2f} days'
        )

        self._log_memory('before raw draw', self.saving_suffix)
        fig_raw, _ = draw_actigraphy_data(raw_df, self.sleep_log, raw_only=True)
        _save_plot(fig_raw, raw_plot_path, 'raw')
        del fig_raw, _, raw_df
        gc.collect()
        self._log_memory('after raw draw', self.saving_suffix)

        logger.info(
            f'[plot-data] {self.saving_suffix}: processed rows={len(processed_df):,}, '
            f'span={(processed_df.index.max() - processed_df.index.min()) / pd.Timedelta(days=1):.2f} days'
        )

        self._log_memory('before processed draw', self.saving_suffix)
        fig_processed, _ = draw_actigraphy_data(processed_df, self.sleep_log, raw_only=False)
        _save_plot(fig_processed, processed_plot_path, 'processed')
        del fig_processed, _, processed_df
        gc.collect()
        self._log_memory('after processed draw', self.saving_suffix)

        self._log_memory('after plot-data cleanup', self.saving_suffix)

    @staticmethod
    def _log_memory(stage: str, saving_suffix: str) -> None:

        def _rss_gb() -> float:
            """ Current resident memory on Linux. Falls back to resource usage on macOS."""

            if platform.system() == "Linux":
                for line in Path("/proc/self/status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024 ** 2

            # macOS fallback
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

            # macOS reports bytes, Linux reports KiB
            if platform.system() == "Darwin":
                return rss / 1024 ** 3
            return rss / 1024 ** 2

        def _peak_rss_gb() -> float:
            """Peak resident memory (GiB)."""
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if platform.system() == "Darwin":
                return rss / 1024 ** 3  # bytes -> GiB

            return rss / 1024 ** 2  # KiB -> GiB

        logger.info(
            f"[memory] {saving_suffix:<20} | {stage:<30} | "
            f"RSS={_rss_gb():6.2f} GiB | "
            f"Peak={_peak_rss_gb():6.2f} GiB"
        )
