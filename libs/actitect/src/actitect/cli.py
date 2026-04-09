import argparse
import gc
import logging
import traceback
from itertools import chain
from pathlib import Path

import pandas as pd

from . import utils
from .actimeter import SUPPORTED_FILETYPES
from .processing.file_processor import FileProcessor

def _process_dataset(args: argparse.Namespace, processing_kwargs: dict, feature_kwargs: dict, logger: logging.Logger):
    _data_dir = utils.resolve_root_or_abs_path(args.root_dir, args.data_dir)
    _meta_file = utils.resolve_root_or_abs_path(args.root_dir, args.meta_file)
    assert _data_dir.is_dir(), f"the passed 'data_dir' must exist and be a directory! ({_data_dir})."
    assert _meta_file.is_file(), f"can't find 'metadata.csv' file {_meta_file}."

    meta_df = utils.read_meta_csv_to_df(_meta_file)
    _raw_files = sorted(
        p for p in chain.from_iterable(_data_dir.glob(f"*{ext}") for ext in SUPPORTED_FILETYPES)
        if not p.name.startswith("._") and not p.name.startswith("."))

    if not _raw_files:
        raise UserWarning(f"No actigraphy files found at {_data_dir}")
    save_dir = utils.check_make_dir(utils.resolve_root_or_abs_path(args.root_dir, args.out_dir), use_existing=True)

    _excluded = {}
    with utils.custom_tqdm(total=len(_raw_files), position=0, leave=True) as pbar:
        for i, file_path in enumerate(_raw_files):
            _rec_label = file_path.name  # fallback only

            try:
                _meta = meta_df[meta_df['filename'] == file_path.name]
                if _meta.empty or _meta.shape[0] > 1:
                    raise UserWarning(
                        f"missing or double entry in 'data/raw/meta/metadata.csv' for {file_path}, fix and try again"
                    )

                _meta = _meta.iloc[0].to_dict()
                _patient_id = _meta.get('ID') or 'none'
                _record_ID = str(_meta.get('record_ID')).strip() if pd.notna(_meta.get('record_ID')) else "none"
                _diagnosis = _meta.get('diagnosis') or 'none'

                _rec_label = f"{_patient_id} ({_record_ID})" if _record_ID and _record_ID != "none" else _patient_id

                if _meta['exclude'] == 1:
                    _excluded[_rec_label] = "meta"
                    raise UserWarning(f"excluding {_rec_label} according to meta.")

                file_processor = FileProcessor(
                    _patient_id, _record_ID, _diagnosis, file_path, save_dir,
                    save_processed_data=args.save_processed,
                    ax6_legacy_mode=args.ax6_legacy_mode,
                )
                file_processor.process(feature_kwargs, processing_kwargs, args, pbar)
                del file_processor
                gc.collect()

            except NotImplementedError as _ne:
                logger.warning(
                    f"NotImplementedError {_ne} encountered in process_single_file({_rec_label}). "
                    f"Might be caused by unknown 'feature_mode': got '{feature_kwargs.get('feature_mode')}', "
                    f"implemented: 'per_night'."
                )
                logger.warning(f"Excluded: {_rec_label}")
                _excluded[_rec_label] = f"{_ne}"

            except UserWarning as _uw:
                logger.warning(f"UserWarning {_uw} encountered in process_single_file({_rec_label}).")
                logger.warning(f"Excluded: {_rec_label}")
                _excluded[_rec_label] = f"{_uw}"

            except Exception as e:
                logger.warning(f"Exception {e} encountered in process_single_file({_rec_label}).")
                logger.warning(f"Excluded: {_rec_label}")
                _excluded[_rec_label] = f"{e}"
                traceback.print_exc()

            pbar.update(1)

        logger.info(f"\n ...processing finished. Excluded: {_excluded} ({len(_excluded)}) "
                    f"-> n_processed={len(_raw_files) - len(_excluded)}")


def _parse_args():
    """ Parses command-line arguments for preprocessing. Use 'ActiTect-process --help' for details."""
    parser = argparse.ArgumentParser(
        prog='actitect-process',
        description='Preprocess actigraphy data by standardizing, segmenting sleep windows,'
                    ' and calculating nocturnal motion features.',
        formatter_class=argparse.RawTextHelpFormatter)

    # file organization arguments
    parser.add_argument(
        '-r', '--root_dir', type=str,
        default=str(utils.get_experiment_root()),  # default points to ActiTect_experiment
        help="The root directory containing 'data/' and 'ActiTect/'."
    )

    parser.add_argument(
        '-c', '--config_file', type=str,
        default='./actitect/libs/actitect/src/actitect/config/default_preprocessing.yaml',
        help='Path to preprocessing config YAML. Absolute path or relative to –root_dir.'
    )

    parser.add_argument(
        '-d', '--data_dir', type=str,
        default='./data/raw/',
        help="Path to the directory containing raw actigraphy files. Absolute path or relative to –root_dir"
    )

    parser.add_argument(
        '-m', '--meta_file', type=str,
        default='./data/meta/metadata.csv',
        help="Path to metadata.csv. Absolute path or relative to –root_dir."
    )

    parser.add_argument(
        '-o', '--out_dir', type=str,
        default='./data/processed/',
        help="Output directory for processed data/features. Absolute path or relative to –root_dir."
    )

    # operational arguments
    parser.add_argument(
        '-ns', '--no-store', dest='save_processed',
        action='store_false', default=True,
        help='If used, will only save the calculated features, not the processed data.'
    )

    parser.add_argument(
        '-nr','--no-resume', dest='resume',
        action='store_false', default=True,  # True means 'args.resume' is True...
        help='Disable resume behavior. By default, actitect-process will skip records that already have '
             'the expected outputs (features + info.json) on disk.'
    )

    parser.add_argument(
        '-cp', '--create_plots', action='store_true', default=True,
        help='Generate plots during processing or using existing processed data. (default: True).'
    )

    parser.add_argument(
        '-rp', '--redo_processing', action='store_true', default=False,
        help='Force reprocessing of raw data even if preprocessed files exist (default: False).'
    )

    parser.add_argument(
        '-sf', '--skip_feature_calc', action='store_true', default=False,
        help='Skip feature calculation (default: False).'
    )

    parser.add_argument(
        '-del', '--delete_processed_files', action='store_true', default=False,
        help='If used, will delete previously processed .parquet files to free up space. Requires confirmation.'
    )

    parser.add_argument(
        '--ax6_legacy_mode', action='store_true', default=False,
        help='Earlier Ax6 parsing code from Openmovement had some issues with duplicate timestamps and '
             'checksums. Use this flag to reproduce this behavior, for backwards reproducibility only.'
    )

    parser.add_argument(
    '--allow_incomplete_preprocessing', action='store_true', default=False,
        help = 'If set, do NOT exclude records when preprocessing is incomplete'
               ' (i.e., processing["all_steps_successful"] != True).'
               ' Instead, continue with downstream steps (features etc.).')


    return parser.parse_args()


def main():
    args = _parse_args()
    log_path = utils.check_make_dir(utils.resolve_root_or_abs_path(args.root_dir, args.out_dir) / 'logs', True)
    utils.setup_logging(log_file_path=log_path.joinpath('preprocess.log'))
    logger = logging.getLogger(__name__)

    config_params = utils.load_yaml_file(utils.resolve_root_or_abs_path(args.root_dir, args.config_file))
    config_params.update({'args': args})

    _process_dataset(**config_params, logger=logger)


if __name__ == '__main__':
    main()
