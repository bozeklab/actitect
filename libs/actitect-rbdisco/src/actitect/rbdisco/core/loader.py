import logging
from collections import defaultdict
from pathlib import Path
from typing import Union, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from actitect import utils
from actitect.config import RebalanceDatasetsConfig
from actitect.rbdisco.configs import get_label_mappings
from .types import FeatureSet

__all__ = ['DataLoader']

logger = logging.getLogger(__name__)


def _build_code_to_label_map(mapping_name: str) -> dict[str, str]:
    mappings = get_label_mappings()
    if mapping_name not in mappings.mappings:
        raise ValueError(
            f"Unknown str_label_mapping='{mapping_name}'. "
            f"Available mappings: {sorted(mappings.mappings.keys())}"
        )

    label_map = mappings.mappings[mapping_name]
    code_to_label = {}
    for label, codes in label_map.items():
        for code in codes:
            code_norm = str(code).strip().lower()
            if code_norm in code_to_label and code_to_label[code_norm] != label:
                raise ValueError(
                    f"Ambiguous code '{code}' in mapping '{mapping_name}': "
                    f"{code_to_label[code_norm]!r} vs {label!r}"
                )
            code_to_label[code_norm] = label
    return code_to_label


class DataLoader:
    """Manages the retrieval of local and global feature files, processes metadata,
    applies feature aggregation strategies, and prepares train/test datasets."""

    def __init__(self, data_dir: Union[Path, List[Path]], meta_path: Union[Path, pd.DataFrame],
                 feature_dir: Path, aggregation: List[str],
                 included_local_features: List[str], included_global_features: List[str],
                 binary_mapping: dict, add_str_labels: bool = False, verbose: bool = True, shuffle: bool = True,
                 rebalance_datasets: RebalanceDatasetsConfig = None, group_level: str = 'patient',
                 str_label_mapping: Optional[str] = None):

        self.dataset_is_pooled = isinstance(data_dir, list) and len(data_dir) > 1
        self.data_dirs = data_dir if isinstance(data_dir, list) else [data_dir]
        self.feature_dir = feature_dir  # relative to patient_dir
        self.str_label_mapping = str_label_mapping
        self.local_feat_files = sorted([
            file for base_dir in self.data_dirs  # loop over patient dirs from datasets
            for file in list(base_dir.glob(f"*/{feature_dir}/*local*.csv"))  # single record/patient
                        + list(base_dir.glob(f"*/*/{feature_dir}/*local*.csv"))  # multi-record
        ])

        if group_level not in ('patient', 'record'):
            raise ValueError(f"group_level must be 'patient' or 'record', got: {group_level!r}")
        self.group_level = group_level

        if len(self.local_feat_files) == 0:
            raise UserWarning(f"could not locate local feature files make sure '{self.data_dirs}' exists and "
                              f"contains patient subdirs populated with the feature files.")

        self.aggregation = aggregation
        self.binary_mapping = binary_mapping
        self.included_local_features = included_local_features
        self.included_global_features = included_global_features
        self.verbose = verbose
        self.add_str_labels = add_str_labels
        if isinstance(rebalance_datasets, dict):
            rebalance_datasets = RebalanceDatasetsConfig.from_dict(rebalance_datasets)
        self.rebalance_datasets = rebalance_datasets

        self.meta_df, self.records_train, self.records_test = self._get_meta(meta_path)

        if self.dataset_is_pooled:
            assert self.meta_df is not None and 'dataset_id' in self.meta_df.columns, \
                "When using multiple datasets, meta file must contain a 'dataset_id' column."
            assert not self.meta_df['dataset_id'].isnull().any(), \
                "'dataset_id' column in meta file contains missing values."
            duplicate_ids = self.meta_df.groupby('ID')['dataset_id'].nunique()
            conflicting_ids = duplicate_ids[duplicate_ids > 1].index.tolist()
            assert not conflicting_ids, f"Subject IDs found in multiple datasets: {conflicting_ids}"

            self.map_subject_to_dataset = self.meta_df.set_index('ID')['dataset_id'].to_dict()
            self.map_dataset_to_data_dir = self._infer_dataset_dir_mapping(self.meta_df, self.data_dirs)

        self.local_feat_files = self._check_and_filter_feature_files(self.local_feat_files, self.meta_df)
        n_hc_train, n_rbd_train, n_hc_test, n_rbd_test = self._get_class_counts(self.meta_df)
        if shuffle:
            np.random.shuffle(self.records_train)
            np.random.shuffle(self.records_test)

        self.train_id_map = None
        self.test_id_map = None
        self.train_group_map = None
        self.test_group_map = None

        self.num_total_rbd = n_rbd_train + n_rbd_test
        self.num_total_hc = n_hc_train + n_hc_test
        self.num_total = self.num_total_rbd + self.num_total_hc
        if verbose:
            self._log_info_(n_rbd_train, n_hc_train, n_rbd_test, n_hc_test)

    def __str__(self):
        return f"DataLoader(n_rbd={self.num_total_rbd}, n_hc={self.num_total_hc})"

    def get_train_test_data(self, agg_level: str = 'night', agg_with_numba: bool = True):

        local_feat_df = self._create_local_feature_dataframe()
        x, y, y_str, feat_map, train_index_mask, test_index_mask = None, None, None, None, None, None
        sample_metadata = None

        if agg_level == 'move':
            _feature_df = local_feat_df[self.included_local_features]
            feat_map = _feature_df.columns.values
            x = _feature_df.to_numpy()
            y = local_feat_df.ground_truth
            y_str = local_feat_df.diagnosis if self.add_str_labels else None
            train_index_mask = local_feat_df.record_key.isin(self.records_train)
            test_index_mask = local_feat_df.record_key.isin(self.records_test)

            self.train_id_map = local_feat_df.loc[train_index_mask, "id"].to_numpy()
            self.test_id_map = local_feat_df.loc[test_index_mask, "id"].to_numpy()

            if self.group_level == "patient":
                self.train_group_map = self.train_id_map
                self.test_group_map = self.test_id_map
            else:  # record
                self.train_group_map = local_feat_df.loc[train_index_mask, "record_key"].to_numpy()
                self.test_group_map = local_feat_df.loc[test_index_mask, "record_key"].to_numpy()

            if self.dataset_is_pooled and getattr(self, 'rebalance_datasets', None) \
                    and getattr(self.rebalance_datasets, 'method', None) not in (None, 'none'):
                logger.warning(
                    "[Rebalance] requested (method=%s) but only implemented for agg_level='night'. Skipping.",
                    self.rebalance_datasets.method
                )

        else:

            feat_df_night = self._create_global_feature_dataframe(local_feat_df.copy(), use_numba=agg_with_numba)
            feat_df_night = self._attach_night_metadata(feat_df_night, local_feat_df)

            if agg_level == 'night':
                _included_local_features = [
                    f"{_feat}_{postfix}" for postfix in self.aggregation for _feat in self.included_local_features]
                _feature_df = feat_df_night[_included_local_features + self.included_global_features]

                feat_map = _feature_df.columns.values
                x = _feature_df.to_numpy()
                y = feat_df_night.ground_truth
                sample_metadata = self._build_night_sample_metadata(feat_df_night)
                y_str = feat_df_night.diagnosis if self.add_str_labels else None
                train_index_mask = feat_df_night.record_key.isin(self.records_train)
                test_index_mask = feat_df_night.record_key.isin(self.records_test)

                self.train_id_map = feat_df_night.loc[train_index_mask, "id"].to_numpy()
                self.test_id_map = feat_df_night.loc[test_index_mask, "id"].to_numpy()

                if self.group_level == "patient":
                    self.train_group_map = self.train_id_map
                    self.test_group_map = self.test_id_map
                else:  # record
                    self.train_group_map = feat_df_night.loc[train_index_mask, "record_key"].to_numpy()
                    self.test_group_map = feat_df_night.loc[test_index_mask, "record_key"].to_numpy()

                # Apply dataset rebalancing (pooled only)
                if self.dataset_is_pooled and getattr(self, 'rebalance_datasets', None) \
                        and getattr(self.rebalance_datasets, 'method', None) not in (None, 'none'):
                    logger.warning(
                        f" applying composite dataset resampling with 'method='{self.rebalance_datasets.method}'")
                    train_index_mask = self._apply_dataset_rebalancing(feat_df_night, train_index_mask)
                    # refresh train_id_map after rebalancing
                    self.train_id_map = feat_df_night.loc[train_index_mask, "id"].to_numpy()
                    if self.group_level == "patient":
                        self.train_group_map = self.train_id_map
                    else:
                        self.train_group_map = feat_df_night.loc[train_index_mask, "record_key"].to_numpy()
                    self._log_rebalanced_train_summary(feat_df_night, train_index_mask)

            elif agg_level == 'patient':
                _included_local_features = [
                    f"{_feat}_{postfix}" for postfix in self.aggregation for _feat in self.included_local_features
                ]
                feat_df_patient = (
                    feat_df_night
                    .groupby(['id'])
                    [[*_included_local_features, *self.included_global_features, 'ground_truth']]
                    .agg('mean')  # numeric columns only, large entropy
                    .join(feat_df_night.groupby('id')[['record_key', 'record_id']].agg('first'))  # take first string/id
                    .reset_index()
                )
                _feature_df = feat_df_patient[_included_local_features + self.included_global_features]

                feat_map = _feature_df.columns.values
                x = _feature_df.to_numpy()
                y = feat_df_patient.ground_truth
                train_index_mask = feat_df_patient.record_key.isin(self.records_train)
                test_index_mask = feat_df_patient.record_key.isin(self.records_test)

                if self.group_level == "record":
                    logger.warning(
                        "group_level='record' is incompatible with agg_level='patient' (already grouped by id). "
                        "Falling back to patient grouping for FeatureSet.group."
                    )

                self.train_id_map = feat_df_patient.loc[train_index_mask, "id"].to_numpy()
                self.test_id_map = feat_df_patient.loc[test_index_mask, "id"].to_numpy()
                self.train_group_map = self.train_id_map
                self.test_group_map = self.test_id_map

                if self.dataset_is_pooled and getattr(self, 'rebalance_datasets', None) \
                        and getattr(self.rebalance_datasets, 'method', None) not in (None, 'none'):
                    logger.warning(
                        "[Rebalance] requested (method=%s) but only implemented for agg_level='night'. Skipping.",
                        self.rebalance_datasets.method
                    )

        x_train = np.array(x[train_index_mask])
        y_train = np.array(y[train_index_mask])
        y_str_train = np.array(y_str[train_index_mask]) if self.add_str_labels else None
        metadata_train = (sample_metadata.loc[train_index_mask].reset_index(drop=True).copy()
                          if sample_metadata is not None else None)

        logger.warning(f"n records_test = {len(self.records_test)}")
        logger.warning(f"records_test sample = {list(self.records_test[:10])}")

        logger.warning(f"n feat_df_night record_keys = {feat_df_night['record_key'].nunique()}")
        logger.warning(
            f"feat_df_night record_key sample = {feat_df_night['record_key'].drop_duplicates().tolist()[:10]}")

        logger.warning(f"n test matches = {int(test_index_mask.sum())}")

        _meta_keys = set(self.records_test.tolist())
        _feat_keys = set(feat_df_night["record_key"].astype(str).tolist())

        logger.warning(f"meta not in feat sample = {sorted(_meta_keys - _feat_keys)[:20]}")
        logger.warning(f"feat not in meta sample = {sorted(_feat_keys - _meta_keys)[:20]}")

        x_test = np.array(x[test_index_mask])
        y_test = np.array(y[test_index_mask])
        y_str_test = np.array(y_str[test_index_mask]) if self.add_str_labels else None
        metadata_test = (sample_metadata.loc[test_index_mask].reset_index(drop=True).copy()
                         if sample_metadata is not None else None)

        self._assert_binary_labels(y_train, 'y_train')
        self._assert_binary_labels(y_test, 'y_test')
        class_weights_train = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train) \
            if y_train.shape[0] > 0 else None

        if self.verbose:
            counts_train = np.unique(y_train, return_counts=True)[1] if y_train.shape[0] > 0 else [0, 0]
            counts_test = np.unique(y_test, return_counts=True)[1] if y_test.shape[0] > 0 else [0, 0]
            logger.info(f"Selected features and aggregated DataFrame shape: {x.shape}")
            if y_train.shape[0] > 0:
                train_rbd_percent = counts_train[1] / y_train.shape[0] * 100
                train_hc_percent = counts_train[0] / y_train.shape[0] * 100
            else:
                train_rbd_percent = train_hc_percent = 0
            if y_test.shape[0] > 0:
                test_rbd_percent = counts_test[1] / y_test.shape[0] * 100
                test_hc_percent = counts_test[0] / y_test.shape[0] * 100
            else:
                test_rbd_percent = test_hc_percent = 0
            logger.info(f"Total {agg_level} instances in train set: {y_train.shape[0]}\n"
                        f"\t - RBD: {counts_train[1]:6.0f} = {train_rbd_percent:4.1f}% "
                        f"\t - HC:  {counts_train[0]:6.0f} = {train_hc_percent:4.1f}%")

            logger.info(f"Total {agg_level} instances in test set: {y_test.shape[0]}\n"
                        f"\t - RBD: {counts_test[1]:6.0f} = {test_rbd_percent:4.1f}% "
                        f"\t - HC:  {counts_test[0]:6.0f} = {test_hc_percent:4.1f}%")

        # check for NaNs
        _datasets = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}
        nan_summary = []
        for name, data in _datasets.items():
            nan_count = np.isnan(data).sum()
            nan_summary.append(f'{name}: {nan_count} NaNs')
        if any(np.isnan(data).sum() > 0 for data in _datasets.values()):
            logging.warning('found NaNs in data: - ' + ', '.join(nan_summary))
        else:
            if self.verbose:
                logging.info('no NaNs in x_train, y_train, x_test, y_test.\n')
        train_dataset_ids = np.array([self.map_subject_to_dataset[_id] for _id in self.train_id_map]) \
            if self.dataset_is_pooled else None
        test_dataset_ids = np.array([self.map_subject_to_dataset[_id] for _id in self.test_id_map]) \
            if self.dataset_is_pooled else None

        train_set = FeatureSet(
            x=x_train, y=y_train, y_str=y_str_train,  # data and labels
            group=self.train_group_map, feat_map=feat_map, dataset=train_dataset_ids,  # mappings
            metadata=metadata_train
        )
        test_set = FeatureSet(
            x=x_test, y=y_test, y_str=y_str_test,
            group=self.test_group_map, feat_map=feat_map, dataset=test_dataset_ids,
            metadata=metadata_test
        )

        return train_set, test_set, class_weights_train

    @staticmethod
    def _check_and_filter_feature_files(local_feat_files: List[Path], meta_df: pd.DataFrame) -> List[Path]:
        """ 1) Warn if any expected feature files (according to meta_df) are missing.
            2) Warn if there are any extra feature files not in the filtered meta_df.
            3) Return only the files matching the meta_df entries."""
        # build expected suffixes
        expected = set()
        for _, row in meta_df.iterrows():
            _id = row['ID']
            rid = str(row.get('record_ID')).strip()
            suf = _id if pd.isna(row.get('record_ID')) or rid.lower() in {'none', ''} else f"{_id}_{rid}"
            expected.add(suf)

        # map each actual file to its suffix
        suffix_to_paths = defaultdict(list)
        for p in local_feat_files:
            suf = p.stem.replace("local-features-", "")
            suffix_to_paths[suf].append(p)
        actual = set(suffix_to_paths.keys())

        missing, extra = expected - actual, actual - expected
        if missing:
            logger.warning(f"Missing {len(missing)} expected feature file(s): {sorted(missing)}")
        if extra:
            logger.debug(
                f"Found {len(extra)} unexpected feature file(s) (excluded or not in meta e.g. because of unmapped "
                f"diagnosis): {sorted(extra)}")

        # filter the original list, keeping only paths whose suffix is expected
        filtered = [p for suf, paths in suffix_to_paths.items() if suf in expected for p in paths]
        return sorted(filtered)

    @staticmethod
    def _assert_binary_labels(y_array, name='y_array'):
        if y_array.size > 0 and not np.array_equal(np.unique(y_array), [0, 1]):
            raise AssertionError(f"{name} contains non-binary labels: {np.unique(y_array)}")

    @staticmethod
    def _meta_make_record_keys(meta_df: pd.DataFrame) -> pd.Series:
        """Construct record_key from meta_df rows.
        Rules (backwards compatible):
        - If record_ID is missing, empty, 'none', or 'nan' → use 'ID'
        - Otherwise → use 'ID_recordID'"""

        if "record_ID" not in meta_df.columns:
            return meta_df["ID"].astype(str)

        def _make_key(row):
            rid = DataLoader._normalize_record_id(row.get("record_ID"))
            sid = str(row["ID"]).strip()
            return sid if rid is None else f"{sid}_{rid}"

        return meta_df.apply(_make_key, axis=1)

    def _get_meta(self, meta_csv_path: Path):

        # read meta file
        meta_df = utils.read_meta_csv_to_df(meta_csv_path, exclude=True, verbose=self.verbose)

        assert 'diagnosis' in meta_df.columns, f"'meta_df' at {meta_csv_path} must contain a 'diagnosis' column."
        _is_null = meta_df.diagnosis.isnull()
        assert not meta_df.diagnosis.isnull().any(), "'diagnosis' column contains NaN values."

        meta_df = meta_df.copy()
        meta_df['diagnosis'], used_mapping = self._map_diagnosis_series_if_needed(
            meta_df['diagnosis'], context=f"meta_df ({meta_csv_path})")

        mapping_keys, data_keys = set(self.binary_mapping.keys()), set(meta_df['diagnosis'].unique())
        unused_keys = mapping_keys - data_keys
        if unused_keys:  # keys defined in mapping but not present in the data
            logger.warning(f"Mapping keys not present in this dataset: {unused_keys}")
        unmapped_keys = data_keys - mapping_keys
        if unmapped_keys:  # Keys present in the data but not covered by the mapping
            logger.warning(f"Dataset contains diagnoses not covered by mapping: {unmapped_keys}")
        # choose only relevant rows according to binary_mapping, will drop all other rows
        orig_counts = meta_df['diagnosis'].value_counts().to_dict()
        if self.verbose:
            logger.info(f"Original diagnoses (pre‐filter): {orig_counts}")
        keep_mask = meta_df['diagnosis'].isin(self.binary_mapping.keys())
        dropped_diags = meta_df.loc[~keep_mask, 'diagnosis'].value_counts().to_dict()
        if self.verbose:
            logger.info(f"Dropping {len(meta_df) - keep_mask.sum()} rows with unmapped diagnoses: {dropped_diags}")

        meta_df = meta_df[keep_mask]
        mapped_counts = meta_df['diagnosis'].value_counts().to_dict()
        if self.verbose:
            logger.info(f"Kept {len(meta_df)} rows; mapped diagnoses: {mapped_counts}")
        # create binary labels
        meta_df.insert(
            meta_df.columns.get_loc('diagnosis') + 1, 'binary_label', meta_df.diagnosis.map(self.binary_mapping))

        if meta_df["binary_label"].isna().any():
            bad = sorted(meta_df.loc[meta_df["binary_label"].isna(), "diagnosis"].unique().tolist())
            raise ValueError(
                f"meta_df contains diagnoses not covered by binary_mapping after mapping: {bad}. "
                f"binary_mapping keys={sorted(self.binary_mapping.keys())}")

        # get the ids of the training and testing records
        records_train = self._meta_make_record_keys(
            meta_df.loc[meta_df['train/test'] == 'train']).astype(str).to_numpy()
        records_test = self._meta_make_record_keys(meta_df.loc[meta_df['train/test'] == 'test']).astype(str).to_numpy()

        return meta_df, records_train, records_test

    @staticmethod
    def _get_class_counts(meta_df: pd.DataFrame):
        _unique_labels = meta_df.binary_label.unique()
        assert len(_unique_labels) == 2, f"only two unique labels allowed, got {_unique_labels}."
        if sorted(_unique_labels) == ['HC', 'RBD']:
            _hc_key, _rbd_key = 'HC', 'RBD'
        elif sorted(_unique_labels) == [0, 1]:
            _hc_key, _rbd_key = 0, 1
        else:
            raise ValueError(f"binary labels are not of expected format 'HC'/'RBD' or 0/1, got {_unique_labels}.")

        train_df = meta_df[meta_df['train/test'] == 'train']
        if not train_df.empty:
            _value_count_train = train_df.binary_label.value_counts()
            n_hc_train, n_rbd_train = _value_count_train.loc[_hc_key], _value_count_train.loc[_rbd_key]
        else:
            n_hc_train, n_rbd_train = 0, 0

        test_df = meta_df[meta_df['train/test'] == 'test']
        if not test_df.empty:
            _value_count_test = test_df.binary_label.value_counts()
            n_hc_test, n_rbd_test = _value_count_test.loc[_hc_key], _value_count_test.loc[_rbd_key]
        else:
            n_hc_test, n_rbd_test = 0, 0

        return n_hc_train, n_rbd_train, n_hc_test, n_rbd_test

    @staticmethod
    def _infer_dataset_dir_mapping(meta_df: pd.DataFrame, patient_root_dirs: List[Path]) -> dict:
        """Get mapping from dataset_id to patient_dir using file existence checks.
         Allows multiple dataset_ids to share the same directory."""
        dir_to_dataset_ids = defaultdict(set)
        for p_dir in patient_root_dirs:
            for _, row in meta_df.iterrows():
                full_path = p_dir / row['ID']
                if full_path.exists():
                    dir_to_dataset_ids[p_dir].add(row['dataset_id'])

        patient_root_dir_to_dataset_id = {}
        for p_dir, ds_ids in dir_to_dataset_ids.items():
            if len(ds_ids) != 1:
                logger.warning(f"Directory {p_dir} maps to multiple dataset_ids: {ds_ids}")
            for ds_id in ds_ids:
                patient_root_dir_to_dataset_id[ds_id] = p_dir

        mapped_dataset_ids = set(patient_root_dir_to_dataset_id.keys())
        all_dataset_ids = set(meta_df['dataset_id'].unique())

        if mapped_dataset_ids != all_dataset_ids:
            missing = all_dataset_ids - mapped_dataset_ids
            raise ValueError(f"No matching patient_root_dir found for dataset_ids: {missing}")

        return patient_root_dir_to_dataset_id

    def _create_local_feature_dataframe(self, util_cols: List[str] = None):
        if self.verbose:
            logger.info("creating local feature DataFrame... ")

        # required for downstream logic
        required_util_cols = ["id", "diagnosis", "time_start", "time_end", "time_diff", "night"]
        # optional / backwards-compatible
        default_optional_util_cols = ["record_id", "sptw_start", "sptw_end", "sptw_idx", "runtime", "ident"]
        optional_util_cols = util_cols if util_cols is not None else default_optional_util_cols

        with utils.Timer() as timer:
            _li = []

            for filename in self.local_feat_files:
                _df = pd.read_csv(filename, index_col=None, header=0)
                _df = _df.loc[:, ~_df.columns.str.contains("Unnamed")]  # drop old indices

                if _df.empty:
                    continue

                # always add ident for debugging / traceability
                _df["ident"] = filename

                # ensure record_id exists (needed to build record_key)
                if "record_id" not in _df.columns:
                    _df["record_id"] = None

                # enforce required util columns
                missing_required = [c for c in required_util_cols if c not in _df.columns]
                if missing_required:
                    raise UserWarning(f"{filename.name}: missing required column(s): {missing_required}")

                # enforce that requested local features exist
                missing_feats = [c for c in self.included_local_features if c not in _df.columns]
                if missing_feats:
                    raise UserWarning(f"{filename.name}: missing local feature column(s): {missing_feats[:10]}"
                                      f"{'...' if len(missing_feats) > 10 else ''}")

                # keep only optional util cols that actually exist (plus always keep record_id + ident)
                optional_present = [c for c in optional_util_cols if c in _df.columns]
                keep_cols = (
                        list(self.included_local_features)
                        + required_util_cols
                        + ["record_id", "ident"]
                        + [c for c in optional_present if c not in {"record_id", "ident"}]
                )

                # select columns (robust against missing optional columns)
                _df = _df[keep_cols]

                _df = _df.copy()
                _df["diagnosis"], used_mapping = self._map_diagnosis_series_if_needed(
                    _df["diagnosis"], context=f"local feature file '{filename.name}'")

                # normalize record_id and build record_key
                _df["record_id"] = _df["record_id"].astype(str).str.strip()
                _df["record_id"] = _df["record_id"].replace(["none", "NaN", "nan", "None", ""], None)
                _df["record_key"] = _df.apply(
                    lambda row: row["id"] if row["record_id"] is None else f"{row['id']}_{row['record_id']}",
                    axis=1
                )

                _li.append(_df)

            if len(_li) == 0:
                raise UserWarning("No non-empty local feature files found after loading.")

            _local_feat_df = pd.concat(_li, axis=0, ignore_index=True)

            # derived time columns
            _local_feat_df["month"] = pd.to_datetime(_local_feat_df.time_start).dt.month
            _local_feat_df["day"] = pd.to_datetime(_local_feat_df.time_start).dt.day
            _local_feat_df["hour"] = pd.to_datetime(_local_feat_df.time_start).dt.hour
            _local_feat_df["minute"] = pd.to_datetime(_local_feat_df.time_start).dt.minute

            # 'toilette' filter
            _local_feat_df = _local_feat_df[(_local_feat_df.time_diff < 50) & (_local_feat_df.time_diff > 0.5)]

            # labels
            _local_feat_df["ground_truth"] = _local_feat_df.diagnosis.map(self.binary_mapping).values

            if _local_feat_df["ground_truth"].isna().any():
                bad = sorted(_local_feat_df.loc[_local_feat_df["ground_truth"].isna(), "diagnosis"].unique().tolist())
                raise ValueError(
                    f"Local feature dataframe contains diagnoses not covered by binary_mapping after mapping: {bad}. "
                    f"binary_mapping keys={sorted(self.binary_mapping.keys())}")

            # handle problematic values in feature columns only
            excluded_cols = required_util_cols + ["record_id", "ident", "record_key", "diagnosis", "ground_truth",
                                                  "month", "day", "hour", "minute"]
            # include optional cols too (if present) so they won't be treated as features
            excluded_cols += [c for c in optional_util_cols if c in _local_feat_df.columns]

            _local_feat_df = utils.handle_problematic_values_in_feature_df(
                _local_feat_df,
                drop=True,
                replace=None,
                df_log_name="local_feat_df",
                excluded_cols=excluded_cols,
                verbose=self.verbose,
            )

        if self.verbose:
            logger.info(f"creating local feature DataFrame... done. ({timer()}s)")
        return _local_feat_df

    def _log_info_(self, n_rbd_train, n_hc_train, n_rbd_test, n_hc_test):

        # --- meta-level counts (after diagnosis filtering / exclusions) ---
        n_records = int(self.meta_df.shape[0])

        # record_key is what the file layout and train/test split are based on
        _rk = self.meta_df.apply(
            lambda row: row['ID'] if pd.isna(row.get('record_ID')) else f"{row['ID']}_{str(row['record_ID']).strip()}",
            axis=1
        )
        n_unique_records = int(_rk.nunique())

        n_subjects = int(self.meta_df['ID'].nunique())

        # subject-level labels (assert consistency across records for the same subject)
        per_subject = self.meta_df.groupby('ID')['binary_label'].nunique()
        if (per_subject > 1).any():
            inconsistent = per_subject[per_subject > 1].index.tolist()
            logger.warning(
                "Found %d subject(s) with inconsistent binary_label across records (showing up to 10): %s",
                len(inconsistent), inconsistent[:10]
            )
        subj_labels = self.meta_df.groupby('ID')['binary_label'].first()
        n_rbd_subjects = int((subj_labels == 1).sum())
        n_hc_subjects = int((subj_labels == 0).sum())

        # record-level labels (rows)
        n_rbd_records = int((self.meta_df['binary_label'] == 1).sum())
        n_hc_records = int((self.meta_df['binary_label'] == 0).sum())

        # group-level (how FeatureSet.group will be constructed for night/move level)
        if self.group_level == 'patient':
            n_groups = n_subjects
            group_desc = "id"
        else:
            n_groups = n_unique_records
            group_desc = "record_key"

        logger.info(
            f"full dataset (meta): subjects = {n_subjects}, records = {n_unique_records} "
            f"(rows = {n_records})"
        )
        logger.info(
            f"label distribution (meta): subjects -> rbd = {n_rbd_subjects} ({(n_rbd_subjects / n_subjects * 100) if n_subjects else 0:4.1f}%), "
            f"hc = {n_hc_subjects} ({(n_hc_subjects / n_subjects * 100) if n_subjects else 0:4.1f}%) | "
            f"records -> rbd = {n_rbd_records} ({(n_rbd_records / n_unique_records * 100) if n_unique_records else 0:4.1f}%), "
            f"hc = {n_hc_records} ({(n_hc_records / n_unique_records * 100) if n_unique_records else 0:4.1f}%)"
        )
        logger.info(f"group_level = {self.group_level} (FeatureSet.group uses {group_desc}); n_groups = {n_groups}")

        # --- train/test split summaries (meta-based) ---
        train_meta = self.meta_df[self.meta_df['train/test'] == 'train']
        test_meta = self.meta_df[self.meta_df['train/test'] == 'test']

        train_subjects = int(train_meta['ID'].nunique()) if not train_meta.empty else 0
        test_subjects = int(test_meta['ID'].nunique()) if not test_meta.empty else 0

        train_records = int(train_meta.apply(
            lambda row: row['ID'] if pd.isna(row.get('record_ID')) else f"{row['ID']}_{str(row['record_ID']).strip()}",
            axis=1
        ).nunique()) if not train_meta.empty else 0

        test_records = int(test_meta.apply(
            lambda row: row['ID'] if pd.isna(row.get('record_ID')) else f"{row['ID']}_{str(row['record_ID']).strip()}",
            axis=1
        ).nunique()) if not test_meta.empty else 0

        if self.group_level == 'patient':
            train_groups = train_subjects
            test_groups = test_subjects
        else:
            train_groups = train_records
            test_groups = test_records

        logger.info(
            f"split summary (meta): train subjects = {train_subjects}, train records = {train_records}, train groups = {train_groups} | "
            f"test subjects = {test_subjects}, test records = {test_records}, test groups = {test_groups}"
        )

        # keep the existing record-wise label logging (backwards compatible with old expectations)
        logger.info(
            f"train/test split (records):\n"
            f"\t\t - train: n = {n_rbd_train + n_hc_train:2.0f},"
            f"rbd = {n_rbd_train:2.0f} "
            f"({n_rbd_train / (n_rbd_train + n_hc_train) * 100 if n_rbd_train + n_hc_train != 0 else 0:4.1f}%)"
            f" hc = {n_hc_train:2.0f} "
            f"({n_hc_train / (n_rbd_train + n_hc_train) * 100 if n_rbd_train + n_hc_train != 0 else 0:4.1f}%)\n"
            f"\t\t - test:  n = {n_rbd_test + n_hc_test:2.0f},"
            f"rbd = {n_rbd_test:2.0f} "
            f"({n_rbd_test / (n_rbd_test + n_hc_test) * 100 if n_rbd_test + n_hc_test != 0 else 0:4.1f}%)"
            f" hc = {n_hc_test:2.0f} "
            f"({n_hc_test / (n_rbd_test + n_hc_test) * 100 if n_rbd_test + n_hc_test != 0 else 0:4.1f}%)"
        )

        if self.dataset_is_pooled:
            logger.info("Dataset-wise contributions (records):")
            for dataset_id in sorted(self.meta_df['dataset_id'].unique()):
                df_subset = self.meta_df[self.meta_df['dataset_id'] == dataset_id]
                for split in ['train', 'test']:
                    df_split = df_subset[df_subset['train/test'] == split]
                    count_total = df_split.shape[0]
                    count_rbd = df_split[df_split['binary_label'] == 1].shape[0]
                    count_hc = df_split[df_split['binary_label'] == 0].shape[0]
                    logger.info(
                        f"\t - dataset {dataset_id} ({split}): total = {count_total:3d}, "
                        f"rbd = {count_rbd:3d}, hc = {count_hc:3d}, "
                        f"rbd = {count_rbd / count_total * 100 if count_total > 0 else 0:4.1f}%, "
                        f"hc = {count_hc / count_total * 100 if count_total > 0 else 0:4.1f}%"
                    )

    def _aggregate_local_to_global(self, _local_df_copy: pd.DataFrame, use_numba: bool):

        global_feat_df = utils.aggregate_local_feat_df_to_global(
            _local_df_copy, self.included_local_features,
            use_numba=use_numba, aggregation_methods=self.aggregation, verbose=self.verbose)

        return utils.handle_problematic_values_in_feature_df(
            global_feat_df, drop=True, replace=None, df_log_name='global_feat_df',
            excluded_cols=['record_key', 'id', 'record_id', 'diagnosis', 'ground_truth', 'block_id'],
            verbose=self.verbose)

    @staticmethod
    def _attach_night_metadata(feat_df_night: pd.DataFrame, local_feat_df: pd.DataFrame) -> pd.DataFrame:
        """Attach one validated metadata row to each aggregated record/night sample.

        Metadata are reconstructed from the local feature rows using the same
        ``record_key``/``night`` keys used for feature aggregation. Any conflicting
        metadata values within one aggregated night raise immediately rather than being
        resolved silently.
        """
        key_columns = ['record_key', 'night']
        candidate_columns = [
            'id', 'record_id', 'sptw_idx', 'sptw_start', 'sptw_end', 'ident'
        ]
        metadata_columns = [column for column in candidate_columns if column in local_feat_df.columns]
        if not metadata_columns:
            return feat_df_night

        grouped = local_feat_df.groupby(key_columns, sort=False, dropna=False)
        conflicts = grouped[metadata_columns].nunique(dropna=False).gt(1)
        if conflicts.any().any():
            bad_groups = conflicts.any(axis=1)
            examples = [tuple(index) for index in conflicts.index[bad_groups][:10]]
            bad_columns = conflicts.loc[bad_groups].any(axis=0)
            raise ValueError(
                'Conflicting night metadata found within aggregated record/night groups. '
                f'columns={bad_columns[bad_columns].index.tolist()}, examples={examples}'
            )

        night_metadata = grouped[metadata_columns].first().reset_index()
        add_columns = [column for column in metadata_columns if column not in feat_df_night.columns]
        if add_columns:
            feat_df_night = feat_df_night.merge(
                night_metadata[key_columns + add_columns],
                on=key_columns,
                how='left',
                validate='one_to_one',
            )

        return feat_df_night

    @staticmethod
    def _build_night_sample_metadata(feat_df_night: pd.DataFrame) -> pd.DataFrame:
        """Create row-aligned metadata for night-level FeatureSet samples."""
        metadata = pd.DataFrame(index=feat_df_night.index)
        metadata['subject_id'] = feat_df_night['id'].astype(str)
        metadata['record_id'] = feat_df_night['record_id'] if 'record_id' in feat_df_night else None
        metadata['record_key'] = feat_df_night['record_key'].astype(str)
        metadata['night'] = feat_df_night['night']

        for column in ('sptw_idx', 'sptw_start', 'sptw_end'):
            metadata[column] = feat_df_night[column] if column in feat_df_night else pd.NA

        for column in ('sptw_start', 'sptw_end'):
            metadata[column] = pd.to_datetime(metadata[column], errors='coerce')

        metadata['sptw_duration_hours'] = (
            metadata['sptw_end'] - metadata['sptw_start']
        ).dt.total_seconds() / 3600.0
        metadata['source_feature_file'] = (
            feat_df_night['ident'].astype(str) if 'ident' in feat_df_night else pd.NA
        )
        metadata['night_id'] = (
            metadata['record_key'].astype(str)
            + '|night='
            + metadata['night'].astype(str)
        )
        metadata['is_synthetic'] = False

        duplicates = metadata['night_id'].duplicated(keep=False)
        if duplicates.any():
            examples = metadata.loc[duplicates, ['night_id', 'sptw_start', 'sptw_end']].head(10)
            raise ValueError(f'Duplicate night_id values found:\n{examples.to_string(index=False)}')

        return metadata.reset_index(drop=True)

    def _create_global_feature_dataframe(self, local_df_copy: pd.DataFrame, use_numba: bool = True,
                                         return_structure_only: bool = False, global_only: bool = False):

        if return_structure_only:
            return local_df_copy[
                ['id', 'record_id', 'record_key', 'night', 'ground_truth']].drop_duplicates().reset_index(drop=True)

        if global_only:  # only get skeleton
            global_feat_df = local_df_copy[
                ['id', 'record_id', 'record_key', 'night', 'ground_truth', 'diagnosis']
            ].drop_duplicates().reset_index(drop=True)
        else:  # aggregate local features
            global_feat_df = self._aggregate_local_to_global(local_df_copy, use_numba=use_numba)

        if not self.included_global_features:
            return global_feat_df  # if no global features requested, return early with only aggregated local feats

        for _record_key, _id, _record_id in zip(
                global_feat_df.record_key.unique(), global_feat_df.groupby('record_key')['id'].first(),
                global_feat_df.groupby('record_key')['record_id'].first()):

            # glob the global feature files for each record
            if self.dataset_is_pooled:
                _dataset_id = self.map_subject_to_dataset[_id]
                patient_dir = self.map_dataset_to_data_dir[_dataset_id]
            else:
                patient_dir = self.data_dirs[0]

            _record_id_str = None if pd.isna(_record_id) else str(_record_id).strip()
            if _record_id_str in [None, '', 'none', 'None']:
                candidate_dirs = [
                    patient_dir / f'{_id}' / self.feature_dir,
                ]
            else:
                candidate_dirs = [
                    patient_dir / f'{_id}' / f'{_id}_{_record_id_str}' / self.feature_dir,  # legacy layout
                    patient_dir / f'{_id}' / _record_id_str / self.feature_dir,  # IRBDSG/new layout
                ]

            _global_feat_paths = []
            for _feature_dir_path in candidate_dirs:
                if _feature_dir_path.exists():
                    _global_feat_paths.extend(sorted(_feature_dir_path.glob('*global*.csv')))

            if len(_global_feat_paths) != 1:
                raise UserWarning(
                    f'none or too many global feature files found for {_record_key}. '
                    f'candidate_dirs={candidate_dirs}, matches={_global_feat_paths}'
                )

            else:
                _global_features = pd.read_csv(_global_feat_paths[0])
                _global_features = _global_features[self.included_global_features + ['id', 'night']]

                global_feat_df = pd.merge(
                    global_feat_df, _global_features, on=['id', 'night'], how='left', suffixes=('', '_dup'))

                for col in _global_features.columns.difference(['id', 'night']):
                    dup_col = f'{col}_dup'
                    if dup_col in global_feat_df.columns:
                        global_feat_df[col] = global_feat_df[col].fillna(global_feat_df.pop(dup_col))
        return global_feat_df

    def _apply_dataset_rebalancing(
            self, df_night: pd.DataFrame,  # aggregated night-level DF (has 'record_key','id','ground_truth',...)
            train_index_mask: np.ndarray,  # boolean mask over df_night rows
    ) -> np.ndarray:
        """Rebalance the *training* portion by downsampling records per dataset.
        Operates at record level and then keeps all nights from the chosen records.
        Supported methods (self.rebalance_datasets.method):
          - 'none' / None: do nothing
          - 'min': equalize all datasets to the size of the smallest (strict downsample)
          - 'median': cap each dataset at the median dataset size
          - 'cap_absolute': cap each dataset at a fixed maximum (uses .max_per_dataset)
          - 'cap_to_second_largest': cap the largest dataset to the size of the
             second largest if it exceeds dominance_ratio × second_largest
             (uses .dominance_ratio; default 1.4–1.5 is sensible)
        If .preserve_class_ratio is True, per-dataset positive/negative sampling
        follows the dataset’s original train-split class proportions."""
        cfg = getattr(self, "rebalance_datasets", None)
        if not cfg or not getattr(cfg, "method", None) or cfg.method in ("none", None):
            return train_index_mask

        assert self.dataset_is_pooled, \
            "Rebalancing by dataset requires pooled mode with 'dataset_id' available."

        # ---------------------------------------------
        # Build a RECORD-LEVEL view of the *train* part
        # ---------------------------------------------
        train_df = df_night.loc[train_index_mask, ["record_key", "id", "ground_truth"]].copy()

        # one row per record_key (record = subject or subject_record)
        rec_df = (
            train_df.groupby("record_key")
            .agg(id=("id", "first"),
                 y=("ground_truth", "first"))
            .reset_index()
        )

        # map to dataset_id
        rec_df["dataset_id"] = rec_df["id"].map(self.map_subject_to_dataset)
        if rec_df["dataset_id"].isna().any():
            missing = rec_df.loc[rec_df["dataset_id"].isna(), "id"].unique().tolist()
            raise ValueError(f"Missing dataset_id mapping for IDs: {missing[:5]}{'...' if len(missing) > 5 else ''}")

        rng = np.random.default_rng(getattr(cfg, "seed", 42))

        # ---------------------------------------------
        # Determine per-dataset target sizes
        # ---------------------------------------------
        sizes = rec_df.groupby("dataset_id").size().sort_values()
        ds_order = list(sizes.index)
        ds_sizes = sizes.to_dict()

        method = cfg.method
        target_sizes = {}

        if method == "min":
            target = int(sizes.min())
            for ds in ds_order:
                target_sizes[ds] = min(ds_sizes[ds], target)

        elif method == "median":
            target = int(np.median(sizes.values))
            for ds in ds_order:
                target_sizes[ds] = min(ds_sizes[ds], target)

        elif method == "cap_absolute":
            max_cap = int(getattr(cfg, "max_per_dataset", 0) or 0)
            if max_cap <= 0:
                raise ValueError("cap_absolute requires a positive 'max_per_dataset'.")
            for ds in ds_order:
                target_sizes[ds] = min(ds_sizes[ds], max_cap)

        elif method == "cap_to_second_largest":
            # cap ONLY the largest dataset if it's too dominant vs the 2nd largest
            if len(sizes) < 2:
                # nothing to do with a single dataset
                return train_index_mask
            second = int(sizes.iloc[-2])
            largest_ds = sizes.index[-1]
            largest_n = int(sizes.iloc[-1])
            ratio = float(getattr(cfg, "dominance_ratio", 1.5))

            for ds in ds_order:
                if ds == largest_ds and largest_n > ratio * second:
                    target_sizes[ds] = second
                else:
                    target_sizes[ds] = ds_sizes[ds]

        else:
            raise ValueError(f"Unknown rebalancing method: {method}")

        # ---------------------------------------------
        # Sample record_keys per dataset (optionally preserving class ratios)
        # ---------------------------------------------
        keep_record_keys = []

        preserve_ratio = bool(getattr(cfg, "preserve_class_ratio", True))
        for ds, target_n in target_sizes.items():
            sub = rec_df[rec_df["dataset_id"] == ds]
            if target_n >= len(sub):
                keep_record_keys.extend(sub["record_key"].tolist())
                continue

            if not preserve_ratio:
                sel = sub.sample(n=target_n, random_state=int(getattr(cfg, "seed", 42)))
                keep_record_keys.extend(sel["record_key"].tolist())
            else:
                # preserve within-dataset class mix
                pos = sub[sub["y"] == 1]
                neg = sub[sub["y"] == 0]
                n_pos = len(pos)
                n_neg = len(neg)
                if n_pos + n_neg == 0:
                    continue

                frac_pos = n_pos / (n_pos + n_neg)
                n_pos_target = int(round(frac_pos * target_n))
                n_neg_target = target_n - n_pos_target

                pos_keep = pos.sample(
                    n=min(n_pos_target, n_pos),
                    random_state=int(getattr(cfg, "seed", 42))
                )
                neg_keep = neg.sample(
                    n=min(n_neg_target, n_neg),
                    random_state=int(getattr(cfg, "seed", 42) + 1)
                )

                # If rounding / scarcity left us short, top up from the larger class
                short = target_n - (len(pos_keep) + len(neg_keep))
                if short > 0:
                    remainder = sub.drop(pos_keep.index.union(neg_keep.index), errors="ignore")
                    if len(remainder) > 0:
                        extra = remainder.sample(
                            n=min(short, len(remainder)),
                            random_state=int(getattr(cfg, "seed", 42) + 2)
                        )
                        take = pd.concat([pos_keep, neg_keep, extra], axis=0)
                    else:
                        take = pd.concat([pos_keep, neg_keep], axis=0)
                else:
                    take = pd.concat([pos_keep, neg_keep], axis=0)

                keep_record_keys.extend(take["record_key"].tolist())

        keep_record_keys = set(keep_record_keys)

        # ----------------------------------------------------
        # Convert the chosen records back to a row-wise mask
        # ----------------------------------------------------
        new_train_mask = train_index_mask.copy()
        train_rows = np.where(train_index_mask)[0]
        train_rec_keys = df_night.loc[train_index_mask, "record_key"].values
        keep_mask_local = np.array([rk in keep_record_keys for rk in train_rec_keys], dtype=bool)
        new_train_mask[train_rows] = keep_mask_local

        # Logging summary
        before = {ds: int(n) for ds, n in ds_sizes.items()}
        after = {
            ds: int((rec_df["dataset_id"].isin([ds]) & rec_df["record_key"].isin(keep_record_keys)).sum())
            for ds in ds_order
        }
        logger.info(f"[Rebalance] method={method}  preserve_class_ratio={preserve_ratio}")
        logger.info(f"[Rebalance] per-dataset counts (records): before={before}  →  after={after}")

        return new_train_mask

    def _log_rebalanced_train_summary(self, df_night: pd.DataFrame, train_index_mask: np.ndarray):
        """Log an info summary comparable to _log_info_ but for the TRAIN split after rebalancing.
           Works at record level (one row per record_key)."""
        # One row per record (record_key), with id, y, dataset_id
        rec_df = (
            df_night.loc[train_index_mask, ['record_key', 'id', 'ground_truth']]
            .drop_duplicates()
            .assign(
                dataset_id=lambda d: d['id'].map(self.map_subject_to_dataset) if self.dataset_is_pooled else 'SINGLE')
        )

        # Subject-level view (may collapse multiple records to one subject)
        subj_df = (
            df_night.loc[train_index_mask, ['id', 'ground_truth']]
            .drop_duplicates()
            .assign(
                dataset_id=lambda d: d['id'].map(self.map_subject_to_dataset) if self.dataset_is_pooled else 'SINGLE')
        )

        n_records = len(rec_df)
        n_subjects = len(subj_df)
        if self.group_level == 'patient':
            n_groups = n_subjects
            group_desc = "id"
        else:
            n_groups = n_records
            group_desc = "record_key"

        # Totals (record-level + subject-level)
        n_rbd_records = int((rec_df['ground_truth'] == 1).sum())
        n_hc_records = n_records - n_rbd_records
        p_rbd_records = (n_rbd_records / n_records * 100) if n_records else 0.0
        p_hc_records = (n_hc_records / n_records * 100) if n_records else 0.0

        n_rbd_subjects = int((subj_df['ground_truth'] == 1).sum())
        n_hc_subjects = n_subjects - n_rbd_subjects
        p_rbd_subjects = (n_rbd_subjects / n_subjects * 100) if n_subjects else 0.0
        p_hc_subjects = (n_hc_subjects / n_subjects * 100) if n_subjects else 0.0

        logger.info("[Rebalance][post] train split: subjects = %d, records = %d", n_subjects, n_records)
        logger.info(
            "[Rebalance][post] group_level = %s (FeatureSet.group uses %s); n_groups = %d",
            self.group_level, group_desc, n_groups
        )
        logger.info("[Rebalance][post] label distribution (subjects): rbd = %d (%.1f%%), hc = %d (%.1f%%)",
                    n_rbd_subjects, p_rbd_subjects, n_hc_subjects, p_hc_subjects)
        logger.info("[Rebalance][post] label distribution (records):  rbd = %d (%.1f%%), hc = %d (%.1f%%)",
                    n_rbd_records, p_rbd_records, n_hc_records, p_hc_records)

        if self.dataset_is_pooled:
            logger.info("[Rebalance][post] Dataset-wise contributions (TRAIN) (records):")
            for ds in sorted(rec_df['dataset_id'].unique()):
                sub = rec_df[rec_df['dataset_id'] == ds]
                t = len(sub)
                r = int((sub['ground_truth'] == 1).sum())
                h = t - r
                pr = (r / t * 100) if t else 0.0
                ph = (h / t * 100) if t else 0.0
                logger.info("\t - dataset %s (train): total = %3d, rbd = %3d, hc = %3d, rbd = %4.1f%%, hc = %4.1f%%",
                            ds, t, r, h, pr, ph)

    def _get_expected_str_labels(self) -> set[str]:
        """
        Canonical harmonized diagnosis labels.

        If a str_label_mapping is configured, use its label space
        (e.g. {'HC', 'iRBD', 'PD-RBD', 'PD+RBD'}), not the current
        binary task subset.
        """
        if self.str_label_mapping is not None:
            mappings = get_label_mappings()
            if self.str_label_mapping not in mappings.mappings:
                raise ValueError(
                    f"Unknown str_label_mapping='{self.str_label_mapping}'. "
                    f"Available mappings: {sorted(mappings.mappings.keys())}"
                )
            return {str(label).strip() for label in mappings.mappings[self.str_label_mapping].keys()}

        return {str(k).strip() for k in self.binary_mapping.keys()}

    @staticmethod
    def _normalize_diag_series(series: pd.Series) -> pd.Series:
        return series.astype('string').str.strip()

    def _labels_already_mapped(self, labels: set[str]) -> bool:
        expected = {str(x).strip().lower() for x in self._get_expected_str_labels()}
        observed = {str(x).strip().lower() for x in labels if str(x).strip() != ""}
        return len(observed) > 0 and observed.issubset(expected)

    @staticmethod
    def _detect_suitable_label_mappings(labels: set[str]) -> list[str]:
        """
        Return mapping names whose code space fully covers the observed labels.
        Only used when labels are not already mapped.
        """
        mappings = get_label_mappings()
        labels_norm = {str(x).strip().lower() for x in labels if str(x).strip() != ""}

        matches = []
        for mapping_name, label_map in mappings.mappings.items():
            covered_codes = {
                str(code).strip().lower()
                for codes in label_map.values()
                for code in codes
            }
            if labels_norm.issubset(covered_codes):
                matches.append(mapping_name)
        return sorted(matches)

    def _resolve_str_label_mapping(self, labels: set[str]) -> Optional[str]:
        """
        Decide whether labels need remapping and which mapping to use.

        Rules:
        - if labels are already mapped -> return None
        - if config explicitly defines str_label_mapping -> validate and return it
        - else autodetect:
            * 0 matches -> raise
            * 1 match -> return it
            * >1 matches -> raise and ask user to define one explicitly
        """
        if self._labels_already_mapped(labels):
            return None

        labels_norm = {str(x).strip().lower() for x in labels if str(x).strip() != ""}

        if self.str_label_mapping is not None:
            code_to_label = _build_code_to_label_map(self.str_label_mapping)
            covered = set(code_to_label.keys())
            missing = sorted(labels_norm - covered)
            if missing:
                raise ValueError(
                    f"Configured str_label_mapping='{self.str_label_mapping}' does not cover all observed "
                    f"diagnosis labels. Missing labels: {missing}"
                )
            logger.warning(
                "Diagnosis labels are not in pipeline label space %s. "
                "Applying configured str_label_mapping='%s'. Observed raw labels: %s",
                sorted(self._get_expected_str_labels()),
                self.str_label_mapping,
                sorted(labels),
            )
            return self.str_label_mapping

        matches = self._detect_suitable_label_mappings(labels)
        if len(matches) == 0:
            raise ValueError(
                f"Diagnosis labels are not already mapped and no suitable string-label mapping was found. "
                f"Observed labels: {sorted(labels)}. "
                f"Please define data.loader.str_label_mapping explicitly in the YAML."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Diagnosis labels are not already mapped and autodetection is ambiguous. "
                f"Observed labels: {sorted(labels)}. Suitable mappings: {matches}. "
                f"Please define data.loader.str_label_mapping explicitly in the YAML."
            )

        logger.warning(
            "Diagnosis labels are not in pipeline label space %s. "
            "Autodetected str_label_mapping='%s'. Observed raw labels: %s",
            sorted(self._get_expected_str_labels()),
            matches[0],
            sorted(labels),
        )
        return matches[0]

    def _map_diagnosis_series_if_needed(self, diagnosis: pd.Series, context: str) -> Tuple[pd.Series, Optional[str]]:
        """
        Map raw diagnosis labels to final pipeline string labels if needed.
        Returns:
            mapped_series, mapping_name_or_None
        """
        diag = self._normalize_diag_series(diagnosis)
        observed = set(diag.dropna().unique().tolist())

        mapping_name = self._resolve_str_label_mapping(observed)
        if mapping_name is None:
            return diag, None

        code_to_label = _build_code_to_label_map(mapping_name)
        mapped = diag.str.lower().map(code_to_label)

        unmapped_mask = mapped.isna()
        if unmapped_mask.any():
            bad = sorted(diag.loc[unmapped_mask].unique().tolist())
            raise ValueError(
                f"{context}: after applying str_label_mapping='{mapping_name}', some diagnosis labels "
                f"remain unmapped: {bad}"
            )

        logger.info(
            "%s: applied str_label_mapping='%s'. Diagnosis counts after mapping: %s",
            context,
            mapping_name,
            mapped.value_counts(dropna=False).to_dict(),
        )
        return mapped, mapping_name

    @staticmethod
    def _normalize_record_id(record_id):
        if pd.isna(record_id):
            return None
        r = str(record_id).strip().lower()
        if r in {"", "none", "nan"}:
            return None
        return r
