import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn import preprocessing

from actitect import utils
from ..processing.apply_smote import apply_smote_with_group_mapping

__all__ = ['FeatureSet', 'Fold']

logger = logging.getLogger(__name__)

SCALER_MAPPING = {'standard': preprocessing.StandardScaler,
                  'minmax': preprocessing.MinMaxScaler,
                  'robust': preprocessing.RobustScaler, }


@dataclass
class FeatureSet:
    x: np.ndarray  # features (n_samples, n_features)
    y: Optional[np.ndarray] = field(default=None)  # labels (n_samples, )
    # group mapping, (n_samples, ), maps the samples (nights) # to the corresponding patient
    group: Optional[np.ndarray] = field(default=None)

    # feature mapping (n_features), maps the feature indices to the feature name
    feat_map: Optional[np.ndarray] = field(default=None)
    # feature ranking: maps each feature to its rank
    feat_rank: Optional[Dict] = field(default=None)
    # if features are processed (e.g. smote, scaling,), contains info about processing params
    process_params: Optional[Dict] = field(default=None)
    # if smote has been used, a flag indicating which samples are synthesized
    smote_mask: Optional[np.ndarray] = field(default=None)
    # if features were used for predictions, include predicted RBD probabilities (n_samples,)
    prob: Optional[np.ndarray] = field(default=None)
    # optional, a list of string labels, e.g. HC, RBD, PD, ..
    y_str: Optional[np.ndarray] = field(default=None)
    # dict-type flag to indicate association to Fold, default is None, i.e. no Fold instance associated
    from_fold: Optional[dict] = field(default=None)
    # optional, mapping between the instance and the corresponding dataset, only relevant for multi-center model
    dataset: Optional[np.ndarray] = field(default=None)  # shape = (n_samples,)
    # optional row-aligned sample metadata. Each row must describe exactly one sample in x.
    metadata: Optional[pd.DataFrame] = field(default=None)

    def __post_init__(self):
        self._validate_sample_alignment()

    def _validate_sample_alignment(self):
        """Validate that all sample-aligned attributes match the number of rows in x."""
        n_samples = int(self.x.shape[0])
        for name in ('y', 'group', 'smote_mask', 'prob', 'y_str', 'dataset'):
            value = getattr(self, name)
            if value is not None and len(value) != n_samples:
                raise ValueError(
                    f"FeatureSet.{name} has length {len(value)}, but x has {n_samples} samples."
                )

        if self.metadata is not None:
            if not isinstance(self.metadata, pd.DataFrame):
                raise TypeError("FeatureSet.metadata must be a pandas DataFrame or None.")
            if len(self.metadata) != n_samples:
                raise ValueError(
                    f"FeatureSet.metadata has {len(self.metadata)} rows, but x has {n_samples} samples."
                )
            self.metadata = self.metadata.reset_index(drop=True).copy()

    @staticmethod
    def _expand_metadata_after_smote(metadata: Optional[pd.DataFrame], smote_mask: np.ndarray) -> Optional[pd.DataFrame]:
        """Expand real-sample metadata to a SMOTE-resampled sample order.

        Synthetic samples intentionally receive missing source metadata and are marked via
        ``is_synthetic=True``. The helper assumes the mask returned by
        ``apply_smote_with_group_mapping`` is aligned with the resampled arrays.
        """
        if metadata is None:
            return None

        mask = np.asarray(smote_mask, dtype=bool)
        n_original = int((~mask).sum())
        if len(metadata) != n_original:
            raise ValueError(
                f"Cannot expand metadata after SMOTE: {len(metadata)} metadata rows for "
                f"{n_original} original samples."
            )

        original = metadata.reset_index(drop=True).copy()
        if 'is_synthetic' not in original.columns:
            original['is_synthetic'] = False
        else:
            original['is_synthetic'] = False

        expanded = pd.DataFrame(index=np.arange(len(mask)), columns=original.columns)
        expanded.loc[~mask, original.columns] = original.to_numpy()
        expanded['is_synthetic'] = mask
        return expanded.reset_index(drop=True)

    def __str__(self):
        ds = set(self.dataset) if self.dataset is not None else None
        probs_shape = self.prob.shape if isinstance(self.prob, np.ndarray) else None
        return (f"FeatureSet(x.shape={self.x.shape}, probs={probs_shape}, process_params={self.process_params}, "
                f"from_fold={self.from_fold}, datasets={ds})")

    def copy(self):
        return deepcopy(self)

    def select_features(
            self,
            feature_names: Union[list, np.ndarray],
            *,
            order: str = "feat_map",
            warn_missing: bool = True,
    ):
        """ Select a subset of features by name and return a new FeatureSet.
        Backward-compatible default:
          - order="feat_map": preserve the existing order of self.feat_map (legacy / safe for old models)
        Optional:
          - order="requested": preserve the order given in `feature_names` (useful if you explicitly want to reorder)
        Parameters:
            :param: feature_names: (List/array) of feature names to keep.
            :param: order: (str) "feat_map" (default) or "requested".
            :param: warn_missing: (bool) If True (default), logs a warning about missing feature names.
        Returns:
            :return: (FeatureSet) A new FeatureSet with x and feat_map restricted to the selected features."""
        assert self.feat_map is not None, "'feat_map' attribute must be set."

        feat_map_arr = np.asarray(self.feat_map)

        # membership lookup
        name_to_idx = {name: i for i, name in enumerate(feat_map_arr)}
        requested = list(feature_names)

        missing = [n for n in requested if n not in name_to_idx]
        if missing and warn_missing:
            logger.warning(
                "select_features: %d feature(s) missing from feat_map (first few: %s)",
                len(missing), missing[:5]
            )

        if order == "feat_map":
            # LEGACY / BACKCOMPAT: preserve feat_map order
            requested_set = set(requested)
            selected_indices = [i for i, name in enumerate(feat_map_arr) if name in requested_set]
        elif order == "requested":
            # EXPLICIT: preserve requested order, skipping missing
            selected_indices = [name_to_idx[n] for n in requested if n in name_to_idx]
        else:
            raise ValueError("select_features: 'order' must be one of {'feat_map', 'requested'}.")

        if not selected_indices:
            raise ValueError("select_features: none of the requested feature names are present.")

        new_x = self.x[:, selected_indices]
        new_feat_map = feat_map_arr[selected_indices]

        return FeatureSet(
            x=new_x,
            y=self.y,
            group=self.group,
            feat_map=new_feat_map,
            process_params=self.process_params,
            prob=self.prob,
            dataset=self.dataset,
            smote_mask=self.smote_mask,
            y_str=self.y_str,
            from_fold=self.from_fold,
            feat_rank=self.feat_rank,
            metadata=self.metadata.copy() if self.metadata is not None else None,
        )

    def select_samples(self, sample_indices):
        """ Select a subset of samples based on provided indices.
        Parameters:
            :param sample_indices: (array-like):Indices of samples to select.
        Returns:
            :return: (FeatureSet) A new FeatureSet instance with selected samples. """
        new_x = self.x[sample_indices, :]
        new_group = self.group[sample_indices] if self.group is not None else None
        new_y = self.y[sample_indices] if self.y is not None else None
        new_prob = self.prob[sample_indices] if self.prob is not None else None
        new_dataset = self.dataset[sample_indices] if self.dataset is not None else None
        new_smote_mask = self.smote_mask[sample_indices] if self.smote_mask is not None else None
        new_y_str = self.y_str[sample_indices] if self.y_str is not None else None
        new_metadata = None
        if self.metadata is not None:
            selected_positions = np.arange(len(self.x))[sample_indices]
            selected_positions = np.atleast_1d(selected_positions)
            new_metadata = self.metadata.iloc[selected_positions].reset_index(drop=True).copy()
        return FeatureSet(x=new_x, y=new_y, group=new_group, feat_map=self.feat_map, process_params=self.process_params,
                          prob=new_prob, dataset=new_dataset, smote_mask=new_smote_mask, y_str=new_y_str,
                          from_fold=self.from_fold, feat_rank=self.feat_rank, metadata=new_metadata)

    def filter_patients_min_nights(self, min_nights: int) -> "FeatureSet":
        """Return a new FeatureSet that keeps only those patients (i.e. `group` IDs)
        that have at least `min_nights` samples per group.
        Parameters
            :param min_nights: (int) Minimum number of samples a patient must contribute to be retained.
        """
        # count samples per patient
        ids, counts = np.unique(self.group, return_counts=True)
        keep_ids = set(ids[counts >= min_nights])

        # boolean mask of samples whose patient ID is in keep_ids
        mask = np.isin(self.group, list(keep_ids))

        return self.select_samples(np.where(mask)[0])

    def fit_transform(self, scaler: str, use_smote: bool, smote_seed: int, scaling_order: Optional[str] = None,
                      rank_kwargs: Optional[dict] = None):
        """ Fit and transform the processing pipeline for the FeatureSet. Should be used on training data only.
        Parameters:
            :param scaler: (str) Type of scaler to apply ('standard', 'minmax', 'robust', or 'none').
            :param use_smote: (bool) Whether to apply SMOTE to the training data.
            :param smote_seed: (int) Seed for SMOTE.
            :param scaling_order: (str)
            :param rank_kwargs (Optional[dict]). Specifies whether and from where the feature rankings should be
                fetched, or computed if the files does not exist yet.
        Returns:
            FeatureSet: The updated FeatureSet instance (self). """
        self.process_params = {'scaler': {'name': scaler}, 'SMOTE': None, 'ranking': None}

        if rank_kwargs:  # get feature ranking
            assert scaling_order in ['before_ranking', 'after_ranking'], \
                f"'scaling_order' must be 'before_ranking' or 'after_ranking', not '{scaling_order}'."
            if scaling_order == 'before_ranking':
                self._apply_scaler()
                self.feat_rank = self._get_feature_ranking(rank_kwargs)
            else:
                self.feat_rank = self._get_feature_ranking(rank_kwargs)
                self._apply_scaler()
        else:  # no feature ranking needed
            self._apply_scaler()

        if use_smote:
            if isinstance(use_smote, str):
                smote_mode = use_smote.lower()
            else:
                smote_mode = 'global'  # legacy True == global

            # Optional: auto-default to per_ds if scaler strategy is per_ds*
            if smote_mode == 'global':
                strat = self.process_params.get('scaler', {}).get('strategy', 'global')
                if isinstance(strat, str) and strat.startswith('per_ds'):
                    smote_mode = 'per_ds'

            self._apply_smote(smote_seed, mode=smote_mode)

        return self

    def transform(self, process_params: Dict, ignore_smote: bool = True):
        """ Transform the FeatureSet using precomputed parameters, with optional control over SMOTE.
        Parameters:
            :param process_params: (dict): Parameters for the processing steps.
            :param ignore_smote: (bool): If True (default), SMOTE will not be applied, regardless of process_params.
                Default is True since .transform() is designed for inference use rather then training.

        Returns:
            FeatureSet: The updated FeatureSet instance (self).
        """
        self.process_params = process_params

        # apply the scaler
        scaler_info = process_params.get('scaler', {})
        self._apply_scaler(scaler_info)

        # optionally apply smote
        smote_info = process_params.get('SMOTE', {})
        if not ignore_smote and smote_info.get('used', False):
            smote_seed = smote_info.get('seed', None)
            if smote_seed is None:
                raise ValueError("SMOTE seed is missing in process_params.")
            self._apply_smote(smote_seed)

        return self

    def merge(self, other: "FeatureSet") -> "FeatureSet":
        """Merge two FeatureSet objects by concatenating their samples.
        Assumes that both feature sets have the same feature mapping and processing parameters.
        Parameters:
            :param other: (FeatureSet) The other FeatureSet to merge with self.
        Returns:
            :return: (FeatureSet) A new FeatureSet with merged data. """
        # check that feature maps match
        if self.feat_map is not None or other.feat_map is not None:
            if not np.array_equal(self.feat_map, other.feat_map):
                raise ValueError("Feature mappings do not match between the two FeatureSets.")
        # merge the arrays
        merged_x = np.vstack((self.x, other.x))
        merged_y = np.concatenate((self.y, other.y))
        merged_group = np.concatenate((self.group, other.group))

        merged_prob = None
        if self.prob is not None and other.prob is not None:
            merged_prob = np.concatenate((self.prob, other.prob))

        merged_dataset = None
        if self.dataset is not None and other.dataset is not None:
            merged_dataset = np.concatenate((self.dataset, other.dataset))

        merged_smote = None
        if self.smote_mask is not None and other.smote_mask is not None:
            merged_smote = np.concatenate((self.smote_mask, other.smote_mask))

        merged_y_str = None
        if self.y_str is not None and other.y_str is not None:
            merged_y_str = np.concatenate((self.y_str, other.y_str))

        if self.metadata is None and other.metadata is None:
            merged_metadata = None
        elif self.metadata is not None and other.metadata is not None:
            if list(self.metadata.columns) != list(other.metadata.columns):
                raise ValueError("Metadata columns do not match between the two FeatureSets.")
            merged_metadata = pd.concat([self.metadata, other.metadata], ignore_index=True)
        else:
            raise ValueError("Cannot merge FeatureSets when only one contains metadata.")

        return FeatureSet(x=merged_x, y=merged_y, group=merged_group, feat_map=self.feat_map, feat_rank=self.feat_rank,
                          process_params=None, smote_mask=merged_smote, prob=merged_prob, y_str=merged_y_str,
                          dataset=merged_dataset, metadata=merged_metadata)

    def get_strat_labels(self, stratify_by_dataset_if_pooled: bool = False):
        """ Create labels for stratification based on the y and dataset. For non_pooled datasets, just by class."""
        is_pooled = self.dataset is not None and len(np.unique(self.dataset)) > 1
        if is_pooled:
            if stratify_by_dataset_if_pooled:
                logger.info("Pooled dataset detected. Performing stratification by class + dataset.")

                if len(self.dataset) != len(self.y):
                    print(self.dataset)
                    print(self.y)
                    raise ValueError(
                        f"Length mismatch between `data.dataset` ({self.dataset.shape}) and `data.y` ({self.y.shape}).")

                dtype = f"<U{max(map(len, self.dataset.astype(str))) + max(map(len, self.y.astype(str))) + 1}"
                sep = np.full(len(self.y), "_", dtype=dtype)
                y_strat = np.char.add(np.char.add(self.dataset.astype(dtype), sep), self.y.astype(dtype))

            else:
                logger.info("Pooled dataset detected. Stratification by dataset is disabled.")
                y_strat = self.y
        else:
            y_strat = self.y

        return y_strat

    def get_feature_indices(self, feature_names: list[str]) -> list[int]:
        """Return indices for the given feature names, using feat_map."""
        if self.feat_map is None:
            raise ValueError("feat_map must be set to resolve feature indices.")
        name_to_idx = {name: i for i, name in enumerate(self.feat_map)}
        try:
            return [name_to_idx[name] for name in feature_names]
        except KeyError as e:
            raise KeyError(f"Feature {e} not found in feat_map.")

    def _apply_scaler(self, scaler_info: Optional[dict] = None):
        """
        Applies scaling to the FeatureSet using the specified or precomputed scaler parameters.

        Strategy encoding (in scaler_info['name']):
          - 'standard' | 'robust' | 'minmax'                 -> global (current behavior)
          - 'standard:per_ds-macro'                          -> per-dataset fit + unweighted macro inference
          - 'standard:per_ds-macro-w'                        -> per-dataset fit + size-weighted macro inference
        (Works analogously for 'robust' and 'minmax'.)

        Notes:
          - Fit path (no 'fitted' flag): compute per-site params from *training* data only,
            then compute a macro 'inference' scaler. No test-site stats are ever added.
          - Transform path ('fitted' flag True): applies per-site scaler for known train sites,
            else applies 'inference' scaler (held-out site).
        """
        if scaler_info is None:  # .fit_transform() case
            scaler_info = self.process_params['scaler']

        name_raw = (scaler_info.get('name') or '').lower()
        if name_raw == '' or name_raw == 'none':
            # no scaling requested
            self.process_params['scaler'] = scaler_info
            return

        # Parse base scaler + strategy suffix
        if ':' in name_raw:
            base_name, strategy = name_raw.split(':', 1)
        else:
            base_name, strategy = name_raw, 'global'

        scaler_class = SCALER_MAPPING.get(base_name)
        if scaler_class is None:
            raise ValueError(f"Scaler type '{base_name}' is not supported. "
                             f"Choose from {list(SCALER_MAPPING.keys())} or 'none'.")

        # Helpers
        def __as_np(a):  # ensure numpy arrays
            return np.asarray(a) if not isinstance(a, np.ndarray) else a

        def __collect_fitted_attrs(scaler_obj):
            """Collect all fitted attributes ending with '_' from a sklearn scaler."""
            out = {}
            for attr in dir(scaler_obj):
                if attr.endswith("_"):
                    val = getattr(scaler_obj, attr, None)
                    if isinstance(val, (list, np.ndarray, float, int)):
                        out[attr] = __as_np(val)
            return out

        def __compute_macro(per_site_dict, _weighted=False, eps=1e-8):
            """
            Compute macro-averaged inference scaler parameters from per-site stats.
            Works for StandardScaler (mean_, scale_), RobustScaler (center_, scale_),
            MinMaxScaler (min_, scale_), etc. We average *all* keys that end with '_'
            and exist in the per-site dicts.
            """
            # Use first site to determine the keys to average
            sample_site = next(iter(per_site_dict.values()))
            feature_keys = [k for k in sample_site.keys() if k.endswith("_")]

            weights = np.array([float(d.get('n', 1)) for d in per_site_dict.values()])
            W = None
            if _weighted and weights.sum() > 0:
                W = (weights / weights.sum())[:, None]

            inference = {'fitted': True, 'name': sample_site.get('name', base_name)}
            for key in feature_keys:
                arrays = np.stack([__as_np(d[key]) for d in per_site_dict.values()], axis=0)
                if W is not None:
                    val = (W * arrays).sum(axis=0)
                else:
                    val = arrays.mean(axis=0)
                if key == "scale_":
                    val = np.maximum(val, eps)  # avoid zero division
                inference[key] = val
            return inference

        # === Fit path (training) ===
        if not scaler_info.get('fitted', False):
            # Global (backward-compatible): fit one scaler on all training samples
            if strategy == 'global':
                scaler = scaler_class()
                self.x = scaler.fit_transform(self.x)
                # persist fitted attributes
                fitted = {'name': base_name, 'strategy': 'global', 'fitted': True}
                fitted.update({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                               for k, v in __collect_fitted_attrs(scaler).items()})
                scaler_info.update(fitted)
                logger.info(f"fitted and applied global {base_name} scaler.")
                self.process_params['scaler'] = scaler_info
                return

            # Per-dataset strategies require dataset labels
            if self.dataset is None:
                raise ValueError("Per-dataset scaling requested but 'dataset' field is None.")

            ds = __as_np(self.dataset)
            unique_sites = np.unique(ds)
            per_site = {}
            X = self.x  # (n_samples, n_features)

            # Fit per-site scaler and transform in-place per site
            for site in unique_sites:
                mask = (ds == site)
                site_scaler = scaler_class()
                X_site = X[mask]
                X[mask] = site_scaler.fit_transform(X_site)
                # persist ALL fitted params dynamically (center_, mean_, scale_, min_, etc.)
                site_dict = {'name': base_name, 'fitted': True, 'n': int(mask.sum())}
                site_dict.update(__collect_fitted_attrs(site_scaler))
                per_site[str(site)] = site_dict

            self.x = X

            # Compute macro inference params (un/weighted)
            weighted = strategy.endswith('-w')
            inference = __compute_macro(per_site, _weighted=weighted)

            scaler_info.update({
                'name': f"{base_name}:{strategy}",
                'strategy': strategy,
                'fitted': True,
                'per_site': {
                    k: {
                        **{kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                           for kk, vv in v.items() if kk.endswith('_')},
                        'n': v['n'], 'name': v['name'], 'fitted': True
                    }
                    for k, v in per_site.items()
                },
                'inference': {
                    **{kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                       for kk, vv in inference.items() if kk.endswith('_')},
                    'name': base_name, 'fitted': True
                }
            })
            logger.info(f"fitted and applied {base_name} scaler with strategy='{strategy}'; "
                        f"sites={list(per_site.keys())}; weighted={weighted}")
            self.process_params['scaler'] = scaler_info
            return

        # === Transform path (inference / later stages) ===
        # Rebuild scaler and apply either per-site or inference params without refitting
        strategy = scaler_info.get('strategy', 'global')
        base_name = scaler_info.get('name', base_name).split(':', 1)[0]  # ensure base name

        if strategy == 'global':
            scaler = scaler_class()
            # load fitted attributes
            for attr, value in scaler_info.items():
                if attr.endswith("_"):
                    setattr(scaler, attr, np.array(value))
            self.x = scaler.transform(self.x)
            logger.info(f"applying pre-fitted global {base_name} scaler.")
            self.process_params['scaler'] = scaler_info
            return

        # Per-dataset transform: use per-site if known train site; otherwise inference macro
        per_site = scaler_info.get('per_site', {})
        inference = scaler_info.get('inference', None)
        if inference is None:
            raise ValueError("Per-dataset scaler requires 'inference' parameters in process_params['scaler'].")

        X = self.x
        if self.dataset is None:
            # No site labels available: apply inference macro to all
            scaler = scaler_class()
            for attr, value in inference.items():
                if attr.endswith("_"):
                    setattr(scaler, attr, np.array(value))
            self.x = scaler.transform(X)
            logger.info(f"applied {base_name} inference macro scaler to all samples (no dataset labels).")
            self.process_params['scaler'] = scaler_info
            return

        ds = __as_np(self.dataset)
        unique_sites_in_batch = np.unique(ds)

        # Apply per-site where available; otherwise inference
        X_out = np.empty_like(X, dtype=float)
        used_inference_for = []
        for site in unique_sites_in_batch:
            mask = (ds == site)
            site_key = str(site)
            scaler = scaler_class()
            params = per_site.get(site_key, None)
            if params is None:
                # held-out site → use inference macro
                used_inference_for.append(site_key)
                params = inference
            for attr, value in params.items():
                if attr.endswith("_"):
                    setattr(scaler, attr, np.array(value))
            X_out[mask] = scaler.transform(X[mask])

        self.x = X_out
        if used_inference_for:
            logger.info(f"applied inference macro scaler for held-out site(s): {used_inference_for}")
        else:
            logger.info(f"applied per-site scalers for known training sites: {list(unique_sites_in_batch)}")

        self.process_params['scaler'] = scaler_info

    def _apply_smote(self, seed: int, mode: str = 'global'):
        """
        Apply SMOTE either globally (pooled) or per dataset (site-wise), and store per-site details.
        - mode: 'global' (default) | 'per_ds'
        Updates:
            self.x, self.y, self.group, self.dataset, self.smote_mask
            self.process_params['SMOTE'] = {
                'used': True,
                'mode': 'global' | 'per_ds',
                'seed': seed,
                'num_new_samples': <int total>,
                'per_site': { '<site_id>': <int new samples>, ... }  # only if dataset labels available
            }
        """
        mode = (mode or 'global').lower()
        if mode not in {'global', 'per_ds'}:
            raise ValueError(f"Unknown SMOTE mode '{mode}'. Use 'global' or 'per_ds'.")

        # Helper: per-site counts for a dataset vector
        def _counts_by_site(ds_vec):
            ds_arr = np.asarray(ds_vec) if ds_vec is not None else None
            if ds_arr is None:
                return {}
            sites, counts = np.unique(ds_arr, return_counts=True)
            return {str(s): int(c) for s, c in zip(sites, counts)}

        # -----------------------
        # GLOBAL SMOTE (pooled)
        # -----------------------
        if mode == 'global' or self.dataset is None:
            before_n_total = int(self.x.shape[0])
            before_site_counts = _counts_by_site(self.dataset)

            x_sm, y_sm, g_sm, ds_sm, m_sm = apply_smote_with_group_mapping(
                self.x, self.y, self.group, self.dataset, seed
            )
            expanded_metadata = self._expand_metadata_after_smote(self.metadata, m_sm)
            self.x, self.y, self.group, self.dataset, self.smote_mask = x_sm, y_sm, g_sm, ds_sm, m_sm
            self.metadata = expanded_metadata

            after_n_total = int(self.x.shape[0])
            num_new_total = after_n_total - before_n_total

            per_site_new = {}
            if ds_sm is not None:
                after_site_counts = _counts_by_site(ds_sm)
                all_sites = set(before_site_counts.keys()) | set(after_site_counts.keys())
                for s in all_sites:
                    per_site_new[s] = int(after_site_counts.get(s, 0) - before_site_counts.get(s, 0))

            self.process_params['SMOTE'] = {
                'used': True,
                'mode': 'global',
                'seed': seed,
                'num_new_samples': int(num_new_total),
                'per_site': per_site_new
            }
            return

        # -----------------------
        # PER-DATASET SMOTE
        # -----------------------
        ds = np.asarray(self.dataset)
        unique_sites = np.unique(ds)

        xs, ys, gs, dss, masks, metadata_parts = [], [], [], [], [], []
        per_site_new = {}
        base_n_total = 0

        for site in unique_sites:
            mask = (ds == site)
            x_sub, y_sub, g_sub = self.x[mask], self.y[mask], self.group[mask]
            ds_sub = self.dataset[mask] if self.dataset is not None else None
            metadata_sub = self.metadata.loc[mask].reset_index(drop=True) if self.metadata is not None else None
            before_n = int(x_sub.shape[0])

            try:
                x_sm, y_sm, g_sm, ds_sm, m_sm = apply_smote_with_group_mapping(
                    x_sub, y_sub, g_sub, ds_sub, seed
                )
            except Exception as e:
                # If SMOTE fails (e.g., too few minority samples), fall back to no change for this site
                logger.warning(f"SMOTE failed for site '{site}': {e}. Skipping SMOTE for this site.")
                x_sm, y_sm, g_sm, ds_sm = x_sub, y_sub, g_sub, ds_sub
                m_sm = np.zeros(before_n, dtype=bool)

            xs.append(x_sm)
            ys.append(y_sm)
            gs.append(g_sm)
            dss.append(ds_sm)
            masks.append(m_sm)
            if self.metadata is not None:
                metadata_parts.append(self._expand_metadata_after_smote(metadata_sub, m_sm))

            after_n = int(x_sm.shape[0])
            per_site_new[str(site)] = int(after_n - before_n)
            base_n_total += before_n

        # Concatenate per-site results
        self.x = np.vstack(xs)
        self.y = np.concatenate(ys)
        self.group = np.concatenate(gs)
        self.dataset = np.concatenate(dss) if dss[0] is not None else None
        self.smote_mask = np.concatenate(masks)
        if self.metadata is not None:
            self.metadata = pd.concat(metadata_parts, ignore_index=True)

        num_new_total = int(sum(per_site_new.values()))
        self.process_params['SMOTE'] = {'used': True, 'mode': 'per_ds', 'seed': seed,
                                        'num_new_samples': num_new_total, 'per_site': per_site_new}

        site_str = ", ".join(f"{s}: +{n}" for s, n in sorted(per_site_new.items()))
        logger.info(
            f"Applied per-dataset SMOTE: {num_new_total} synthetic samples added "
            f"({site_str})")

    def _get_feature_ranking(self, rank_kwargs: dict):
        """
        If rank_kwargs['fair_agg'] is set and there are multiple datasets in this FeatureSet,
        load/compute one cached ranking per dataset under <root>/by_dataset/<DS>/..., then
        aggregate (mean/median/borda) across the train datasets. The aggregated ranking is
        cached under <root>/by_dataset_fair/fair_<agg>__<DS1+DS2+...>.csv, keyed by the
        train set composition. Otherwise, fall back to the standard single-run ranking.
        """
        from pathlib import Path
        import numpy as np
        import pandas as pd
        from ..processing.feature_ranking import FeatureRanker

        # Parse fair aggregation request
        fair_agg = rank_kwargs.get('fair_agg')
        use_fair = bool(fair_agg)
        agg = fair_agg if fair_agg else None

        # Keep only valid FeatureRanker kwargs
        base_root = Path(rank_kwargs['root_dir']) if 'root_dir' in rank_kwargs else None
        valid_ranker_kwargs = {
            key: rank_kwargs[key]
            for key in (
                'root_dir',
                'data_config',
                'n_jobs',
                'draw_plots',
                'random_state',
                'legacy_mode',
                'feature_selection_config',
            )
            if key in rank_kwargs
        }

        # Simple path: single dataset or fair disabled -> vanilla ranking
        if (not use_fair) or (self.dataset is None) or (len(np.unique(self.dataset)) <= 1):
            ranker = FeatureRanker(**valid_ranker_kwargs)
            return ranker.fetch_or_compute(data=self, return_df=False)

        # FAIR path: reuse per-dataset caches and aggregate for current train composition
        unique_ds = np.unique(self.dataset)
        # Cache aggregated FAIR ranking keyed by train-set composition (sorted for stability)
        train_key = "+".join(map(str, sorted(unique_ds)))
        fair_folder = utils.check_make_dir(base_root.joinpath('by_dataset_fair'), use_existing=True)
        fair_cache = fair_folder.joinpath(f"fair_{agg}__{train_key}.csv")

        logger.info(
            f"using averaged feature ranking across {len(unique_ds)} datasets "
            f"with '{agg}' aggregation."
        )

        # If a combined fair cache exists, just load and map to the usual dict format
        if fair_cache.exists():
            rank_df = pd.read_csv(fair_cache).set_index('total_rank')
            _feature_ranking = rank_df.name.values
            return {
                name: {'idx': idx, 'rank': int(np.where(_feature_ranking == name)[0][0] + 1)}
                for idx, name in enumerate(self.feat_map)
            }

        # Otherwise, load/compute per-dataset caches (ONE per dataset; no fold-scoped paths) and aggregate
        per_ds_rank = {}
        for ds in unique_ds:
            idx = np.where(self.dataset == ds)[0]
            fs_ds = self.select_samples(idx)

            # Force "non-fold" mode so FeatureRanker writes/reads from <root>/by_dataset/<DS>/...
            setattr(fs_ds, 'from_fold', None)

            rk_kwargs = dict(valid_ranker_kwargs)
            rk_kwargs['root_dir'] = base_root.joinpath('by_dataset', str(ds))

            # This will READ the cached CSV if present; otherwise compute once and write it
            rk_df = FeatureRanker(**rk_kwargs).fetch_or_compute(data=fs_ds, return_df=True)  # index: feature idx
            # Normalize to {name -> rank} using current FeatureSet feature ordering/names
            ser_by_name = pd.Series(rk_df['total_rank'].values,
                                    index=[self.feat_map[i] for i in rk_df.index])
            per_ds_rank[str(ds)] = ser_by_name

        # Build a DataFrame: rows=feature names, cols=datasets, values=per-dataset ranks
        rank_mat = pd.DataFrame(per_ds_rank)  # shape (n_feats, n_ds)

        # Aggregate across datasets
        if agg == 'median':
            agg_val = rank_mat.median(axis=1)
        elif agg == 'borda':
            # Convert ranks (1..K) to scores (K..1), average scores, then invert so lower is better
            K = rank_mat.shape[0]
            scores = K - rank_mat + 1
            agg_val = -(scores.mean(axis=1))  # negative so sorting ascending gives best first
        else:  # default to 'mean'
            agg_val = rank_mat.mean(axis=1)

        # Create combined fair ranking DataFrame in the same shape as FeatureRanker outputs
        ordered = agg_val.sort_values(kind='mergesort')  # stable
        out = pd.DataFrame({'name': ordered.index, 'fair_agg_val': ordered.values})
        out['total_rank'] = np.arange(1, len(out) + 1)

        # Save the fair cache keyed by train composition
        out.to_csv(fair_cache, index=False)

        # Return in the usual dict format expected downstream
        name_order = out['name'].values
        return {
            name: {'idx': idx, 'rank': int(np.where(name_order == name)[0][0] + 1)}
            for idx, name in enumerate(self.feat_map)
        }

    def split_by_dataset(self) -> dict:
        """ Split a FeatureSet into per-dataset FeatureSets.
        Returns an ordered dict {<dataset_id str>: FeatureSet} preserving all fields."""
        if self.dataset is None:
            raise ValueError("FeatureSet.dataset is None — nothing to split on.")
        ds = np.asarray(self.dataset)
        uniq = [str(u) for u in np.unique(ds)]  # stable, sorted

        out = {}
        for u in uniq:
            idx = np.where(ds == u)[0]
            fs_u = self.select_samples(idx)  # preserves feat_map, prob, y_str, smote_mask, feat_rank, process_params
            # Keep dataset vector restricted to this site (select_samples already did this)
            out[u] = fs_u
        return out


@dataclass
class Fold:
    name: str
    k: int
    feature_set: FeatureSet

    def __str__(self):
        return f"Fold({self.name}, k={self.k}, x={self.x.shape}, feature_set={self.feature_set})"

    def __post_init__(self):
        self.feature_set.from_fold = {'name': self.name, 'k': self.k}  # set the flag on the underlying FeatureSet:

    def __getattr__(self, attr):
        """Delegate methods and attributes to feature_set. Wrap returned FeatureSet
        objects in a Fold to preserve the interface."""
        fs = object.__getattribute__(self, "feature_set")  # avoid RecursionError
        feature_set_attr = getattr(fs, attr)
        if callable(feature_set_attr):
            def _delegated_method(*args, **kwargs):
                result = feature_set_attr(*args, **kwargs)
                if isinstance(result, FeatureSet):  # if delegated method returns FeatureSet, wrap it in a Fold.
                    if result is self.feature_set:
                        return self  # if same internal FeatureSet, return self
                    else:
                        return Fold(name=self.name, k=self.k, feature_set=result)
                return result

            return _delegated_method
        return feature_set_attr

    def __setattr__(self, name, value):  # backward delegation, e.g. Fold.probs == .. -> also set for inner FeatureSet!
        if name in {"name", "k", "feature_set"} or name.startswith("_"):
            super().__setattr__(name, value)  # keep Fold's own fields
        elif "feature_set" in self.__dict__ and hasattr(self.feature_set, name):
            setattr(self.feature_set, name, value)  # write-through to inner FeatureSet
        else:
            super().__setattr__(name, value)  # fallback: new attr on Fold
