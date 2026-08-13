import json
import logging
import math
import sys
import tracemalloc
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.utils.validation import has_fit_parameter
from skopt.plots import plot_convergence

from actitect import utils
from actitect.config import PipelineConfig, DatasetConfig, ModelConfig
from actitect import vis
from .cv_iterator import cv_iterator
from .hp_optimizers import BayesianOptCV
from ..core import Evaluator, FeatureSet, Fold
from ..models import ModelFactory, ModelSetup
from ..processing.classification_threshold import get_night_and_patient_threshold
from ..processing.probability_calibration import CustomCalibratedClassifierCV
from ..processing.metrics import calc_evaluation_metrics

logger = logging.getLogger(__name__)

__all__ = ['KFoldNestedCV', 'LODONestedCV', 'PredefinedKFoldNestedCV']


class NestedCVBase(ABC):

    def __init__(self, config: PipelineConfig, save_path: Path, early_stopping_options: list = None,
                 calibration: str = None):
        self.config = config
        self.save_path = save_path
        self.random_state = config.random_state
        self.n_jobs = config.nested_cv.n_jobs
        self.n_datasets, self.inner_seeds, self.n_repeats, self.outer_splits = None, None, None, None

        self.calibration = calibration
        self.early_stopping_options = early_stopping_options if early_stopping_options is not None \
            else [True, False] if config.model.which == 'xgboost' else [False]

        self.use_fixed_features = bool(getattr(self.config.model.feature_selection, 'fixed_features', None))

        raw_ds_mode = getattr(config.model, 'dataset_weighting', None)  # e.g. dsw:bay_on
        if raw_ds_mode:
            mode_part, flag_part = (raw_ds_mode.split(":", 1) + ['bay_off'])[:2]
            mode, flag = mode_part.lower().strip(), flag_part.lower().strip()
            if mode not in {'dsw', 'dswc'}:
                logger.warning(
                    f"Unknown dataset_weighting mode '{mode_part}', expected 'dsw' or 'dswc'. Disabling weighting.")
                self.ds_weighting_kwargs = None
            else:
                use_bayes = flag in {'bay_on', 'bon'}
                logger.info(f"Using dataset_weighting mode '{mode}' with using bayes = {use_bayes}.")
                self.ds_weighting_kwargs = {'mode': mode, 'use_bayes': use_bayes}
        else:
            self.ds_weighting_kwargs = None

    def __str__(self) -> str:

        def __fmt(val):
            """Return val as str, but show “?” when None."""
            return "?" if val is None else val

        flavour = self.__class__.__name__
        return (
            f"{flavour}("
            f"datasets={__fmt(self.n_datasets)}, "
            f"repeats={__fmt(self.n_repeats)}, "
            f"outer_splits={__fmt(getattr(self, 'outer_splits', None))},"
            f"inner_splits={self.config.nested_cv.inner_cv.n_splits}, "
            f"n_jobs={self.n_jobs}, seed={self.random_state})"
        )

    @abstractmethod
    def _get_outer_splitter(self, seed: int):
        """Should return a parametrised sklearn splitter instance (that supports grouping)."""
        raise NotImplementedError('abstract method')

    @abstractmethod
    def _get_outer_groups(self, data: FeatureSet):
        """Array passed as `groups=` to cv_iterator for the outer loop."""
        raise NotImplementedError('abstract method')

    def _get_inner_splits(self, train_outer: Fold, *, repeat: int, outer_fold: int):
        """Return predefined inner splits relative to ``train_outer``, or ``None`` for generated CV."""
        return None

    def _validate_data(self, data: FeatureSet):
        """Validate data requirements specific to a Nested-CV flavour."""
        return None

    def fit(self, data: FeatureSet):

        # post-init setup (needs data)
        self._validate_data(data)
        self.n_datasets = self._count_unique_datasets(data)
        if isinstance(self, PredefinedKFoldNestedCV):
            self.n_repeats = len(self.repeat_labels)
            self.outer_splits = len(self.outer_fold_labels)
        else:
            self.n_repeats = 1 if isinstance(self, LODONestedCV) else self.config.nested_cv.n_repeats
            self.outer_splits = self.n_datasets if isinstance(self, LODONestedCV) \
                else self.config.nested_cv.outer_cv.n_splits
        self.inner_seeds = self._generate_random_seeds(self.random_state, self.n_repeats, self.outer_splits)
        if data.dataset is None:
            ds_summary = {"single": len(data.y)}
        else:
            labels, counts = np.unique(data.dataset, return_counts=True)
            ds_summary = {str(_l): int(_c) for _l, _c in zip(labels, counts)}
        seeds_dict = dict(outer=[self.random_state + r for r in range(self.n_repeats)], inner=self.inner_seeds)
        _manifest_kwargs = dict(
            save_dir=self.save_path, flavour=self.__class__.__name__.replace("NestedCV", "").lower(),
            repeats=self.n_repeats, outer_folds_expected=self.outer_splits,
            inner_folds=self.config.nested_cv.inner_cv.n_splits, seeds=seeds_dict, dataset_summary=ds_summary)

        with (utils.custom_tqdm(total=self.n_repeats) as pbar, self._manifest_guard(**_manifest_kwargs) as manifest):
            pbar.set_description(f"[PROGRESS]: {self.__class__.__name__}")
            _process_kwargs = self.config.data.processing.dict()
            _process_kwargs.update({'smote_seed': self.random_state})
            logger.info(f"({self.__class__.__name__}): using processing params {_process_kwargs}.")
            if isinstance(self, LODONestedCV):
                manifest['fold_to_dataset'] = {}
            tracemalloc.start()
            for repeat in range(self.n_repeats):
                outer_seed = self.random_state + repeat
                outer_cv = self._get_outer_splitter(outer_seed)

                _fixed_rank_seed = getattr(self.config.nested_cv, "ranking_seed", None)
                if _fixed_rank_seed is not None:
                    logger.warning(
                        f"Overriding ranking seed with fixed value {_fixed_rank_seed} "
                        f"(ignoring outer_seed={outer_seed}).")
                rank_seed = outer_seed if _fixed_rank_seed is None else int(_fixed_rank_seed)

                _rankings_root_dir = self.config.nested_cv.load_path_cv_feature_rankings.joinpath(
                    f"seed_{rank_seed}_{sys.platform}")
                manifest.setdefault("rankings_root_dirs", {})[f"seed{outer_seed}"] = str(_rankings_root_dir)
                _rank_kwargs = {
                    'root_dir': _rankings_root_dir,
                    'data_config': self.config.data,
                    'n_jobs': self.n_jobs,
                    'fair_agg': self.config.model.feature_selection.fair_agg,
                    'feature_selection_config': self.config.model.feature_selection,
                } if not self.use_fixed_features else None

                save_path_repeat = utils.check_make_dir(self.save_path.joinpath(
                    f"repeats/repeat{repeat}_seed{outer_seed}"), True)

                tasks = [delayed(self._perform_outer_fold)(
                    k_outer, train_outer, valid_outer, save_path_repeat, outer_seed=outer_seed
                ) for k_outer, train_outer, valid_outer in cv_iterator(
                    outer_cv, data, _process_kwargs, _rank_kwargs,
                    self.config.nested_cv.stratify_by_dataset_if_pooled, groups=self._get_outer_groups(data))]

                with Parallel(n_jobs=self.n_jobs) as parallel:
                    ds_info = parallel(tasks)

                # in LODO case, keep track of which fold belongs to which held-out dataset
                if isinstance(self, LODONestedCV) and ds_info is not None:
                    fold_to_dataset = {f"seed{r}_fold{f}": ds for r, f, ds in ds_info if r is not None}
                    manifest['fold_to_dataset'] = fold_to_dataset

                pbar.update(1)
                current, peak = tracemalloc.get_traced_memory()
                logger.info(f"repeat {repeat + 1}/{self.n_repeats}:"
                            f"current memory usage: {current / 1024 / 1024:.2f} MB;"
                            f"peak: {peak / 1024 / 1024:.2f} MB")

                tracemalloc.clear_traces()

            tracemalloc.stop()

    def _perform_outer_fold(self, k_outer, train_outer, valid_outer, save_path_repeat, *, outer_seed):

        with (self._handle_parallel_logging()):

            save_path_fold = utils.check_make_dir(save_path_repeat.joinpath(f'outer_fold_{k_outer}'), True)
            inner_seed = self.inner_seeds[outer_seed][k_outer]
            self._set_random_seed(inner_seed)
            utils.dump_to_json({'seed': outer_seed, 'fold': k_outer,
                                'train': np.unique(train_outer.group), 'valid': np.unique(valid_outer.group)},
                               save_path_fold.joinpath(f"groups.json"))

            # set up the model:
            model_setup = self._initialize_model(self.config.model, train_outer, inner_seed)

            # fit and eval non-optimized model:
            default_model = model_setup.model().set_params(**model_setup.default_params)
            _default_scores = self._fit_and_eval_model(default_model, train_outer, valid_outer)
            logger.info(f"default model outer fold {k_outer}: {utils.fmt_dict(_default_scores['valid'])}")
            utils.dump_to_json(_default_scores, save_path_fold.joinpath(f'default_model_scores.json'))
            del _default_scores

            # perform BayesOpt using Inner-CV:
            fit_params = self.config.model.bayes_params.dict().copy()
            if self.use_fixed_features:
                model_setup.bayesian_param_space = \
                    [p for p in model_setup.bayesian_param_space if p.name != "top_k_feats"]
                logger.warning("Subsetting training data to fixed features before BayesOpt.")
                train_for_bayes = train_outer.select_features(
                    self.config.model.feature_selection.fixed_features)
            else:
                train_for_bayes = train_outer

            if not model_setup.bayesian_param_space:
                logger.warning("No tunable hyperparameters (fixed features). Skipping BayesOpt.")
                opt_params, best_score = {}, None
            else:
                fit_params.update({'cv_params': {
                    'group': train_outer.group, 'n_folds': self.config.nested_cv.inner_cv.n_splits,
                    'shuffle': self.config.nested_cv.inner_cv.shuffle, 'verbose': False},
                    'n_jobs': 1 if self.n_jobs != 1 else -1,
                    'dataset_weighting': None, 'ds_vector': None})

                if self.ds_weighting_kwargs and self.ds_weighting_kwargs.get('use_bayes', False):
                    fit_params.update({'dataset_weighting': self.ds_weighting_kwargs['mode'],
                                       'ds_vector': getattr(train_outer, 'dataset', None)})

                repeat = outer_seed - self.random_state
                inner_splits = self._get_inner_splits(
                    train_for_bayes, repeat=repeat, outer_fold=k_outer)
                opt_params, best_score = self._perform_inner_cv_bayes_opt(
                    outer_fold=train_for_bayes, model_setup=model_setup, rank_map=train_outer.feat_rank,
                    save_path=save_path_fold, fit_params=fit_params, inner_cv_seed=inner_seed,
                    stratify_by_dataset=self.config.nested_cv.stratify_by_dataset_if_pooled,
                    cv_splits=inner_splits)

            best_params = model_setup.bayesian_fixed_params.copy()
            best_params.update(opt_params)  # opt_params is empty if bayes skipped, so will fallback to library defaults

            if not self.use_fixed_features:  # regular case, feature selection with ranking and top_k tuned
                _top_k = best_params.pop('top_k_feats')
                _top_k_feat_names = {
                    name: val['rank'] for name, val in train_outer.feat_rank.items() if val['rank'] <= _top_k}
                _top_k_feat_names = dict(sorted(_top_k_feat_names.items(), key=lambda item: item[1]))
                utils.dump_to_json(_top_k_feat_names, save_path_fold.joinpath(f'top_k_feat_names.json'))
                selected_feats = list(_top_k_feat_names.keys())
            else:
                logger.warning(
                    f"Using fixed feature set with {len(self.config.model.feature_selection.fixed_features)} features.")
                selected_feats = self.config.model.feature_selection.fixed_features
                utils.dump_to_json(selected_feats, save_path_fold.joinpath('fixed_feat_names.json'))

            # choose only selected features
            train_outer_top_k = train_outer.select_features(selected_feats)
            valid_outer_top_k = valid_outer.select_features(selected_feats)

            if self.ds_weighting_kwargs and train_outer_top_k.dataset is None:
                logger.warning("dataset_weighting is set but no dataset labels present; skipping weighting.")

            ds_sample_weight_train = None
            if self.ds_weighting_kwargs and (train_outer_top_k.dataset is not None):
                weight_mode = self.ds_weighting_kwargs['mode']
                if weight_mode not in {'dsw', 'dswc'}:
                    logger.warning(
                        f"Unknown dataset_weighting='{weight_mode}', expected 'dsw' or 'dswc'; skipping weighting.")
                else:

                    ds_sample_weight_train = utils.compute_composite_dataset_weights(
                        ds_vec=train_outer_top_k.dataset, y_vec=train_outer_top_k.y, mode=weight_mode)

                    # log composition + weight range
                    u_tr, c_tr = np.unique(train_outer_top_k.dataset, return_counts=True)
                    comp_tr = ", ".join(f"{u}: n={c}" for u, c in zip(u_tr, c_tr))
                    logger.info(f"Main fit weighting mode='{weight_mode}'; train comp [{comp_tr}];"
                                f" w-range=[{ds_sample_weight_train.min():.3f}, {ds_sample_weight_train.max():.3f}]")

            # fit and eval optimized model:
            for early_stopping in self.early_stopping_options:

                opt_model = model_setup.model().set_params(**best_params)
                logger.info(f"Bayesian opt: setting model params to {utils.fmt_dict(opt_params, 3)}.")
                _eval_set = [(train_outer_top_k.x, train_outer_top_k.y), (valid_outer_top_k.x, valid_outer_top_k.y)]

                if early_stopping:
                    opt_model.set_params(**self.config.model.early_stopping.dict())
                    _suffix = 'early_stopping'
                else:
                    opt_model.set_params(**{'early_stopping_rounds': None, 'eval_metric': None})  # reset
                    _suffix = 'no_early_stopping'

                if self.calibration:
                    calib_opt = self.calibration.lower()
                    use_unsmoted = ':unsmoted' in calib_opt
                    use_dsw = ':dsw' in calib_opt or ':dswc' in calib_opt
                    use_dswc = ':dswc' in calib_opt
                    base_method = calib_opt.split(':', 1)[0]  # 'sigmoid' or 'isotonic'
                    assert base_method in {'sigmoid', 'isotonic'}, \
                        (f"calibration must be 'sigmoid'/'isotonic' (optionally with ':unsmoted or :dsw/dswc'),"
                         f" not '{self.calibration}'.")
                    mode_str = "unSMOTEd" if use_unsmoted else "SMOTEd"
                    w_mode = "dswc" if use_dswc else ("dsw" if use_dsw else "none")
                    logger.info(f"Calibration config: method='{base_method}', data={mode_str}, weighting={w_mode}.")

                    if use_unsmoted:
                        sm = getattr(train_outer_top_k, 'smote_mask', None)
                        if sm is None:
                            logger.warning("Calibration ':unsmoted' requested, but 'smote_mask' is None. "
                                           "Assuming no SMOTE; using all training samples for calibration.")
                            true_mask = np.ones(len(train_outer_top_k.y), dtype=bool)
                        else:
                            true_mask = ~sm
                            n_total, n_kept = len(sm), int(true_mask.sum())
                            logger.info(f"Calibration subset: kept {n_kept}/{n_total} original (unSMOTEd) samples.")

                        # build a consistent calibration FeatureSet view,  using only original samples
                        fs_calib: FeatureSet = train_outer_top_k.select_samples(np.where(true_mask)[0])
                    else:
                        logger.info("Calibration on SMOTEd training data (prevalence may be distorted).")
                        fs_calib = train_outer_top_k  # use full SMOTEd fold

                    # stratification labels (class × dataset if enabled)
                    y_str_cal = fs_calib.get_strat_labels(self.config.nested_cv.stratify_by_dataset_if_pooled)

                    _sgkf = StratifiedGroupKFold(**self.config.nested_cv.inner_cv.dict(), random_state=inner_seed)
                    cv_splits = list(_sgkf.split(X=fs_calib.x, y=y_str_cal, groups=fs_calib.group))
                    logger.info(f"Calibration CV: {len(cv_splits)} folds with StratifiedGroupKFold.")

                    if use_dsw or use_dswc:  # optionally get sample weights to balance dataset biases in calibration
                        if fs_calib.dataset is None:  # no site labels available → fall back to unweighted
                            calib_sample_weight = None
                            logger.warning("Calibration requested with dataset weighting, but no dataset labels found; "
                                           "falling back to unweighted calibration.")
                        else:
                            uniq_ds = np.unique(fs_calib.dataset)
                            if len(uniq_ds) == 1:
                                logger.warning(
                                    "Dataset weighting requested but calibration view contains a single dataset; "
                                    "weights will be effectively uniform.")
                            ds_calib = fs_calib.dataset
                            calib_sample_weight = utils.compute_composite_dataset_weights(ds_calib, fs_calib.y, w_mode)

                            # extra sanity/logging
                            uniq, cnt = np.unique(ds_calib, return_counts=True)
                            comp = ", ".join(f"{u}: n={c}" for u, c in zip(uniq, cnt))
                            w_min, w_max = float(calib_sample_weight.min()), float(calib_sample_weight.max())
                            logger.info(f"Calibration dataset composition [{comp}]; "
                                        f"weight mode={'dswc' if use_dswc else 'dsw'}; "
                                        f"weight range [{w_min:.3f}, {w_max:.3f}].")
                    else:
                        calib_sample_weight = None

                    calibrated_cv = CustomCalibratedClassifierCV(
                        estimator=opt_model, method=base_method, cv=cv_splits, pass_fold_as_eval_set=True,
                        n_jobs=self.n_jobs)
                    logger.info(f"Training calibrated model with method='{base_method}' "
                                f"on {'unSMOTEd' if use_unsmoted else 'SMOTEd'} calibration data.")
                    if calib_sample_weight is not None:
                        assert len(calib_sample_weight) == len(fs_calib.y), \
                            "sample_weight length must match calibration sample count."
                    opt_model = calibrated_cv.fit(
                        fs_calib.x, fs_calib.y, groups=fs_calib.group, sample_weight=calib_sample_weight, verbose=False)

                else:

                    fit_kwargs = dict()
                    for param_name, param_val in \
                            {'sample_weight': ds_sample_weight_train, 'eval_set': _eval_set, 'verbose': False}.items():
                        if not param_val is None:
                            if has_fit_parameter(opt_model, param_name):
                                fit_kwargs[param_name] = param_val
                            else:
                                logger.warning(f"Estimator does not support '{param_name}' parameter.")
                    if fit_kwargs.get('sample_weight', None) is not None:
                        logger.info(f"Using dataset sampling weights for model fit.")
                    opt_model.fit(train_outer_top_k.x, train_outer_top_k.y, **fit_kwargs)
                    try:
                        opt_model.set_params(**{'early_stopping_rounds': None, 'eval_metric': None})  # reset
                    except (ValueError, TypeError, AttributeError):
                        pass

                # predict probabilities and calculate thresholds
                train_outer_top_k.prob = opt_model.predict_proba(train_outer_top_k.x)[:, 1]
                valid_outer_top_k.prob = opt_model.predict_proba(valid_outer_top_k.x)[:, 1]

                fold_night_threshold, fold_patient_threshold = get_night_and_patient_threshold(
                    train_outer_top_k, self.config.nested_cv.default_experiment.threshold)
                thresholds = {'night': fold_night_threshold, 'patient': fold_patient_threshold}

                # evaluate on all patients and, as a sensitivity check, on patients with ≥ 2 nights
                _min_n_nights = self.config.nested_cv.min_patient_nights_eval
                for _valid_set, _save_tag in [
                    (valid_outer_top_k, _suffix),  # all patients
                    (valid_outer_top_k.filter_patients_min_nights(  # at least N nights
                        _min_n_nights), f"{_suffix}_min_{_min_n_nights}_nights")]:
                    if _valid_set.x.size == 0:  # nothing to score after filtering
                        continue
                    _out_dir = utils.check_make_dir(save_path_fold.joinpath(_save_tag), True)
                    # Export raw predictions only for the primary, unfiltered validation set.
                    # The minimum-night sensitivity subset contains duplicate samples.
                    _output_predictions = self.config.nested_cv.output_predictions and _save_tag == _suffix
                    ev = Evaluator(_out_dir, self.config.nested_cv.default_experiment, thresholds,
                                   cv_config=self.config.nested_cv, cv_mode=True,
                                   output_patient_csv=_output_predictions,
                                   output_night_csv=_output_predictions,
                                   bootstrap_ci=isinstance(self, LODONestedCV),
                                   extra_diagnostic_metrics=bool(self.config.nested_cv.extra_diagnostic_metrics))
                    ev.evaluate(
                        train_outer_top_k, _valid_set, generate_night_output=self.config.nested_cv.log_night_eval)

                del opt_model

            del model_setup

            if isinstance(self, LODONestedCV):
                ds = str(np.unique(valid_outer.dataset)[0])
                return outer_seed, k_outer, ds

            return None

    @staticmethod
    def _perform_inner_cv_bayes_opt(outer_fold: Fold, model_setup: ModelSetup, rank_map: dict, fit_params: dict,
                                    save_path: Path, inner_cv_seed: int, stratify_by_dataset: bool,
                                    plot_search_history: bool = True, cv_splits=None):
        # run Bayesian Optimization on inner-cv:
        bayesian_opt = BayesianOptCV(model_setup=model_setup, feat_rank_map=rank_map, seed=inner_cv_seed)
        y_strat = outer_fold.get_strat_labels(stratify_by_dataset)
        results = bayesian_opt.fit(outer_fold.x, outer_fold.y, y_strat, **fit_params, cv_splits=cv_splits)

        opt_params = {dim.name: val for dim, val in zip(model_setup.bayesian_param_space, results.x)}
        logger.info(f"best score: {-results.fun:.3f} with params {utils.fmt_dict(opt_params, 3)}")

        # saving and plotting:
        save_path_bay_opt = utils.check_make_dir(save_path.joinpath(f'bay_opt'), use_existing=True)
        _best_params = {'best_params': opt_params, 'best score': -results.fun}
        for filename, _dict in zip(
                ['best_params', 'fixed_params', 'param_space'],
                [_best_params, model_setup.bayesian_fixed_params, model_setup.bayesian_param_space]
        ):
            utils.dump_to_json(_dict, save_path_bay_opt.joinpath(f"{filename}.json"))

        if plot_search_history:  # plot hp search
            plt.close('all')
            _plotting_funcs = {
                'convergence': (plot_convergence, {}),
                # 'evaluations': (plot_evaluations, {'cmap': 'YlGnBu_r'}),
                # 'objective': (plot_objective, {'sample_source': 'result', 'cmap': 'YlGnBu'})
            }
            for _name, (func, kwargs) in _plotting_funcs.items():
                for _ext in ['png']:  # , 'pdf'):
                    func(results, **kwargs).get_figure() \
                        .savefig(save_path_bay_opt.joinpath(f'{_name}.{_ext}'), bbox_inches='tight')
                plt.close('all')

        return opt_params, -results.fun

    @staticmethod
    def _initialize_model(model_cfg: ModelConfig, fold: Fold, random_state: int):
        _n_hc_train_outer, _n_rbd_train_outer = np.unique(fold.y, return_counts=True)[1]
        _cls_balance_outer = _n_hc_train_outer / _n_rbd_train_outer
        model_factory = ModelFactory(
            model_cfg.which, cls_balance=_cls_balance_outer, seed=random_state,
            top_k_cfg=model_cfg.feature_selection.top_k_feats, overrides=model_cfg.hp_overrides)
        return model_factory.build()

    @staticmethod
    def _fit_and_eval_model(model, fold_train: Fold, fold_valid: Fold, threshold: float = .5):
        model.fit(fold_train.x, fold_train.y)
        return {'train': calc_evaluation_metrics(
            fold_train.y, y_prob=model.predict_proba(fold_train.x)[:, 1], threshold=threshold),
            'valid': calc_evaluation_metrics(
                fold_valid.y, y_prob=model.predict_proba(fold_valid.x)[:, 1], threshold=threshold)}

    def eval(self):
        cv_subdir = self._restore_cv_state_from_manifest()

        fold_dirs = sorted([_dir for _dir in cv_subdir.glob("repeats/**/outer_fold_*") if _dir.is_dir()])
        _expected_outer = self.n_datasets if isinstance(self, LODONestedCV) else self.config.nested_cv.outer_cv.n_splits
        assert len(fold_dirs) == self.n_repeats * _expected_outer, \
            (f"incomplete outer cv detected, found {len(fold_dirs)} folds at {self.save_path.stem} "
             f"but got n_splits={_expected_outer}.")

        # HP* search summary
        hp_summary, top_k_summary = self._make_hp_summary(fold_dirs)
        hp_summary.to_csv(cv_subdir.joinpath('hp_summary.csv'))
        top_k_summary.to_csv(cv_subdir.joinpath('top_k_summary.csv'))

        # collect the scores anr roc/pr curves per fold:
        scores, curves, cms, from_fold = self._collect_scoring_history(
            fold_dirs, self.early_stopping_options, min_n_nights=self.config.nested_cv.min_patient_nights_eval)

        # boxplot of the scores:
        bp_night, bp_patient = self._draw_history_boxplots(scores, self.config.nested_cv.log_night_eval)
        for _lvl, _fig_dict in zip(['night', 'patient'], [bp_night, bp_patient]):
            if _fig_dict:  # non-empty dict
                __save_path = utils.check_make_dir(cv_subdir.joinpath(f'boxplots/{_lvl}'), True)
                for _name, _fig in _fig_dict.items():
                    _fig.savefig(__save_path.joinpath(f'{_name}.png'), dpi=400, bbox_inches='tight')

        # mean roc/pr curve plots:
        if isinstance(self, LODONestedCV):
            lodo_label_map = DatasetConfig.from_yaml()
            ds_labels = [self.fold_2_dataset.get(
                f"seed{self.random_state}_fold{_fold.replace('outer_fold_', '')}", "UNKNOWN")
                for _fold in from_fold]
            lodo_labels = [lodo_label_map.resolve(lbl) for lbl in ds_labels]
        else:
            lodo_labels = None

        curve_figs_night, curve_figs_patient = self._draw_mean_curves(
            curves, self.config.nested_cv.log_night_eval, lodo_labels)
        for _lvl, _fig_dict in zip(['night', 'patient'], [curve_figs_night, curve_figs_patient]):
            if _fig_dict:  # non-empty dict
                _save_path_curves = utils.check_make_dir(cv_subdir.joinpath(f'curves/{_lvl}'), True)
                for _name, _fig in _fig_dict.items():
                    _fig.savefig(_save_path_curves.joinpath(f'{_name}.png'), dpi=400, bbox_inches='tight')

        if isinstance(self, LODONestedCV):
            self._summarize_lodo_performance(scores, from_fold, output_dir=cv_subdir)

    @staticmethod
    def _generate_random_seeds(initial_seed, n_repeats, n_splits_outer):
        offset = n_repeats * n_splits_outer if initial_seed == 0 else 0
        factor = max(10, 10 ** (math.ceil(math.log10(n_repeats * n_splits_outer)) - 1))
        outer_seeds, inner_seeds = [initial_seed + i for i in range(n_repeats)], {}
        for i, outer_seed in enumerate(outer_seeds):
            inner_seeds[outer_seed] = [(offset + (outer_seed * factor) + j) for j in range(n_splits_outer)]
        all_seeds = outer_seeds + [seed for inner_list in inner_seeds.values() for seed in inner_list]
        assert len(all_seeds) == len(np.unique(all_seeds)), \
            "found non-unique seeds in 'inner_seeds'. Fix it to ensure reproducibility."
        return inner_seeds

    @staticmethod
    def _set_random_seed(seed):
        np.random.seed(seed)

    @staticmethod
    def _make_hp_summary(fold_dirs: list[Path]):
        opt_params, top_k_feats = pd.DataFrame(), pd.DataFrame()
        fixed_feats_cache = None  # read once and reuse across folds

        for k, _fold_dir in enumerate(fold_dirs):
            # best params
            _best_path = _fold_dir.joinpath('bay_opt/best_params.json')
            if _best_path.exists():
                _opt_params = utils.read_from_json(_best_path)['best_params']
            else:
                _opt_params = {}  # handle edge case: skipped BayesOpt
            opt_params = pd.concat([opt_params, pd.DataFrame([_opt_params]).T], axis=1)

            # feature list per fold
            topk_path = _fold_dir.joinpath('top_k_feat_names.json')
            fixed_path = _fold_dir.joinpath('fixed_feat_names.json')
            if topk_path.exists():  # standard case: dict {feat: rank}
                _top_k_dict = utils.read_from_json(topk_path)
                feats = list(_top_k_dict.keys())
            elif fixed_path.exists():  # fixed-feature case: list of feats; read once
                if fixed_feats_cache is None:
                    fixed_feats_cache = utils.read_from_json(fixed_path)
                feats = fixed_feats_cache
            else:
                feats = []

            if feats:
                top_k_feats = pd.concat([top_k_feats, pd.DataFrame(index=feats, data={f'fold_{k + 1}': 1})],
                                        axis=1, sort=False)

        #  finalize summaries
        if not top_k_feats.empty:
            # only sum across fold_* cols
            fold_cols = [c for c in top_k_feats.columns if c.startswith('fold_')]
            top_k_feats['sum'] = top_k_feats[fold_cols].fillna(0).sum(axis=1)
            top_k_feats = top_k_feats.sort_values(by='sum', ascending=False).fillna(0)

        opt_params.columns = [f'fold_{i + 1}' for i in range(len(fold_dirs))]
        if not opt_params.empty:
            opt_params['mean'] = opt_params.mean(axis=1, numeric_only=True)
            opt_params['std'] = opt_params.std(axis=1, numeric_only=True)
            opt_params = opt_params.round(2)

        return opt_params, top_k_feats

    @staticmethod
    def _collect_scoring_history(fold_dirs: list[Path], early_stop_options: list,
                                 min_n_nights: int = None) -> (dict, dict, dict):
        """ Collects scoring history across folds for the new Evaluator output.
        For each fold and for each early-stopping option (i.e. "early_stopping" and "no_early_stopping"),
        it reads the following files (if available):
              - default_model_scores.json (at the root of each fold)
              - night_scores.json from the fixed subfolder (e.g. <fold_dir>/<es_key>/night/night_scores.json)
              - patient_scores.json from the fixed subfolder (e.g. <fold_dir>/<es_key>/patient_scores.json)
            It then aggregates these results across folds.
        Returns:
              scores: dict with keys "default_night", "opt_night", and "opt_patient".
              (curves and cms could be handled similarly if needed.)
            """
        # Convert boolean early_stop_options to string keys.
        es_keys = ['early_stopping' if es else 'no_early_stopping' for es in early_stop_options]
        dir_keys = es_keys + [f"{es}_min_{min_n_nights}_nights" for es in es_keys] if min_n_nights else es_keys

        # Initialize DataFrames/dicts.
        from_fold = []
        default_night_scores = pd.DataFrame()
        opt_night = {d: [] for d in dir_keys}
        opt_patient = {d: [] for d in dir_keys}

        curves = {lvl: {d: {'roc': {'fpr': [], 'tpr': [], 'auc': [], 'op_point': []},
                            'pr': {'rec': [], 'prec': [], 'f1_max': [], 'op_point_dist': [], 'op_point_f1': []}}
                        for d in dir_keys} for lvl in ['night', 'patient']}

        cms = {lvl: {d: [] for d in dir_keys} for lvl in ['night', 'patient']}

        for k, fold_dir in enumerate(fold_dirs):
            fold_name = f"fold_{k + 1}"
            # Read default model scores from the fold folder.
            try:
                default_scores = utils.read_from_json(fold_dir.joinpath("default_model_scores.json"))["valid"]
            except Exception as e:
                logger.warning(f"Could not read default_model_scores.json from {fold_dir}: {e}")
                continue

            from_fold.append(fold_dir.name)
            # Convert the dictionary of metrics into a one-row DataFrame.
            df_default = pd.DataFrame({k: [v] for k, v in default_scores.items()}, index=[fold_name])
            default_night_scores = pd.concat([default_night_scores, df_default], axis=0)

            for _dir in dir_keys:
                score_folder = fold_dir.joinpath(_dir)
                if not score_folder.exists():
                    logger.warning(f"Folder {score_folder} does not exist; skipping.")
                    continue

                for lvl in ['night', 'patient']:
                    score_file = score_folder.joinpath(
                        'night/night_scores.json' if lvl == 'night' else 'patient_scores.json')
                    if score_file.exists():
                        try:
                            scores = utils.read_from_json(score_file) if lvl == 'patient' \
                                else utils.read_from_json(score_file)['valid']
                            cm = scores.pop("cm", None)
                        except Exception as e:
                            logger.warning(f"Could not read {lvl}_scores.json from {score_file}: {e}")
                            continue

                        df_scores = pd.DataFrame({k: [v] for k, v in scores.items()}, index=[fold_name])
                        (opt_night if lvl == 'night' else opt_patient)[_dir].append(df_scores)
                        if cm is not None:
                            cms[lvl][_dir].append(cm)

                # collect roc and pr interpolated curves
                for curve_type in ['roc', 'pr']:

                    for _lvl in ['night', 'patient']:

                        curve_dict = curves[_lvl][_dir][curve_type]

                        curve_file = score_folder.joinpath(
                            f'night/night_interp_{curve_type}_curve.json' if _lvl == 'night'
                            else f'patient_interp_{curve_type}_curve.json')

                        if curve_file.exists():
                            curve_data = utils.read_from_json(curve_file)

                            if curve_type == 'roc':
                                if not curve_dict['fpr']:
                                    curve_dict['fpr'] = curve_data['fpr']
                                curve_dict['tpr'].append(curve_data['tpr'])
                                curve_dict['auc'].append(curve_data.get('auc', None))  # Handle missing 'auc'
                                curve_dict['op_point'].append(curve_data.get('op_point', None))

                            elif curve_type == 'pr':
                                if not curve_dict['rec']:
                                    curve_dict['rec'] = curve_data['rec']
                                curve_dict['prec'].append(curve_data['prec'])
                                curve_dict['f1_max'].append(curve_data.get('f1_max', None))  # Handle missing 'f1_max'
                                curve_dict['op_point_dist'].append(curve_data.get('op_point', {}).get('dist', None))
                                curve_dict['op_point_f1'].append(curve_data.get('op_point', {}).get('f1', None))

        # For each early-stopping option, concatenate the per-fold DataFrames.
        for _dir in dir_keys:
            if opt_night[_dir]:
                opt_night[_dir] = pd.concat(opt_night[_dir], axis=0)
            else:
                opt_night[_dir] = pd.DataFrame()
            if opt_patient[_dir]:
                opt_patient[_dir] = pd.concat(opt_patient[_dir], axis=0)
            else:
                opt_patient[_dir] = pd.DataFrame()

        scores = dict(default_night=default_night_scores, opt_night=opt_night, opt_patient=opt_patient)
        return scores, curves, cms, from_fold

    @staticmethod
    def _draw_history_boxplots(scores: dict[str, pd.DataFrame], include_night: bool):
        figs_night, figs_patient = {}, {}

        def _recurse_dict_(d, key_path='', figs_dict=None):
            for key, value in d.items():
                new_key_path = f"{key_path}_{key}" if key_path else key
                if isinstance(value, dict):
                    _recurse_dict_(value, new_key_path, figs_dict)
                elif isinstance(value, pd.DataFrame):
                    value = value.T
                    figs_dict[new_key_path] = vis.draw_cv_boxplot(
                        {metric: value.loc[metric, :].values for metric in value.index})
                    plt.close('all')

        if include_night:
            _recurse_dict_(scores['opt_night'], figs_dict=figs_night)
        _recurse_dict_(scores['opt_patient'], figs_dict=figs_patient)
        return figs_night, figs_patient

    @staticmethod
    def _draw_mean_curves(curves: dict, include_night: bool, lodo_lbls: dict):

        figs_night, figs_patient = {}, {}

        def _recurse_dict_(d, key_path='', figs_dict=None):
            for key, value in d.items():
                new_key_path = f"{key_path}_{key}" if key_path else key
                first_value = list(value.values())[0]
                if isinstance(first_value, dict):
                    _recurse_dict_(value, new_key_path, figs_dict)
                elif isinstance(first_value, list):
                    if 'roc' in new_key_path:
                        figs_dict[new_key_path] = vis.draw_cv_roc_or_pr_curve(
                            value, 'roc', lodo_labels=lodo_lbls)
                    elif 'pr' in new_key_path:
                        figs_dict[new_key_path] = vis.draw_cv_roc_or_pr_curve(
                            value, 'pr', lodo_labels=lodo_lbls)

        if include_night:
            _recurse_dict_(curves.get('night'), figs_dict=figs_night)
        _recurse_dict_(curves.get('patient'), figs_dict=figs_patient)

        return figs_night, figs_patient

    @contextmanager
    def _handle_parallel_logging(self):
        self.is_parallel = self.n_jobs != 1
        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict
                   if 'actitect' in name or '__main__' in name]

        previous_levels = {_lggr: logger.level for _lggr in loggers}

        if self.is_parallel:
            for _logger in loggers:
                _logger.setLevel(logging.WARNING)
            self.config.model.bayes_params.verbose = False
        else:
            self.config.model.bayes_params.verbose = True

        try:
            yield
        finally:
            if self.is_parallel:
                for _logger, level in previous_levels.items():
                    _logger.setLevel(level)

    @staticmethod
    @contextmanager
    def _manifest_guard(save_dir: Path, *, flavour: str, repeats: int, outer_folds_expected: int, inner_folds: int,
                        seeds: dict, dataset_summary: dict):
        """ Create/maintain *manifest.json* that tracks the progress of a Nested-CV run."""
        manifest_path = save_dir.joinpath('manifest.json')

        def __utc_now() -> str:
            return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        def __atomic_write(path: Path, payload: dict):
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(path)

        manifest = dict(run_id=str(uuid.uuid4()), flavour=flavour, started_at=__utc_now(), finished_at=None,
                        dataset_summary=dataset_summary, outer_folds_expected=outer_folds_expected,
                        inner_folds=inner_folds, repeats=repeats, seeds=seeds)
        __atomic_write(manifest_path, manifest)

        try:
            yield manifest  # give caller a mutable reference
        except Exception:
            raise  # manifest already indicates unfinished status
        else:
            manifest['finished_at'] = __utc_now()
            __atomic_write(manifest_path, manifest)

    def _restore_cv_state_from_manifest(self):
        """Restore key state variables for eval mode by reading manifest.json."""
        _is_lodo = isinstance(self, LODONestedCV)
        _is_predefined = isinstance(self, PredefinedKFoldNestedCV)
        _cv_tag = 'LODO' if _is_lodo else 'PredefinedKFold' if _is_predefined else 'KFold'

        # Try direct load: save_path points to a CV subdir
        manifest_path = self.save_path / 'manifest.json'
        if manifest_path.exists():
            manifest = utils.read_from_json(manifest_path)
            cv_subdir = self.save_path
        else:
            # Fallback: save_path is the run directory, search for matching CV subdir
            candidates = list(self.save_path.glob(f"{_cv_tag}*/manifest.json"))
            if not candidates:
                raise FileNotFoundError(
                    f"No manifest.json found matching '{_cv_tag}*/manifest.json' in {self.save_path}")
            if len(candidates) > 1:
                raise RuntimeError(f"Multiple manifest.json files found for {_cv_tag}: {candidates}")
            manifest = utils.read_from_json(candidates[0])
            cv_subdir = candidates[0].parent

        self.n_repeats = manifest['repeats']
        self.outer_splits = manifest['outer_folds_expected']
        self.n_datasets = len(manifest.get('dataset_summary', {}))
        self.fold_2_dataset = manifest.get('fold_to_dataset', {}) if _is_lodo else None

        return cv_subdir

    @staticmethod
    def _count_unique_datasets(data: FeatureSet):
        return len(np.unique(data.dataset)) if data.dataset is not None else 1


class KFoldNestedCV(NestedCVBase):

    def _get_outer_splitter(self, seed: int):
        """Stratified group-aware k-kfold splitter."""
        return StratifiedGroupKFold(**self.config.nested_cv.outer_cv.dict(), random_state=seed)

    def _get_outer_groups(self, data: FeatureSet):
        """Group by patient ID."""
        return data.group


class LODONestedCV(NestedCVBase):

    def _get_outer_splitter(self, seed: int):
        """Leave one dataset out splitting."""
        return LeaveOneGroupOut()  # deterministic, no seed needed

    def _get_outer_groups(self, data: FeatureSet):
        """Group by datasets."""
        return data.dataset

    def _summarize_lodo_performance(self, scores: dict, from_fold: list[str], output_dir: Path):
        """Summarize LODO patient-level metrics into JSON under ./summaries/.
        Also inlines (if available) the per-fold patient diagnostics as:
          { ..., "diagnostic_metrics": { <DATASET>: { ...diag... }, ... } }.
        """
        assert isinstance(self, LODONestedCV), "This function is only valid for LODO evaluation."
        assert hasattr(self, 'fold_2_dataset'), "fold_2_dataset mapping not found."

        summaries_dir = (output_dir / "summaries").resolve()
        summaries_dir.mkdir(parents=True, exist_ok=True)

        # Iterate over all patient-level result variants present in `scores`
        for variant_key, metrics_df in scores.get('opt_patient', {}).items():
            if metrics_df is None or metrics_df.empty:
                continue

            metric_summary = {}
            diag_summary = {}  # will be merged under "diagnostic_metrics"

            # Map each fold's row to the held-out dataset using the manifest mapping
            for fold_name, row in zip(from_fold, metrics_df.itertuples(index=False)):
                fold_idx = fold_name.replace("outer_fold_", "")
                dataset_id = self.fold_2_dataset.get(f"seed{self.random_state}_fold{fold_idx}", "UNKNOWN")

                # 1) regular metrics
                for metric, value in zip(metrics_df.columns, row):
                    metric_summary.setdefault(metric, {})[dataset_id] = value

                # 2) try to inline diagnostics for this fold+variant (if present)
                #    expected path: repeats/**/<fold_name>/<variant_key>/patient_prob_diagnostic.json
                diag_file = None
                for p in output_dir.rglob(f"repeats/**/{fold_name}/{variant_key}/patient_prob_diagnostic.json"):
                    diag_file = p
                    break
                if diag_file and diag_file.exists():
                    try:
                        diag = utils.read_from_json(diag_file)
                        diag_summary[dataset_id] = diag
                    except Exception as e:
                        logger.warning(f"Could not read diagnostic metrics from {diag_file}: {e}")

            # merge diagnostics inside the same JSON:
            if diag_summary:
                metric_summary["diagnostic_metrics"] = diag_summary

            out_path = summaries_dir / f"lodo_eval_summary_{variant_key}.json"
            with open(out_path, "w") as f:
                json.dump(metric_summary, f, indent=2)


class _PredefinedOuterSplitter:
    """Minimal sklearn-compatible splitter backed by subject-level outer assignments."""

    def __init__(self, assignments: pd.DataFrame, fold_labels: list[int]):
        self.assignments = assignments.copy()
        self.fold_labels = list(fold_labels)

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.fold_labels)

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("Predefined outer splitting requires subject IDs via groups.")

        groups = np.asarray(groups).astype(str)
        available_subjects = set(groups)
        for fold_label in self.fold_labels:
            fold_df = self.assignments[self.assignments['outer_fold'] == fold_label]
            train_subjects = set(fold_df.loc[fold_df['outer_role'] == 'train', 'subject_id'])
            test_subjects = set(fold_df.loc[fold_df['outer_role'] == 'test', 'subject_id'])

            missing = available_subjects.difference(train_subjects | test_subjects)
            if missing:
                raise ValueError(
                    f"Outer fold {fold_label}: {len(missing)} loaded subjects have no predefined assignment; "
                    f"examples: {sorted(missing)[:5]}"
                )

            train_indices = np.flatnonzero(np.isin(groups, list(train_subjects)))
            test_indices = np.flatnonzero(np.isin(groups, list(test_subjects)))
            if train_indices.size == 0 or test_indices.size == 0:
                raise ValueError(
                    f"Outer fold {fold_label}: predefined train and test partitions must both be non-empty.")
            yield train_indices, test_indices


class PredefinedKFoldNestedCV(NestedCVBase):
    """Nested CV using predefined subject assignments for outer and inner folds.

    ``splits`` must use long format with columns ``subject_id``, ``repeat``,
    ``outer_fold``, ``inner_fold`` and ``role``. Roles must be ``train``,
    ``validation`` or ``test``. Subject IDs must match ``FeatureSet.group`` exactly.
    """

    REQUIRED_SPLIT_COLUMNS = {'subject_id', 'repeat', 'outer_fold', 'inner_fold', 'role'}
    VALID_ROLES = {'train', 'validation', 'test'}

    def __init__(self, config: PipelineConfig, save_path: Path, splits: pd.DataFrame,
                 early_stopping_options: list = None, calibration: str = None):
        super().__init__(config, save_path, early_stopping_options=early_stopping_options, calibration=calibration)
        self.split_assignments = self._prepare_split_assignments(splits)
        self.repeat_labels = sorted(self.split_assignments['repeat'].unique().tolist())
        self.outer_fold_labels = sorted(self.split_assignments['outer_fold'].unique().tolist())
        self.inner_fold_labels = sorted(self.split_assignments['inner_fold'].unique().tolist())
        self._validate_split_structure()
        self._outer_assignments = self._build_outer_assignments()

    @classmethod
    def _prepare_split_assignments(cls, splits: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(splits, pd.DataFrame):
            raise TypeError(f"splits must be a pandas DataFrame, got {type(splits).__name__}.")

        missing_columns = cls.REQUIRED_SPLIT_COLUMNS.difference(splits.columns)
        if missing_columns:
            raise ValueError(f"Predefined splits are missing columns: {sorted(missing_columns)}")

        prepared = splits.loc[:, sorted(cls.REQUIRED_SPLIT_COLUMNS)].copy()
        if prepared.empty:
            raise ValueError("Predefined splits must not be empty.")
        if prepared.isna().any().any():
            null_columns = prepared.columns[prepared.isna().any()].tolist()
            raise ValueError(f"Predefined splits contain missing values in columns: {null_columns}")

        prepared['subject_id'] = prepared['subject_id'].astype(str).str.strip()
        prepared['role'] = prepared['role'].astype(str).str.strip().str.lower()
        if (prepared['subject_id'] == '').any():
            raise ValueError("Predefined splits contain empty subject IDs.")

        for column in ['repeat', 'outer_fold', 'inner_fold']:
            numeric = pd.to_numeric(prepared[column], errors='raise')
            if not np.all(np.equal(numeric, np.floor(numeric))):
                raise ValueError(f"Predefined split column '{column}' must contain integer values.")
            prepared[column] = numeric.astype(int)
            if (prepared[column] < 0).any():
                raise ValueError(f"Predefined split column '{column}' must be non-negative.")

        invalid_roles = set(prepared['role']).difference(cls.VALID_ROLES)
        if invalid_roles:
            raise ValueError(f"Unsupported predefined split roles: {sorted(invalid_roles)}")

        duplicate_columns = ['subject_id', 'repeat', 'outer_fold', 'inner_fold']
        duplicates = prepared.duplicated(duplicate_columns, keep=False)
        if duplicates.any():
            examples = prepared.loc[duplicates, duplicate_columns].head().to_dict('records')
            raise ValueError(
                "Predefined splits contain duplicate subject/repeat/outer_fold/inner_fold assignments; "
                f"examples: {examples}")

        return prepared.sort_values(['repeat', 'outer_fold', 'inner_fold', 'subject_id']).reset_index(drop=True)

    def _validate_split_structure(self):
        expected_repeats = list(range(len(self.repeat_labels)))
        expected_outer = list(range(len(self.outer_fold_labels)))
        expected_inner = list(range(len(self.inner_fold_labels)))
        if self.repeat_labels != expected_repeats:
            raise ValueError(f"Predefined repeat labels must be contiguous and zero-based, got {self.repeat_labels}.")
        if self.outer_fold_labels != expected_outer:
            raise ValueError(
                f"Predefined outer-fold labels must be contiguous and zero-based, got {self.outer_fold_labels}.")
        if self.inner_fold_labels != expected_inner:
            raise ValueError(
                f"Predefined inner-fold labels must be contiguous and zero-based, got {self.inner_fold_labels}.")
        if len(self.inner_fold_labels) != self.config.nested_cv.inner_cv.n_splits:
            raise ValueError(
                f"Predefined splits define {len(self.inner_fold_labels)} inner folds, but pipeline config expects "
                f"{self.config.nested_cv.inner_cv.n_splits}.")

        all_subjects = set(self.split_assignments['subject_id'])
        expected_combinations = {
            (repeat, outer_fold, inner_fold)
            for repeat in self.repeat_labels
            for outer_fold in self.outer_fold_labels
            for inner_fold in self.inner_fold_labels
        }
        actual_combinations = set(map(tuple, self.split_assignments[
            ['repeat', 'outer_fold', 'inner_fold']].drop_duplicates().to_numpy()))
        if actual_combinations != expected_combinations:
            missing = sorted(expected_combinations.difference(actual_combinations))
            extra = sorted(actual_combinations.difference(expected_combinations))
            raise ValueError(f"Incomplete predefined split grid. Missing={missing[:5]}, extra={extra[:5]}.")

        for (repeat, outer_fold, inner_fold), fold_df in self.split_assignments.groupby(
                ['repeat', 'outer_fold', 'inner_fold'], sort=True):
            fold_subjects = set(fold_df['subject_id'])
            if fold_subjects != all_subjects:
                missing = sorted(all_subjects.difference(fold_subjects))
                extra = sorted(fold_subjects.difference(all_subjects))
                raise ValueError(
                    f"repeat={repeat}, outer_fold={outer_fold}, inner_fold={inner_fold} does not assign every "
                    f"configured subject exactly once. Missing={missing[:5]}, extra={extra[:5]}.")

            role_sets = {role: set(fold_df.loc[fold_df['role'] == role, 'subject_id']) for role in self.VALID_ROLES}
            if not role_sets['train'] or not role_sets['validation'] or not role_sets['test']:
                raise ValueError(
                    f"repeat={repeat}, outer_fold={outer_fold}, inner_fold={inner_fold} must contain non-empty "
                    "train, validation and test partitions.")
            if role_sets['train'] & role_sets['validation'] or role_sets['train'] & role_sets['test'] or \
                    role_sets['validation'] & role_sets['test']:
                raise ValueError(
                    f"repeat={repeat}, outer_fold={outer_fold}, inner_fold={inner_fold} contains overlapping roles.")

        for (repeat, outer_fold), outer_df in self.split_assignments.groupby(['repeat', 'outer_fold'], sort=True):
            test_sets = [
                set(fold_df.loc[fold_df['role'] == 'test', 'subject_id'])
                for _, fold_df in outer_df.groupby('inner_fold', sort=True)
            ]
            if any(test_set != test_sets[0] for test_set in test_sets[1:]):
                raise ValueError(
                    f"Outer-test subjects vary across inner folds for repeat={repeat}, outer_fold={outer_fold}.")

        for repeat, repeat_df in self.split_assignments.groupby('repeat', sort=True):
            outer_test_sets = []
            for _, outer_df in repeat_df.groupby('outer_fold', sort=True):
                first_inner = self.inner_fold_labels[0]
                fold_df = outer_df[outer_df['inner_fold'] == first_inner]
                outer_test_sets.append(set(fold_df.loc[fold_df['role'] == 'test', 'subject_id']))
            if set().union(*outer_test_sets) != all_subjects:
                missing = sorted(all_subjects.difference(set().union(*outer_test_sets)))
                raise ValueError(f"Repeat {repeat}: outer-test folds do not cover all subjects; missing={missing[:5]}.")
            for i, test_i in enumerate(outer_test_sets):
                for j, test_j in enumerate(outer_test_sets[i + 1:], start=i + 1):
                    overlap = test_i & test_j
                    if overlap:
                        raise ValueError(
                            f"Repeat {repeat}: outer test folds {i} and {j} overlap; examples={sorted(overlap)[:5]}.")

    def _build_outer_assignments(self) -> dict[int, pd.DataFrame]:
        assignments = {}
        for repeat in self.repeat_labels:
            repeat_df = self.split_assignments[self.split_assignments['repeat'] == repeat]
            rows = []
            for (outer_fold, subject_id), subject_df in repeat_df.groupby(['outer_fold', 'subject_id'], sort=True):
                roles = set(subject_df['role'])
                if roles == {'test'}:
                    outer_role = 'test'
                elif 'test' not in roles and roles.issubset({'train', 'validation'}):
                    outer_role = 'train'
                else:
                    raise ValueError(
                        f"Inconsistent outer assignment for repeat={repeat}, outer_fold={outer_fold}, "
                        f"subject={subject_id}: roles={sorted(roles)}")
                rows.append({'subject_id': subject_id, 'outer_fold': outer_fold, 'outer_role': outer_role})
            assignments[repeat] = pd.DataFrame(rows)
        return assignments

    def _validate_data(self, data: FeatureSet):
        if data.group is None:
            raise ValueError("Predefined nested CV requires subject IDs in FeatureSet.group.")
        groups = np.asarray(data.group).astype(str)
        if groups.ndim != 1 or len(groups) != len(data.y):
            raise ValueError("FeatureSet.group must be one-dimensional and aligned with FeatureSet.y.")

        loaded_subjects = set(groups)
        configured_subjects = set(self.split_assignments['subject_id'])
        loaded_but_unconfigured = loaded_subjects.difference(configured_subjects)
        configured_but_not_loaded = configured_subjects.difference(loaded_subjects)
        if loaded_but_unconfigured or configured_but_not_loaded:
            raise ValueError(
                "Loaded subjects and predefined split subjects do not match exactly. "
                f"Loaded but unconfigured: {len(loaded_but_unconfigured)} "
                f"(examples: {sorted(loaded_but_unconfigured)[:5]}); configured but not loaded: "
                f"{len(configured_but_not_loaded)} (examples: {sorted(configured_but_not_loaded)[:5]}).")

        y = np.asarray(data.y)
        inconsistent_labels = [subject for subject in loaded_subjects if np.unique(y[groups == subject]).size != 1]
        if inconsistent_labels:
            raise ValueError(
                f"Each subject must have exactly one label; inconsistent subjects: {sorted(inconsistent_labels)[:5]}")

        if data.dataset is not None:
            datasets = np.asarray(data.dataset).astype(str)
            if datasets.ndim != 1 or len(datasets) != len(data.y):
                raise ValueError("FeatureSet.dataset must be one-dimensional and aligned with FeatureSet.y.")
            inconsistent_datasets = [
                subject for subject in loaded_subjects if np.unique(datasets[groups == subject]).size != 1
            ]
            if inconsistent_datasets:
                raise ValueError(
                    "Each subject must belong to exactly one dataset; inconsistent subjects: "
                    f"{sorted(inconsistent_datasets)[:5]}")

        for repeat in self.repeat_labels:
            outer_splitter = _PredefinedOuterSplitter(self._outer_assignments[repeat], self.outer_fold_labels)
            outer_splits = list(outer_splitter.split(data.x, data.y, groups))
            if len(outer_splits) != len(self.outer_fold_labels):
                raise ValueError(f"Repeat {repeat}: expected {len(self.outer_fold_labels)} outer folds.")
            for outer_fold, (train_idx, test_idx) in zip(self.outer_fold_labels, outer_splits):
                train_subjects = set(groups[train_idx])
                test_subjects = set(groups[test_idx])
                if train_subjects & test_subjects:
                    raise ValueError(f"Repeat {repeat}, outer fold {outer_fold}: train/test subjects overlap.")
                if train_subjects | test_subjects != loaded_subjects:
                    raise ValueError(
                        f"Repeat {repeat}, outer fold {outer_fold}: train/test subjects do not cover data.")

        logger.info(
            f"Validated predefined nested-CV splits for {len(loaded_subjects)} subjects, "
            f"{len(self.repeat_labels)} repeats, {len(self.outer_fold_labels)} outer folds and "
            f"{len(self.inner_fold_labels)} inner folds.")

    def _get_outer_splitter(self, seed: int):
        repeat = seed - self.random_state
        if repeat not in self._outer_assignments:
            raise ValueError(f"No predefined assignments found for repeat {repeat}.")
        return _PredefinedOuterSplitter(self._outer_assignments[repeat], self.outer_fold_labels)

    def _get_outer_groups(self, data: FeatureSet):
        return data.group

    def _get_inner_splits(self, train_outer: Fold, *, repeat: int, outer_fold: int):
        groups = np.asarray(train_outer.group).astype(str)
        available_subjects = set(groups)
        split_df = self.split_assignments[
            (self.split_assignments['repeat'] == repeat) &
            (self.split_assignments['outer_fold'] == outer_fold)
            ]
        if split_df.empty:
            raise ValueError(f"No predefined inner assignments for repeat={repeat}, outer_fold={outer_fold}.")

        cv_splits = []
        for inner_fold in self.inner_fold_labels:
            fold_df = split_df[split_df['inner_fold'] == inner_fold]
            train_subjects = set(fold_df.loc[fold_df['role'] == 'train', 'subject_id'])
            validation_subjects = set(fold_df.loc[fold_df['role'] == 'validation', 'subject_id'])
            test_subjects = set(fold_df.loc[fold_df['role'] == 'test', 'subject_id'])

            unexpected_test = available_subjects.intersection(test_subjects)
            if unexpected_test:
                raise ValueError(
                    f"repeat={repeat}, outer_fold={outer_fold}, inner_fold={inner_fold}: outer-test subjects are "
                    f"present in the outer-training fold; examples: {sorted(unexpected_test)[:5]}")

            assigned_outer_train = train_subjects | validation_subjects
            missing = available_subjects.difference(assigned_outer_train)
            extra = assigned_outer_train.difference(available_subjects)
            if missing or extra:
                raise ValueError(
                    f"repeat={repeat}, outer_fold={outer_fold}, inner_fold={inner_fold}: inner assignments do not "
                    f"match outer-training subjects. Missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}.")
            if train_subjects & validation_subjects:
                raise ValueError(
                    f"repeat={repeat}, outer_fold={outer_fold}, inner_fold={inner_fold}: train/validation overlap.")

            train_indices = np.flatnonzero(np.isin(groups, list(train_subjects)))
            validation_indices = np.flatnonzero(np.isin(groups, list(validation_subjects)))
            if train_indices.size == 0 or validation_indices.size == 0:
                raise ValueError(
                    f"repeat={repeat}, outer_fold={outer_fold}, inner_fold={inner_fold}: predefined train and "
                    "validation partitions must both be non-empty.")
            cv_splits.append((train_indices, validation_indices))

        return cv_splits
