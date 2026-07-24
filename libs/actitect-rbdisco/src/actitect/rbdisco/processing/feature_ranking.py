import gc
import logging
import sys
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from boruta import BorutaPy
from joblib import Parallel, delayed
from mrmr import mrmr_classif
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

from actitect import utils
from actitect.config import DataConfig, FeatureSelectionConfig
from ..core.types import FeatureSet
from ..models import ModelFactory

logger = logging.getLogger(__name__)

__all__ = ['FeatureRanker']


class FeatureRanker:

    def __init__(self, root_dir: Path, data_config: DataConfig, n_jobs: int,
                 draw_plots: bool = False, random_state: int = 42, legacy_mode: bool = False,
                 feature_selection_config: Optional[FeatureSelectionConfig] = None):
        """Class to run different feature ranking algorithms and produce an ensemble ranking to identify the best
        features.
        Parameters:
            :param data_config: (DataConfig) instance containing meta-info of the data.
            :param root_dir: (Path) defining the root directory where ranking files are stored.
            :param n_jobs: (int) to handle parallel-processing.
            :param draw_plots: (bool. default False) whether to draw a summary plot or not.
            :param legacy_mode: (bool. default False) legacy mode to restore old versions, currently only relevant
                for correlation based rankings after speed implementation (should be only O(e-16) difference though.)"""

        self.data = None  # populate in entry point .fetch_or_compute()
        self.data_config = data_config
        self.root_dir = root_dir
        self.out_dir = None  # set dynamically in .fetch_or_compute() depending on cv or non-cv mode.
        self.n_jobs = n_jobs
        self.draw_plots = draw_plots
        self.random_state = random_state
        self.legacy_mode = legacy_mode
        self.feature_selection_config = feature_selection_config

        self.ranking_file_postfix = f"combined_rankings_{self.data_config.agg_level}.csv"
        self.method_dispatch = {
            'mrmr': self._run_mrmr,
            'boruta': self._run_boruta,
            'corr_label': lambda: self._eval_corr_wrapper()[0],
            'corr_inter': lambda: self._eval_corr_wrapper()[1],
            'stat_sign': self._eval_stat_sign_groups,
            'var': self._eval_variance
        }

    def fetch_or_compute(self, data: FeatureSet, return_df: bool = False) -> pd.DataFrame:
        """ Fetches or computes feature rankings for either a fold or the full dataset.
        Parameters:
            :param data: (FeatureSet) containing the data. If not data.from_fold, full dataset will be processed,
                else fold-level rankings.
            :param return_df: Boolean flag to return a DataFrame instead of a dictionary.
        Returns:
            :return: Feature ranking as a DataFrame or dictionary. """
        self.data = data
        assert isinstance(self.data, FeatureSet), f"argument 'data' must be of type 'FeatureSet' ({type(FeatureSet)})"
        _from_fold = getattr(self.data, 'from_fold', None)
        if isinstance(_from_fold, dict):  # data is on fold-level
            self.out_dir = self.root_dir.joinpath(f"fold_{_from_fold['k']}/{_from_fold['name']}")
            log_message_prefix = f"fold {_from_fold['k']}"
        else:  # data contains full dataset
            self.out_dir = self.root_dir
            log_message_prefix = "full training set"

        ranking_file = self.out_dir.joinpath(self.ranking_file_postfix)
        if ranking_file.exists():
            logger.info(f"using pre-computed ranking for {self.root_dir.stem} at {log_message_prefix}.")
            rank_df = pd.read_csv(ranking_file).set_index('total_rank')
        else:
            logger.warning(f"no pre-computed ranking file found for '{self.root_dir}' at {log_message_prefix},"
                           f"computation may take a while.")
            self.out_dir = utils.check_make_dir(self.out_dir, use_existing=True)
            rank_df = self._run_ensemble()

        if return_df:
            rank_mapping_df = rank_df.reset_index()[['name', 'total_rank']]
            rank_mapping_df['idx'] = rank_mapping_df['name'].map({name: idx for idx, name in enumerate(data.feat_map)})
            rank_mapping_df = rank_mapping_df.set_index('idx').sort_index()
            return rank_mapping_df
        else:
            _feature_ranking = rank_df.name.values  # df is sorted by rank
            return {  # maps each feat. name to its idx in x and its rank in _feature_ranking
                name: {'idx': idx, 'rank': np.where(_feature_ranking == name)[0][0] + 1}
                for idx, name in enumerate(self.data.feat_map)}

    def _run_ensemble(self, ranking_methods: List[Tuple] = None):
        """ Run the ensemble ranker to run the desired sub-functions and produce a wighted ensemble ranking.
        Parameters:
            :param ranking_methods: (list(tuple)) defining which ranking methods to use and which weight is assigned to
            each. Available options are 'mrmr', 'boruta', 'corr_label', 'corr_inter', 'stat_sign', 'var'.
        Returns:
            :return: (pd.DataFrame) Containing ranks of each feature selection segment and a combined, weighted rank."""

        if ranking_methods is None:
            ranking_methods = [
                ('mrmr', 1.0),
                ('corr_label', 1.0),
                ('corr_inter', 0.2),
                ('stat_sign', 1.0),
                ('var', 1.0),
            ]

            if self.use_boruta:
                ranking_methods.insert(1, ('boruta', 1.0))
            else:
                logger.warning('Boruta disabled through model.feature_selection.use_boruta.')
        logger.info('Active feature-ranking methods: %s', [method for method, _ in ranking_methods])

        rank_df = pd.DataFrame({})
        _weight_arr = []
        for _method_key, _method_weight in ranking_methods:
            if _method_key not in self.method_dispatch:
                raise ValueError(f"Unknown ranking method: {_method_key}."
                                 f"Choose from {sorted(list(self.method_dispatch.keys()))}")

            _method_rank_df = self.method_dispatch[_method_key]().get(f'{self.data_config.agg_level}').reset_index()
            rank_df[f"{_method_key}(x{_method_weight})"] = _method_rank_df[
                ['name', 'rank']].sort_values(by='name').set_index('name')
            _weight_arr.append(_method_weight)

        # aggregate rankings and produce ensemble ranking (NaN-aware)
        method_cols = [c for c in rank_df.columns]  # currently only method columns at this point
        values = rank_df[method_cols].to_numpy(dtype=float)
        weights = np.array(_weight_arr, dtype=float)

        mask = ~np.isnan(values)

        # mean rank ignoring NaNs
        mean_rank = np.nanmean(values, axis=1)

        # weighted mean ignoring NaNs, with per-row weight renormalization
        weighted_sum = np.nansum(values * weights, axis=1)
        weight_den = np.sum(mask * weights, axis=1)
        weighted_mean = weighted_sum / np.where(weight_den == 0, np.nan, weight_den)

        rank_df['mean_rank'] = mean_rank
        rank_df['weighted_mean'] = weighted_mean
        rank_df['diff_weight_mean'] = rank_df['weighted_mean'] - rank_df['mean_rank']

        # rankdata cannot handle NaN well: push NaNs to bottom by temporary fill with +inf
        _tmp = rank_df['weighted_mean'].to_numpy()
        _tmp = np.where(np.isnan(_tmp), np.inf, _tmp)
        rank_df['total_rank'] = stats.rankdata(_tmp, method='max')
        rank_df = rank_df.reset_index().sort_values(by='total_rank').set_index('total_rank')
        rank_df.to_csv(self.out_dir.joinpath(self.ranking_file_postfix))

        if self.draw_plots:
            fig = self._plot_feature_selection_summary(rank_df)
            fig.savefig(self.out_dir.joinpath(self.ranking_file_postfix.replace('.csv', '.png')), bbox_inches='tight')
            del fig
        del _weight_arr
        gc.collect()
        return rank_df

    def _run_boruta(self, proxy_model: str = 'rf'):
        def __run_boruta_iteration(_data: FeatureSet, _model, iteration_seed: int):
            boruta_selector = BorutaPy(
                verbose=0,
                estimator=_model,
                n_estimators=settings['n_estimators'],
                perc=100,
                alpha=0.05,
                max_iter=settings['max_iter'],
                early_stopping=True,
                n_iter_no_change=settings['n_iter_no_change'],
                random_state=iteration_seed,
            )
            boruta_selector.fit(_data.x, _data.y)
            return boruta_selector.ranking_, boruta_selector.importance_history_

        with utils.Timer() as timer:
            out_dir = utils.check_make_dir(self.out_dir / 'boruta', True)
            settings = self._get_boruta_settings()

            n_hc_train, n_rbd_train = np.unique(self.data.y, return_counts=True)[1]
            model_factory = ModelFactory(
                proxy_model,
                cls_balance=n_hc_train / n_rbd_train,
                seed=self.random_state,
                top_k_cfg=None,
            )
            model_setup = model_factory.build()
            model = model_setup.model().set_params(**model_setup.default_params)
            model.set_params(max_depth=settings['max_depth'])

            if settings['rf_n_jobs'] is not None:
                model.set_params(n_jobs=settings['rf_n_jobs'])

            if settings['max_samples'] is not None:
                assert model.get_params()['bootstrap'], 'RandomForest max_samples requires bootstrap=True.'

                model.set_params(max_samples=settings['max_samples'])

            n_iterations = settings['n_iterations']
            n_parallel_jobs = min(n_iterations, self.n_jobs) if self.n_jobs > 0 else n_iterations

            logger.info(
                'Boruta setup: samples=%d, speedup=%s, threshold=%s, repetitions=%d, trees=%s, max_depth=%s, '
                'max_samples=%s, max_iter=%d, patience=%d, outer_jobs=%d, rf_jobs=%s',
                len(self.data.y),
                settings['speedup_applied'],
                settings['sample_threshold'],
                n_iterations,
                settings['n_estimators'],
                settings['max_depth'],
                settings['max_samples'],
                settings['max_iter'],
                settings['n_iter_no_change'],
                n_parallel_jobs,
                model.get_params().get('n_jobs'),
            )

            boruta_history = []
            boruta_ranks = np.zeros((n_iterations, len(self.data.feat_map)))

            with Parallel(n_jobs=n_parallel_jobs) as boruta_executor:
                results = boruta_executor(
                    delayed(__run_boruta_iteration)(self.data, model, self.random_state + iteration)
                    for iteration in range(n_iterations)
                )

            for iteration, (ranking, importance_history) in enumerate(results):
                boruta_ranks[iteration] = ranking
                boruta_history.append(importance_history)

            boruta_rank_mean = np.mean(boruta_ranks, axis=0).astype(int)
            boruta_rank_std = np.std(boruta_ranks, axis=0).astype(int)

            mean_importance_per_iteration = np.asarray([np.nanmean(history, axis=0) for history in boruta_history])
            importance_mean = np.mean(mean_importance_per_iteration, axis=0)
            importance_std = np.std(mean_importance_per_iteration, axis=0)
            rank = stats.rankdata(-importance_mean, method='max')

            ranking = pd.DataFrame({
                'name': self.data.feat_map,
                'rank': rank,
                ('rel_importance', 'mean'): importance_mean * 100,
                ('rel_importance', 'std'): importance_std * 100,
                ('rel_importance', 'std/mean'): importance_std / importance_mean,
                ('boruta_rank', 'mean'): boruta_rank_mean,
                ('boruta_rank', 'std'): boruta_rank_std,
                ('boruta_rank', 'std/mean'): boruta_rank_std / boruta_rank_mean,
            })
            ranking.columns = pd.MultiIndex.from_tuples([
                ('name', ''),
                ('rank', ''),
                ('rel_importance', 'mean'),
                ('rel_importance', 'std'),
                ('rel_importance', 'std/mean'),
                ('boruta_rank', 'mean'),
                ('boruta_rank', 'std'),
                ('boruta_rank', 'std/mean'),
            ])
            ranking = ranking.sort_values(by='rank').set_index('rank')
            ranking.to_csv(out_dir / f'boruta_{self.data_config.agg_level}.csv')

            settings_json = {
                'proxy_model': proxy_model,
                'platform': sys.platform,
                'random_state': self.random_state,
                'n_jobs': self.n_jobs,
                'n_parallel_jobs': n_parallel_jobs,
                'n_samples': len(self.data.y),
                **settings,
                'elapsed(s)': timer(),
            }
            utils.dump_to_json(settings_json, out_dir / 'setting.json')

            logger.info('Applied Boruta. (%.2fs)', timer())

            del rank, boruta_rank_mean, boruta_rank_std, importance_mean, importance_std
            gc.collect()

        return {self.data_config.agg_level: ranking}

    @property
    def use_boruta(self) -> bool:
        return True if self.feature_selection_config is None else self.feature_selection_config.use_boruta

    def _get_boruta_settings(self) -> dict:
        """ Resolve the effective Boruta settings for the current dataset."""
        defaults = {
            'n_iterations': 10,
            'n_estimators': 'auto',
            'max_depth': None,
            'max_samples': None,
            'max_iter': 1_000,
            'n_iter_no_change': 10,
            'rf_n_jobs': None,

        }

        if self.feature_selection_config is None:
            return {**defaults, 'speedup_applied': False, 'sample_threshold': None}

        cfg = self.feature_selection_config
        threshold = cfg.boruta_speedup_sample_threshold
        speedup_applied = threshold is not None and len(self.data.y) >= threshold

        if not speedup_applied:
            return {**defaults, 'speedup_applied': False, 'sample_threshold': threshold}

        settings = {
            'n_iterations': cfg.boruta_speedup_n_iterations or defaults['n_iterations'],
            'n_estimators': cfg.boruta_speedup_n_estimators or defaults['n_estimators'],
            'max_depth': cfg.boruta_speedup_max_depth,
            'max_samples': cfg.boruta_speedup_max_samples,
            'max_iter': cfg.boruta_speedup_max_iter or defaults['max_iter'],
            'n_iter_no_change': cfg.boruta_speedup_n_iter_no_change or defaults['n_iter_no_change'],
            'rf_n_jobs': cfg.boruta_speedup_rf_n_jobs,
            'speedup_applied': True,
            'sample_threshold': threshold,
        }

        assert settings['n_iterations'] > 0, 'Boruta n_iterations must be positive.'
        assert settings['max_iter'] > 1, 'Boruta max_iter must be greater than one.'
        assert settings['n_iter_no_change'] > 0, 'Boruta n_iter_no_change must be positive.'

        if settings['n_estimators'] != 'auto':
            assert settings['n_estimators'] > 0, 'Boruta n_estimators must be positive.'

        if settings['max_depth'] is not None:
            assert settings['max_depth'] > 0, 'Boruta max_depth must be positive.'

        if settings['max_samples'] is not None:
            assert 0 < settings['max_samples'] <= 1, 'Boruta max_samples must be in (0, 1].'

        if settings['rf_n_jobs'] is not None:
            assert settings['rf_n_jobs'] != 0, 'Boruta RF n_jobs must not be zero.'

        return settings

    def _run_mrmr(self):

        with utils.Timer() as timer:
            _out_dir = utils.check_make_dir(self.out_dir.joinpath('mrmr/'), True)

            # check for constant features: (mrmr excludes them internally)
            constant_flags = np.apply_along_axis(lambda x: np.all(x == x[0]), 0, self.data.x)
            if constant_flags.any():
                constant_idx = np.where(constant_flags == 1)[0]
                _constant_dict = {self.data.feat_map[_idx]: self.data.x[0, _idx] for _idx in constant_idx}
                utils.dump_to_json(_constant_dict, _out_dir.joinpath(f"constant_features.json"))

                logger.info(f"found constant features: {_constant_dict}")
                _feat_map = self.data.feat_map[~constant_flags]
            else:
                logger.debug(f"no constant features found.")
                _feat_map = self.data.feat_map

            x_train_df = pd.DataFrame(self.data.x, columns=self.data.feat_map)
            y_true_train_df = pd.Series(self.data.y)

            n_feats = len(_feat_map)
            _K, _n_jobs = n_feats, 1

            if n_feats >= 1000:
                _K = min(200, n_feats)
                _n_jobs = -1
                logger.warning(
                    f"MRMR: large feature set detected (n_feats={n_feats}). "
                    f"Using K={_K} and n_jobs={_n_jobs} for speed"
                    f" → Unselected features will have rank=NaN and MRMR ranking is not deterministic anymore!")

            selected, relevance, redundancy = mrmr_classif(
                X=x_train_df,
                y=y_true_train_df,
                K=_K,
                return_scores=True,
                n_jobs=_n_jobs,
                show_progress=False
            )

            # Full table over all features: MRMR ranks only for selected, NaN otherwise
            ranking = pd.DataFrame({'name': _feat_map})
            ranking['rank'] = np.nan  # IMPORTANT: NaN means "not ranked by MRMR"

            feature_ranks = {feature: rank for rank, feature in enumerate(selected)}  # 0..K-1
            for feat in selected:
                ranking.loc[ranking['name'] == feat, 'rank'] = int(feature_ranks[feat] + 1)

            if _K == n_feats:
                redundancy_matrix = np.array(redundancy)
                symmetry_diff = redundancy_matrix - redundancy_matrix.T
                _thresh = 1e-5
                non_symmetric_indices = np.sum(np.abs(symmetry_diff) > _thresh)

                if not np.allclose(redundancy_matrix, redundancy_matrix.T):
                    logger.debug(
                        f"Redundancy matrix is not symmetric! "
                        f"asymmetry at {_thresh}: {non_symmetric_indices}/"
                        f"{redundancy_matrix.shape[0] * redundancy_matrix.shape[1]}"
                        f"= {100 * non_symmetric_indices / (redundancy_matrix.shape[0] * redundancy_matrix.shape[1]):.2f}"
                        f"%")

                redundancy_matrix = redundancy.to_numpy()
                # redundancy_matrix = (redundancy_matrix + redundancy_matrix.T) / 2

                ranking['relevance'] = np.round(relevance.to_numpy(), 3)
                ranking['mean_redundancy_1'] = np.round(np.mean(redundancy_matrix, axis=0), 3)
                ranking['mean_redundancy_2'] = np.round(np.mean(redundancy_matrix, axis=1), 3)

                top3_redundant_features = []
                for i, feature in enumerate(_feat_map):
                    redundant_indices = np.argsort(-redundancy_matrix[i])[:4]
                    redundant_features = [_feat_map[idx] for idx in redundant_indices if idx != i][:3]
                    top3_redundant_features.append(redundant_features)
                ranking['top3_redundant'] = top3_redundant_features

                if constant_flags.any():
                    ranking = pd.concat([ranking, pd.DataFrame({
                        'name': list(_constant_dict.keys()),
                        'rank': range(len(ranking) + 1, len(ranking) + 1 + len(_constant_dict)),
                        'relevance': [np.nan] * len(_constant_dict),
                        'mean_redundancy_1': [np.nan] * len(_constant_dict),
                        'mean_redundancy_2': [np.nan] * len(_constant_dict),
                        'top3_redundant': [np.nan] * len(_constant_dict),
                    })], ignore_index=True)
            else:
                ranking['relevance'] = np.nan
                ranking['mean_redundancy_1'] = np.nan
                ranking['mean_redundancy_2'] = np.nan
                ranking['top3_redundant'] = np.nan

                if constant_flags.any():
                    ranking = pd.concat([ranking, pd.DataFrame({
                        'name': list(_constant_dict.keys()),
                        'rank': [np.nan] * len(_constant_dict),
                        'relevance': [np.nan] * len(_constant_dict),
                        'mean_redundancy_1': [np.nan] * len(_constant_dict),
                        'mean_redundancy_2': [np.nan] * len(_constant_dict),
                        'top3_redundant': [np.nan] * len(_constant_dict),
                    })], ignore_index=True)

            ranking = ranking.sort_values(by='rank', na_position='last').set_index('rank')
            ranking.to_csv(_out_dir.joinpath(f"mrmr_{self.data_config.agg_level}.csv"))
            if 'redundancy_matrix' in locals():
                del redundancy_matrix
            logger.info(f"applied MRMR. ({timer()}s)")

        return {f"{self.data_config.agg_level}": ranking}

    def _eval_corr_wrapper(self):
        # cache the correlation results to avoid duplicate computation.
        if not hasattr(self, '_cached_corr'):
            self._cached_corr = self._eval_corr()
        return self._cached_corr  # _cached_corr is a tuple: (label_corr_dict, inter_corr_dict)

    def _eval_corr(self):
        with utils.Timer() as timer:
            _out_dir = utils.check_make_dir(self.out_dir.joinpath('pca_corr/'), True)
            _result_dict_label = {f"{self.data_config.agg_level}": None}
            _result_dict_inter = {f"{self.data_config.agg_level}": None}

            # label correlation:
            n_total_features = len(self.data.feat_map)
            assert n_total_features == self.data.x.shape[1], \
                f"n_features={n_total_features} does not match x.shape={self.data.x.shape}"

            constant_flags = np.apply_along_axis(lambda x: np.all(x == x[0]), 0, self.data.x)
            if constant_flags.any():
                constant_idx = np.where(constant_flags == 1)[0]
                _constant_dict = {self.data.feat_map[_idx]: self.data.x[0, _idx] for _idx in constant_idx}
                utils.dump_to_json(_constant_dict, _out_dir.joinpath(f"constant_features.json"))

                logger.info(f"found constant features: {_constant_dict}")
                x_train_no_constants = self.data.x[:, ~constant_flags]
                n_features = n_total_features - len(constant_idx)
                feat_map_no_constants = self.data.feat_map[~constant_flags]
            else:
                logger.debug(f"no constant features found.")
                x_train_no_constants = self.data.x
                n_features = n_total_features
                feat_map_no_constants = self.data.feat_map

            label_corr, inter_corr = self._compute_correlations(
                x_train_no_constants, self.data.y, n_features, self.n_jobs, legacy_mode=self.legacy_mode)

            # create ranking for feat2label correlations: (mean over pearson, kendall, spearman)
            _label_corr_mean = np.nanmean([_label_corr for _label_corr in label_corr.values()], axis=0)

            _rank_label = stats.rankdata(-np.abs(_label_corr_mean), method='max', nan_policy='omit')
            results_label = pd.DataFrame({
                'name': feat_map_no_constants,
                'rank': _rank_label,
                'mean': _label_corr_mean,
                'pearson': label_corr['pearson'],
                'spearman': label_corr['spearman'],
                'kendall': label_corr['kendall'],
            })
            if constant_flags.any():
                results_label = pd.concat([results_label, pd.DataFrame({
                    'name': list(_constant_dict.keys()),
                    'rank': range(len(results_label) + 1, len(results_label) + 1 + len(_constant_dict)),
                    'mean': [np.nan] * len(_constant_dict),
                    'pearson': [np.nan] * len(_constant_dict),
                    'spearman': [np.nan] * len(_constant_dict),
                    'kendall': [np.nan] * len(_constant_dict),
                })], ignore_index=True)

            results_label = results_label.sort_values(by='rank').set_index('rank')
            results_label.to_csv(_out_dir.joinpath(f"label_corr_{self.data_config.agg_level}.csv"))
            _result_dict_label.update({f"{self.data_config.agg_level}": results_label})

            # create ranking for inter-feature correlations: (mean over pearson, kendall, spearman)
            _inter_corr_mean = np.nanmean([_inter_corr for _inter_corr in inter_corr.values()], axis=(0, 1))
            _rank_inter = stats.rankdata(
                np.abs(_inter_corr_mean),
                method='max',  # smaller mean corr -> less redundant
                nan_policy='omit'
            )

            results_inter = pd.DataFrame({
                'name': feat_map_no_constants,
                'rank': _rank_inter,
                'mean': _inter_corr_mean,
                'pearson': np.nanmean(inter_corr['pearson'], axis=0),  # nanmean because nan if constant values
                'spearman': np.nanmean(inter_corr['spearman'], axis=0),
                'kendall': np.nanmean(inter_corr['kendall'], axis=0),
            })
            if constant_flags.any():
                results_inter = pd.concat([results_inter, pd.DataFrame({
                    'name': list(_constant_dict.keys()),
                    'rank': range(len(results_inter) + 1, len(results_inter) + 1 + len(_constant_dict)),
                    'mean': [np.nan] * len(_constant_dict),
                    'pearson': [np.nan] * len(_constant_dict),
                    'spearman': [np.nan] * len(_constant_dict),
                    'kendall': [np.nan] * len(_constant_dict),
                })], ignore_index=True)

            results_inter = results_inter.sort_values(by='rank').set_index('rank')
            results_inter.to_csv(_out_dir.joinpath(f"inter_corr_{self.data_config.agg_level}.csv"))
            _result_dict_inter.update({f"{self.data_config.agg_level}": results_inter})

            del results_label, results_inter, label_corr, inter_corr
            logger.info(f"evaluated correlations. ({timer()}s)")

        return _result_dict_label, _result_dict_inter

    def _eval_variance(self):
        with utils.Timer() as timer:
            _out_dir = utils.check_make_dir(self.out_dir.joinpath('statistics/'), True)

            scaler = MinMaxScaler()
            x = scaler.fit_transform(self.data.x)
            var, med = np.var(x, axis=0), np.median(x, axis=0)
            rel_var = var / np.max(var)

            # sorted_indices = np.argsort(var)
            # sorted_var, sorted_labels = var[sorted_indices], np.array(self.data.feat_map)[sorted_indices]
            ranking = pd.DataFrame({})
            ranking['name'] = self.data.feat_map
            ranking['var'] = var
            ranking['rel_var'] = rel_var
            ranking['med'] = med
            ranking['rank'] = stats.rankdata(-var, method='max')
            ranking = ranking.sort_values(by='rank').set_index('rank')
            ranking.to_csv(_out_dir.joinpath(f"variance_{self.data_config.agg_level}.csv"))

            del var, rel_var, med, x
            logger.info(f"evaluated variance. ({timer()}s)")

        return {f"{self.data_config.agg_level}": ranking}

    def _eval_stat_sign_groups(self):

        def __stat_dict_to_row(sig_stat_dict, _feat_name):
            return {
                'name': _feat_name,
                'combined_p': stats.combine_pvalues([
                    sig_stat_dict['difference']['T-test']['p'],
                    sig_stat_dict['difference']['Mann-Whitney-U']['p']],
                    method='fisher').pvalue,
                't_test_p': sig_stat_dict['difference']['T-test']['p'],
                'mann_whitney_p': sig_stat_dict['difference']['Mann-Whitney-U']['p'],
                'combined_normality_hc': stats.combine_pvalues([
                    sig_stat_dict['normality']['hc']['shapiro']['p'],
                    sig_stat_dict['normality']['hc']['kolmogorov']['p']],
                    method='fisher').pvalue,
                'combined_normality_rbd': stats.combine_pvalues([
                    sig_stat_dict['normality']['rbd']['shapiro']['p'],
                    sig_stat_dict['normality']['rbd']['kolmogorov']['p']],
                    method='fisher').pvalue,
            }

        with utils.Timer() as timer:
            _out_dir = utils.check_make_dir(self.out_dir.joinpath('rbd_vs_hc/'), True)
            ranking = pd.DataFrame()
            feature_info = [("local", [_feat for _feat in self.data.feat_map
                                       if _feat not in self.data_config.loader.included_global_features], 0), ]
            if self.data_config.agg_level in ('night', 'patient'):
                feature_info.append(("global", self.data_config.loader.included_global_features,
                                     len(self.data_config.loader.included_local_features)))

            rbd_idx, hc_idx = np.where(self.data.y == 1), np.where(self.data.y == 0)
            total_feats = sum(len(feat_list) for _, feat_list, _ in feature_info)
            with utils.custom_tqdm(total=total_feats, leave=True) as pbar:
                pbar.set_description(f"[PROGRESS]: ({self.data_config.agg_level}) Features")
                for scope, feat_list, offset in feature_info:
                    for i, feature in enumerate(feat_list):
                        pbar.set_postfix({"name": f"{feature}"})
                        rbd_feat, hc_feat = self.data.x[:, offset + i][rbd_idx], self.data.x[:, offset + i][hc_idx]

                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning,
                                                    message="p-value may not be accurate for N > 5000.")
                            warnings.filterwarnings('ignore', category=RuntimeWarning,
                                                    message="divide by zero encountered in log")
                            warnings.filterwarnings('ignore', category=RuntimeWarning,
                                                    message="invalid value encountered in divide")
                            warnings.filterwarnings('ignore', category=UserWarning,
                                                    message="Input data for shapiro has range zero."
                                                            " The results may not be accurate.")
                            warnings.filterwarnings('ignore', category=RuntimeWarning,
                                                    message="Precision loss occurred in moment calculation due to "
                                                            "catastrophic cancellation. This occurs when the data are"
                                                            " nearly identical. Results may be unreliable.")
                            _stat_dict_ = utils.independent_stat_significance_test(
                                hc_feat, rbd_feat, names=('hc', 'rbd'))
                            ranking = pd.concat(
                                [ranking, pd.DataFrame([__stat_dict_to_row(_stat_dict_, feature)])], ignore_index=True)
                        del _stat_dict_, rbd_feat, hc_feat
                        pbar.update(1)

            ranking['rank'] = stats.rankdata(ranking['combined_p'], method='max', nan_policy='omit')
            nan_ranks = ranking[ranking['rank'].isna()].index
            ranking.loc[nan_ranks, 'rank'] = len(ranking)
            ranking = ranking.sort_values(by='rank').set_index('rank')
            ranking.to_csv(_out_dir.joinpath(f"rbd_vs_hc_{self.data_config.agg_level}.csv"))
            logger.info(f"evaluated statistical significances. ({timer()}s)")

        return {f"{self.data_config.agg_level}": ranking}

    @staticmethod
    def _compute_correlations(
            x_train_no_constants: np.ndarray, _y_true_train: np.ndarray, _n_features: int, _n_jobs: int,
            *, legacy_mode: bool = False) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """ One backend with two modes.
        legacy_mode=True:
            exact older behavior:
              - label: pearson/spearman/kendall via scipy per feature
              - inter: pearson/spearman/kendall via scipy per pair (lower-tri filled), diag=1
        legacy_mode=False (optimized for speedup in case of large n features):
            - label: pearson + spearman vectorized; kendall parallel (kept)
            - inter: pearson matrix vectorized; spearman/kendall returned as NaN matrices (diag=1)"""
        X = np.asarray(x_train_no_constants, dtype=float)
        y = np.asarray(_y_true_train, dtype=float)
        assert X.ndim == 2, "X must be 2D"
        assert X.shape[1] == _n_features, f"X has {X.shape[1]} cols but _n_features={_n_features}"

        if legacy_mode:
            _label_corr = {
                "pearson": np.zeros(_n_features, dtype=float),
                "spearman": np.zeros(_n_features, dtype=float),
                "kendall": np.zeros(_n_features, dtype=float),
            }
            _inter_corr = {
                "pearson": np.zeros((_n_features, _n_features), dtype=float),
                "spearman": np.zeros((_n_features, _n_features), dtype=float),
                "kendall": np.zeros((_n_features, _n_features), dtype=float),
            }

            def __compute_for_feature(i: int):
                _pr_label, _ = stats.pearsonr(X[:, i], y)
                _sr_label, _ = stats.spearmanr(X[:, i], y)
                _kr_label, _ = stats.kendalltau(X[:, i], y)
                _label_corr_result = (i, _pr_label, _sr_label, _kr_label)

                _inter_corr_results = []
                for _j in range(i + 1, _n_features):
                    _pr_inter, _ = stats.pearsonr(X[:, i], X[:, _j])
                    _sr_inter, _ = stats.spearmanr(X[:, i], X[:, _j])
                    _kr_inter, _ = stats.kendalltau(X[:, i], X[:, _j])
                    _inter_corr_results.append((_j, i, _pr_inter, _sr_inter, _kr_inter))

                return _label_corr_result, _inter_corr_results

            with Parallel(n_jobs=_n_jobs) as corr_executor:
                results = corr_executor(delayed(__compute_for_feature)(i) for i in range(_n_features))

            for label_corr_result, inter_corr_results in results:
                i, pr, sr, kr = label_corr_result
                _label_corr["pearson"][i] = pr
                _label_corr["spearman"][i] = sr
                _label_corr["kendall"][i] = kr

                for j, i2, corr_pearson, corr_spearman, corr_kendall in inter_corr_results:
                    _inter_corr["pearson"][j, i2] = corr_pearson
                    _inter_corr["spearman"][j, i2] = corr_spearman
                    _inter_corr["kendall"][j, i2] = corr_kendall

            np.fill_diagonal(_inter_corr["pearson"], 1.0)
            np.fill_diagonal(_inter_corr["spearman"], 1.0)
            np.fill_diagonal(_inter_corr["kendall"], 1.0)
            return _label_corr, _inter_corr

        # -------- optimized mode below --------

        def _safe_zscore_cols(A: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
            """
            Column-wise z-score using population variance (mean of squares).
            Returns (Az, std). Constant cols -> all-zeros in Az, std=0.
            """
            mu = np.mean(A, axis=0)
            Ac = A - mu
            var = np.mean(Ac * Ac, axis=0)
            std = np.sqrt(var)
            denom = np.where(std < eps, 1.0, std)  # avoid div0
            Az = Ac / denom
            return Az, std

        def _pearson_feat_to_y(A: np.ndarray, b: np.ndarray) -> np.ndarray:
            b = b.astype(float)
            bc = b - np.mean(b)
            b_std = np.sqrt(np.mean(bc * bc))
            if b_std == 0:
                return np.full(A.shape[1], np.nan, dtype=float)

            Az, Astd = _safe_zscore_cols(A)
            bz = bc / b_std
            corr = np.mean(Az * bz[:, None], axis=0)
            corr = np.where(Astd == 0, np.nan, corr)  # match scipy pearsonr for constant X
            return corr

        def _spearman_feat_to_y(A: np.ndarray, b: np.ndarray) -> np.ndarray:
            Ar = stats.rankdata(A, axis=0)
            br = stats.rankdata(b)
            return _pearson_feat_to_y(Ar, br)

        def _kendall_feat_to_y_parallel(A: np.ndarray, b: np.ndarray, n_jobs: int) -> np.ndarray:
            def one(i: int) -> float:
                v, _ = stats.kendalltau(A[:, i], b)
                return v

            return np.array(Parallel(n_jobs=n_jobs)(delayed(one)(i) for i in range(A.shape[1])), dtype=float)

        def _pearson_inter_matrix(A: np.ndarray) -> np.ndarray:
            Az, Astd = _safe_zscore_cols(A)
            n = float(A.shape[0])
            C = (Az.T @ Az) / n

            const = (Astd == 0)
            if np.any(const):
                C[const, :] = np.nan
                C[:, const] = np.nan
            np.fill_diagonal(C, 1.0)
            return C

        # label: pearson + spearman vectorized; kendall kept (parallel)
        pr = _pearson_feat_to_y(X, y)
        sr = _spearman_feat_to_y(X, y)
        kr = _kendall_feat_to_y_parallel(X, y, n_jobs=_n_jobs)

        _label_corr = {"pearson": pr, "spearman": sr, "kendall": kr}

        # inter: pearson fast; spearman/kendall placeholders (keys preserved)
        pearson_mat = _pearson_inter_matrix(X)

        spearman_mat = np.full_like(pearson_mat, np.nan, dtype=float)
        kendall_mat = np.full_like(pearson_mat, np.nan, dtype=float)
        np.fill_diagonal(spearman_mat, 1.0)
        np.fill_diagonal(kendall_mat, 1.0)

        _inter_corr = {"pearson": pearson_mat, "spearman": spearman_mat, "kendall": kendall_mat}
        return _label_corr, _inter_corr

    @staticmethod
    def _plot_feature_selection_summary(_rank_df_: pd.DataFrame, _cmap: str = 'Greens_r', _top_k: int = 30):
        _rank_df_ = _rank_df_.reset_index()
        _rank_df_.drop(columns='diff_weight_mean', inplace=True)
        feat_names, total_rank, mean_rank, w_mean_rank = _rank_df_.pop('name').values, _rank_df_.pop(
            'total_rank').values, _rank_df_.pop('mean_rank').values, _rank_df_.pop('weighted_mean').values
        methods = _rank_df_.columns.values
        matrix_data = _rank_df_.to_numpy()

        _vmin, _vmax = 1, len(feat_names)
        if _top_k:
            _im_alpha = .7
        else:
            _im_alpha = 1

        fig, axd = plt.subplot_mosaic(
            [
                ['cbar', 'cbar', 'cbar', 'cbar'],
                ['im', 'mean', 'w_mean', 'total']
            ],
            figsize=(10, 10),
            width_ratios=(1, .1, .1, .1),
            height_ratios=(.04, 1)
        )

        ax_cbar, ax_im, ax_mean, ax_w_mean, ax_total = axd.values()
        plt.subplots_adjust(top=.92, bottom=.1, left=.1, right=.97, wspace=.1, hspace=.1)

        im = ax_im.imshow(matrix_data, vmin=_vmin, vmax=_vmax, cmap=_cmap, aspect='auto', alpha=_im_alpha)  # YlGnBu
        if _top_k:
            ax_im.axhline(_top_k - 1, c='r', zorder=20, lw=2)
        ax_im.spines['top'].set_visible(False)
        ax_im.spines['right'].set_visible(False)
        ax_im.set_xticks(np.arange(matrix_data.shape[1]), np.arange(1, matrix_data.shape[1] + 1))
        tick_positions = np.arange(4, matrix_data.shape[0], 5)
        tick_labels = [pos + 1 for pos in tick_positions]
        ax_im.set_yticks(tick_positions)
        ax_im.set_yticklabels(tick_labels)
        ax_im.tick_params(which='major', width=2, length=5, tickdir='out', labelsize=15)
        ax_im.set_xticks(np.arange(-.5, matrix_data.shape[1]), minor=True)
        ax_im.set_yticks(np.arange(-.5, len(feat_names)), minor=True)
        ax_im.set_xticks(np.arange(len(methods)), methods, rotation=45)
        ax_im.grid(which='minor', color='w', alpha=1, lw=.5)
        ax_im.set_xlabel('ranking method', fontsize=20, labelpad=15)
        ax_im.set_ylabel('feature', fontsize=20, labelpad=15)

        ax_mean.imshow(mean_rank[:, np.newaxis], vmin=_vmin, vmax=_vmax, cmap=_cmap, aspect='auto', alpha=_im_alpha)
        ax_w_mean.imshow(w_mean_rank[:, np.newaxis], vmin=_vmin, vmax=_vmax, cmap=_cmap, aspect='auto', alpha=_im_alpha)
        ax_total.imshow(total_rank[:, np.newaxis], vmin=_vmin, vmax=_vmax, cmap=_cmap, aspect='auto', alpha=_im_alpha)

        for ax, label in zip((ax_mean, ax_w_mean, ax_total), ('mean', 'w_mean', 'final')):
            ax.set_yticks(np.arange(-.5, len(feat_names)), minor=True)
            ax.grid(which='minor', color='w', alpha=1, lw=.5, axis='y')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(which='minor', left=False)
            ax.set_xlabel(label, rotation=45, fontsize=20, labelpad=15)
            if _top_k:
                ax.axhline(_top_k - 1, c='r', zorder=20, lw=2)

            for _loc in ('left', 'right', 'bottom', 'top'):
                ax.spines[_loc].set_visible(False)

        cbar = fig.colorbar(im, cax=ax_cbar, orientation='horizontal')
        cbar.ax.set_title('feature rank', fontsize=20, pad=15, fontweight='bold')
        cbar.ax.invert_xaxis()
        ax_cbar.tick_params(which='major', width=3, length=10, tickdir='in', bottom=False, labelbottom=False, top=True,
                            labeltop=True, labelsize=15)

        return fig
