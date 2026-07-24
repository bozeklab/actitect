import logging
import warnings
from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize.linesearch import LineSearchWarning
from sklearn.model_selection import ParameterGrid, ParameterSampler
from skopt import gp_minimize

from actitect.utils import custom_tqdm
from .cross_validation import perform_group_cv
from ..models import ModelSetup

logger = logging.getLogger(__name__)

__all__ = ['BruteForceGridSearchCV', 'RandomGridSearchCV', 'BayesianOptCV']


class BaseGridSearchCV(ABC):
    """
    Base class to perform hyperparameter grid-search. Either random points on grid or brute-force over all combinations.
    Evaluates each grid point with stratified group k-fold cross-validation.

    Parameters:
        :param model: a classifier with a .fit(), .set_params() and .predict() method.
        :param param_space: (dict) defining the range of values for each HP.
        :param fixed_params: (dict) of fixed HP that are not included in the search but needed for the .
        :param n_points: (int) sets the number of grid points in case of random sampling.
        :param random_state: (int) sets the random seed in case of random sampling.

    Abstract Methods:
        _get_total_n_of_combinations(): return the total number of grid points that will be searched.
        _create_grid_point_generator(): return a generator object to create the grid points.
    """

    def __init__(self, model, param_space: dict, fixed_params: dict, n_points: int, random_state: int):
        self.model = model
        self.fixed_params = fixed_params
        self.param_space = param_space
        self.n_points = n_points
        self.random_state = random_state

        self.n_combinations = self._get_total_n_of_combinations()
        self.best_score = self.best_params = self.history = None
        self.cv_score_weights = {'accuracy': .2, 'balanced_accuracy': .2, 'precision': .2, 'recall': .2, 'f1': .2}

    @abstractmethod
    def _get_total_n_of_combinations(self):
        raise NotImplementedError

    @abstractmethod
    def _create_grid_point_generator(self):
        raise NotImplementedError

    def fit(self, x, y, group, cv_folds, cv_repeats, get_history=True):

        grid_point_generator = self._create_grid_point_generator()
        n_iter = self.n_combinations

        if get_history:
            self.history = {'iter': [], 'score': []}
        with logging_redirect_tqdm(), tqdm(total=n_iter, position=0, leave=True) as pbar:
            pbar.set_description(f"[PROGRESS]: Grid search:")

            iteration = 0

            for _grid_point in grid_point_generator:
                iteration += 1
                params = self.fixed_params.copy()
                params.update(_grid_point)
                self.model.set_params(**params)
                _cv_results = perform_group_cv(self.model, x, y, group, cv_folds, cv_repeats, False)
                _score = sum(self.cv_score_weights[metric] * _cv_results[metric]['mean'] for metric in _cv_results)
                # potential improvements: use geometric mean, including

                if self.best_score is None or _score > self.best_score:
                    self.best_score = _score
                    self.best_params = _grid_point
                    if get_history:
                        self.history['iter'].append(iteration)
                        self.history['score'].append(self.best_score)

                pbar.set_postfix({'score': _score, 'best': f"{self.best_score:.3f}"})
                pbar.update(1)

        return self.best_params, self.best_score


class BruteForceGridSearchCV(BaseGridSearchCV):
    """ Brute force over all available grid points."""

    def __init__(self, model, param_space: dict, fixed_params: dict, n_points: int = None):
        assert all(isinstance(v, list) for v in param_space.values()), \
            "The parameter ranges in 'param_space' must be given as lists."
        super().__init__(model, param_space, fixed_params, n_points, random_state=None)

    def _get_random_grid_point_generator(self, n_points):
        """mostly for debugging, use random grid search for smaller n."""
        all_grid_points = list(iter(ParameterGrid(self.param_space)))
        np.random.shuffle(all_grid_points)
        rand_grid_points = all_grid_points[:n_points]
        del all_grid_points
        for grid_point in rand_grid_points:
            yield grid_point

    def _get_total_n_of_combinations(self):
        if self.n_points:
            total_combinations = self.n_points
        else:
            total_combinations = 1
            for key in self.param_space:
                total_combinations *= len(self.param_space[key])
        return total_combinations

    def _create_grid_point_generator(self):
        if self.n_points:
            return self._get_random_grid_point_generator(self.n_points)
        else:
            return iter(ParameterGrid(self.param_space))


class RandomGridSearchCV(BaseGridSearchCV):
    """ Generate n random points from parameter distributions."""

    def __init__(self, model, param_space: dict, fixed_params: dict, n_points: int, random_state):
        super().__init__(model, param_space, fixed_params, n_points, random_state)

    def _get_total_n_of_combinations(self):
        return self.n_points

    def _create_grid_point_generator(self):
        return iter(ParameterSampler(self.param_space, n_iter=self.n_points, random_state=self.random_state))


class BayesianOptCV:
    # todo: refactor for readability.
    def __init__(self, model_setup: ModelSetup, feat_rank_map: dict, seed: int):
        self.model = model_setup.model()
        self.fixed_params = model_setup.bayesian_fixed_params
        self.param_space = model_setup.bayesian_param_space
        self.seed = seed
        self.cv_score_weights = {'accuracy': .2, 'balanced_accuracy': .2, 'precision': .2, 'recall': .2, 'f1': .2}

        self.feature_map = feat_rank_map or {}
        self._assert_all_param_names_valid(exclude=['top_k_feats'])

        self.dataset_weighting, self.ds_vector = None, None

    def _assert_all_param_names_valid(self, exclude):
        _valid_xgb_params = sorted(list(self.model.get_params().keys()))
        _all_params = sorted([param.name for param in self.param_space] + list(self.fixed_params.keys()))
        _inv_params = [param for param in _all_params if param not in _valid_xgb_params]
        for _element in exclude:
            if _element in _inv_params:
                _inv_params.remove(_element)
        assert len(_inv_params) == 0, f"not all argument names are valid for {self.model} API. not valid: {_inv_params}"

    @staticmethod
    def _objective(_x, _y, _y_strat, _model, _params, _fixed_params, _cv_params, _score_weights, _use_early_stopping,
                   _seed, _n_jobs, _dataset_weighting, _ds_vector, _cv_splits=None):
        _params.update(_fixed_params)
        _model.set_params(**_params)
        _cv_results = perform_group_cv(_model, _x, _y, _y_strat, **_cv_params,
                                       use_early_stopping=_use_early_stopping,
                                       random_seed_splitting=_seed, n_jobs=_n_jobs,
                                       dataset_weighting=_dataset_weighting, ds_vector=_ds_vector,
                                       cv_splits=_cv_splits)

        _score = (_cv_results.scoring['night']['default_thresh']['f1']['mean']
                  + _cv_results.scoring['night']['default_thresh']['auc']['mean']) / 2

        return -_score  # for gp_minimize minus is needed

    def fit(self, x, y, y_strat, cv_params, n_calls: int, use_early_stopping: bool, n_jobs: int,
            dataset_weighting: str = None, ds_vector: np.ndarray = None, verbose: bool = True, cv_splits=None):

        self.dataset_weighting = dataset_weighting
        self.ds_vector = ds_vector

        if dataset_weighting is not None:
            logger.info(f"using mode='{dataset_weighting}' weighting for dataset weighting for Bayesian HP tuning.")

        def objective(params):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=LineSearchWarning)
                param_dict = {dim.name: param for dim, param in zip(self.param_space, params)}
                if 'top_k_feats' in param_dict:
                    top_k = param_dict.pop('top_k_feats')
                    if not self.feature_map:
                        raise ValueError("top_k_feats in param space but no feature_map was provided")
                    _top_k_feat_idcs = [val['idx'] for val in self.feature_map.values() if val['rank'] <= top_k]
                    x_sel_feats = x[:, _top_k_feat_idcs]
                else:  # no ranking used, just keep all features
                    x_sel_feats = x
                return self._objective(x_sel_feats, y, y_strat, self.model, param_dict, self.fixed_params, cv_params,
                                       self.cv_score_weights, use_early_stopping, _seed=self.seed, _n_jobs=n_jobs,
                                       _dataset_weighting=self.dataset_weighting, _ds_vector=self.ds_vector,
                                       _cv_splits=cv_splits)

        optimizer_kwargs = {
            'n_calls': n_calls,  # (int, 100)
            'base_estimator': None,  # (Estimator, GaussianProcessRegressor) surrogate for gp optimizer
            'n_random_starts': 100,  # (int, 10) N of random inits before optimization
            'acq_func': 'gp_hedge',
            # (str, 'gp_hedge') func. to select the next points in ('gp_hedge', 'LCB', 'EI', 'PI').
            'acq_optimizer': 'lbfgs',  # (str, 'lbfgs') how to minimize acq_func. Possible values: 'sampling', 'lbfgs'.
            'x0': None,  # (list of lists, None) optional list of input points.
            'y0': None,  # (list, None) Evaluator values at `x0`. List of scalar values corresponding to `x0`.
            'random_state': self.seed,  # (int, RandomState instance, None)
            'verbose': False,  # (bool, False)
            'n_points': 10000,  # (int, 10000) Number of points to sample when minimizing the acquisition function.
            'n_restarts_optimizer': 5,
            # (int, 5) N of restarts for the optimizer when minimizing the acquisition function.
            'xi': 0.01,  # (float, 0.01) Exploration-exploitation trade-off parameter for 'EI' and 'PI'.
            'kappa': 1.96,
            # (float, 1.96) Exploration-exploitation trade-off parameter for 'LCB'. Any non-negative float.
            'n_jobs': n_jobs,  # (int, 1) Number of parallel jobs to run. Any integer, with `-1` using all processors.
        }
        if verbose:
            logger.info(f"performing bayes optimization with n_calls={n_calls} and early_stopping={use_early_stopping}")

        with custom_tqdm(total=n_calls, disable=(not verbose)) as pbar, warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=LineSearchWarning)
            pbar.set_description(f"[PROGRESS]: Bayesian optimization")
            _best_score = 0

            def _pbar_callback(_results):
                nonlocal _best_score
                _score = -_results.fun
                if _score > _best_score:
                    _best_score = _score
                pbar.set_postfix({'score': f"{_score:.4f}", 'best': f"{_best_score:.4f}"})
                pbar.update(1)

            result = gp_minimize(
                func=objective, dimensions=self.param_space, callback=_pbar_callback, **optimizer_kwargs)

        return result
