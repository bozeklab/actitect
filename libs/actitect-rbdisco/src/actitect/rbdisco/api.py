import logging
from dataclasses import dataclass
from importlib.resources import files as _pkg_files
from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd

from actitect.api import compute_sleep_motor_features as actitect_compute_features
from actitect.api import process as actitect_process
from actitect.config import ExperimentConfig
from actitect.features import build_aggregated_feature_list
from .core import FeatureSet, ModelManager
from .processing.aggregation import aggregate_night_predictions_to_patient_level
from .processing.classification_threshold import ClassThreshold, classify_with_threshold

logger = logging.getLogger(__name__)

__all__ = ['predict']

_MODEL_REGISTRY = dict(  # paths relative to rbdisco
    multiCenterCore='models/pretrained/multiCenterCore.joblib',
    multiCenterExtended='models/pretrained/multiCenterExtended.joblib',
    singleCenterCore='models/pretrained/singleCenterCore.joblib',
    singleCenterExtended='models/pretrained/singleCenterExtended.joblib')


def _load_pretrained(model: str = 'multiCenterCore', ) -> 'RBDiscoPredictor':
    """Load a pretrained RBDisco model with bundled metadata & thresholds.
    Parameters
        :param model: (str, optional) One of: 'multiCenterCore', 'multiCenterExtended', 'singleCenterCore',
            'singleCenterExtended'. Default is 'multiCenterCore' for the multi-site core model.
    Returns
        :return: RBDiscoPredictor"""

    def __resolve_model_path(rel_path: str) -> Path:
        """Resolve a model path packaged inside `actitect.rbdisco`.
        Falls back to filesystem relative to this file if importlib.resources is unavailable."""
        if _pkg_files is not None:
            return Path(_pkg_files('actitect.rbdisco') / rel_path)
        here = Path(__file__).resolve().parent  # fallback if env is weird
        return (here / rel_path).resolve()

    _MODEL_POSTFIX = 'model2'  # default threshold
    if model not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model}'. Available: {list(_MODEL_REGISTRY)}")

    model_path = __resolve_model_path(_MODEL_REGISTRY[model])
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. "
            f"Verify that actitect-rbdisco was installed with package data."
        )

    model_dict = joblib.load(model_path)[_MODEL_POSTFIX]
    return RBDiscoPredictor.from_model_dict(model_dict, name=model)


def predict(data: Union[str, Path, pd.DataFrame], model: str = 'multiCenterCore',
            subject_id: Union[str, None] = None, return_level: str = 'patient') -> pd.DataFrame:
    """ Use the pretrained RBDisco model to predict night or patient level RBD scores from multi-night actigraphy
    recordings.
    Parameters
        :param data: (pathlike or DataFrame) Pathlike pointing to a actitect-supported device binary
            (.cwa, .bin. gt3x, .csv) or, already computed features from actitect.api.compute_sleep_motor_features.
        :param model: (str) Choosing the model for inference, options are multiCenterCore, multiCenterExtended,
            singleCenterCore and singleCenterExtended. Choose the multi-site core model as default.
        :param subject_id: (str, optional) A str identifier on subject-level, mainly for logging and presentation
            purposes.
        :param return_level: (str) either 'night' to return the RBD probabilities and predictions per night of data, or
            'patient' to return the aggregated probability and prediction. Default is 'patient'.
    Returns
        :return: pd.DataFrame containing probability scores and thresholded predictions. """
    predictor = _load_pretrained(model)
    night_fs, night_df = predictor.predict_nights(data, subject_id)
    if return_level == 'night':
        return night_df

    else:
        patient_df = predictor.aggregate_night_df_to_patient_level(night_fs, night_df)
        return patient_df


@dataclass
class RBDiscoPredictor:
    """Lightweight predictor with fixed, model-bundled thresholds and aggregation."""
    estimator: object
    hps: ModelManager.HpSetup
    processing: dict
    thresholds: dict[str, ClassThreshold]
    meta: ExperimentConfig
    flavor: str

    def __str__(self):
        return f"RBDiscoPredictor(flavor={self.flavor})"

    @staticmethod
    def from_model_dict(model_dict: dict, name: str) -> 'RBDiscoPredictor':
        model, hps, pr, thr, exp = model_dict.values()
        return RBDiscoPredictor(estimator=model, hps=hps, processing=pr, thresholds=thr, meta=exp, flavor=name)

    def predict_nights(self, data: Union[str, Path, pd.DataFrame], subject_id: Union[str, None] = None) -> pd.DataFrame:
        """ Produce per-night probabilities and binary labels.
        Parameters
            :param data: (str, Path, pd.DataFrame) The input data to predict. Either a pathlike object, pointing to a
                actitect-supported device binary, or a DataFrame object as return by
                actitect.api.compute_sleep_motor_features.
            :param subject_id: (str, None) Optional subject ID used for logging and final df.
        Returns
            :return: (pd.DataFrame) containing per-night probabilities and binary labels."""
        if isinstance(data, (str, Path)):
            proc_df, _ = actitect_process(data, subject_id=subject_id)
            feat_df = actitect_compute_features(proc_df, subject_id=subject_id)
        elif isinstance(data, pd.DataFrame):
            feat_df = data.copy()
        else:
            raise TypeError("'data' must be a path-like object or a pandas.DataFrame.")

        fs_raw = self._df_to_feature_set(feat_df, subject_id=subject_id)
        fs_processed = fs_raw.transform(process_params=self.processing)
        fs_night = fs_processed.select_features(self.hps.selected_features)

        y_prob_night = self.estimator.predict_proba(fs_night.x)[:, 1]
        fs_night.prob = y_prob_night
        y_pred_night = classify_with_threshold(y_prob_night, self.thresholds['night'].value)
        n_nights = y_prob_night.shape[0]

        out = pd.DataFrame(
            dict(subject=[subject_id or 'none'] * n_nights,
                 night_id=np.arange(n_nights),
                 proba_rbd=y_prob_night,
                 pred=y_pred_night))
        return fs_night, out

    @staticmethod
    def _df_to_feature_set(df: pd.DataFrame, subject_id: Union[str, None] = None) -> FeatureSet:
        """ Convert a DataFrame returned by compute_sleep_motor_features into a FeatureSet for RBDisco inference. """
        feature_list = [feature for feature in build_aggregated_feature_list(from_yaml=True, order='dataloader')
            if feature in df.columns]

        feat_subset = df.loc[:, feature_list]
        groups = np.array([subject_id] * feat_subset.shape[0]) if subject_id else None
        return FeatureSet(x=feat_subset.to_numpy(), group=groups, feat_map=np.asarray(feature_list))

    def aggregate_night_df_to_patient_level(self, night_fs: FeatureSet, night_df: pd.DataFrame) -> pd.DataFrame:
        night_fs.y = np.zeros(night_fs.x.shape[0])  # temporarily set ground_truth for compat
        _temp_pat_df = aggregate_night_predictions_to_patient_level(night_fs, y_pred=night_df.pred)
        return _temp_pat_df[['id', 'n_total_nights', 'mean_prob_per_night', f'pred({self.meta.patient_aggregation})']]

