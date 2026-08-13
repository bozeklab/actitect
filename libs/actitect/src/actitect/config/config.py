import re
from dataclasses import dataclass, fields, asdict, MISSING
from pathlib import Path
from typing import Type, TypeVar, Any, Dict, List, Optional, get_origin, get_args, Union

import yaml

__all__ = ['PipelineConfig', 'ModelConfig', 'DataConfig', 'NestedCVConfig',
           'FinalModelConfig', 'ExternalTestConfig', 'ExperimentConfig', 'LoaderConfig', 'DatasetConfig',
           'TopKFeatsConfig', 'FeatureSelectionConfig', 'RebalanceDatasetsConfig']


@dataclass
class BaseConfig:
    """Base dataclass to load pipeline.yaml config file."""
    T = TypeVar('T', bound='BaseConfig')

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        init_kwargs = {}
        for field_ in fields(cls):
            field_name = field_.name
            field_type = cls._unwrap_optional(field_.type)

            # allow missing keys if defaults or Optional[...] exist
            if field_name not in data:
                if field_.default is not MISSING:
                    init_kwargs[field_name] = field_.default
                    continue
                if field_.default_factory is not MISSING:  # type: ignore[attr-defined]
                    init_kwargs[field_name] = field_.default_factory()  # type: ignore[misc]
                    continue
                if get_origin(field_.type) is Union and type(None) in get_args(field_.type):
                    init_kwargs[field_name] = None
                    continue
                raise KeyError(f"Missing key '{field_name}' in configuration.")

            value = data[field_name]

            # Handle nested dataclasses
            if hasattr(field_type, '__dataclass_fields__'):
                init_kwargs[field_name] = field_type.from_dict(value)

            # Handle lists of dataclasses
            elif getattr(field_type, '__origin__', None) is list:
                elem_type = field_type.__args__[0]  # Extract list element type
                if hasattr(elem_type, '__dataclass_fields__'):  # Ensure elements are dataclasses
                    init_kwargs[field_name] = [elem_type.from_dict(v) for v in value]
                else:
                    init_kwargs[field_name] = value  # Handle regular lists (e.g., List[str])

            else:
                init_kwargs[field_name] = value

        return cls(**init_kwargs)

    @classmethod
    def from_yaml(cls, which: str, yaml_path: Path = None) -> 'BaseConfig':
        from importlib.resources import files as _pkg_files
        _rbdisco_configs = _pkg_files('actitect.rbdisco').joinpath('configs')
        if not yaml_path:
            if which == 'internal_cv':
                yaml_path = _rbdisco_configs.joinpath('pipeline_singleCenter.yaml')
            elif which == 'external_cv':
                yaml_path = _rbdisco_configs.joinpath('pipeline_multiCenter.yaml')
            elif which == 'external_test':
                yaml_path = _rbdisco_configs.joinpath('external_test.yaml')
            else:
                raise ValueError(
                    f"Unknown value for {which}: {yaml_path}, either directly set 'yaml_path' or choose"
                    f" 'which' from 'internal_cv', 'external_cv' or 'external_test'.")

        with yaml_path.open('r') as file:
            config_dict = yaml.safe_load(file)
        return cls.from_dict(config_dict)

    @staticmethod
    def _unwrap_optional(tp):
        """If type is Optional[T], return T, else return type unchanged."""
        origin = get_origin(tp)
        if origin is Union:
            args = [a for a in get_args(tp) if a is not type(None)]
            if len(args) == 1:
                return args[0]
        return tp

    def dict(self) -> dict:
        return asdict(self)

    def copy(self: T) -> T:
        return type(self).from_dict(self.dict())


@dataclass
class EarlyStoppingConfig(BaseConfig):
    early_stopping_rounds: int
    eval_metric: List[str]


@dataclass
class BayesParamsConfig(BaseConfig):
    n_calls: int
    use_early_stopping: bool
    verbose: bool


@dataclass
class TopKFeatsConfig(BaseConfig):
    tune: bool
    low: int
    high: int
    fixed_value: int


@dataclass
class FeatureSelectionConfig(BaseConfig):
    top_k_feats: TopKFeatsConfig
    fixed_features: Optional[List[str]] = None
    fair_agg: Optional[str] = None

    # optionally skip or speedup boruta for large N of samples
    use_boruta: bool = True
    boruta_speedup_sample_threshold: Optional[int] = None
    boruta_speedup_max_depth: Optional[int] = None
    boruta_speedup_n_iterations: Optional[int] = None
    boruta_speedup_n_estimators: Optional[int] = None
    boruta_speedup_max_samples: Optional[float] = None
    boruta_speedup_max_iter: Optional[int] = None
    boruta_speedup_n_iter_no_change: Optional[int] = None
    boruta_speedup_rf_n_jobs: Optional[int] = None


@dataclass
class ModelConfig(BaseConfig):
    which: str
    dummy: str
    early_stopping: EarlyStoppingConfig
    bayes_params: BayesParamsConfig
    feature_selection: Optional[FeatureSelectionConfig]
    hp_overrides: Optional[Dict[str, Any]] = None
    dataset_weighting: Optional[str] = None


@dataclass
class ProcessingConfig(BaseConfig):
    scaling_order: str
    use_smote: bool
    scaler: str = None  # 'minmax', 'standard', 'robust' or None


@dataclass
class RebalanceDatasetsConfig(BaseConfig):
    """Optional dataset rebalancing during training.
    method:
      none | cap_to_second_largest | min | equalize_to_median |
      cap_to_k | proportional_to_quota | weights_only"""
    method: str
    apply_to: str
    random_state: int
    preserve_class_ratio: bool

    # Optional parameters for specific methods
    dominance_ratio: float = 1.4  # cap_to_second_largest gate
    k_per_dataset: Optional[int] = None  # cap_to_k absolute cap
    quotas: Optional[Dict[str, float]] = None  # proportional_to_quota quotas per dataset


@dataclass
class LoaderConfig(BaseConfig):
    binary_mapping: dict
    feature_dir: str  # relative to patient_dir
    aggregation: List[str]
    included_local_features: List[str]
    included_global_features: List[str]
    rebalance_datasets: Optional[RebalanceDatasetsConfig] = None
    group_level: Optional[str] = 'patient'  # or 'record', will decide aggregation level
    str_label_mapping: Optional[str] = None


@dataclass
class DataConfig(BaseConfig):
    agg_level: str
    processing: ProcessingConfig
    loader: LoaderConfig


@dataclass
class CVConfig(BaseConfig):  # needs to be compatible with sklearn.model_selection.BaseCrossValidator kwargs
    n_splits: int
    shuffle: bool


@dataclass
class ExperimentConfig(BaseConfig):
    name: str
    threshold: str
    patient_aggregation: str
    hp_setup_name: str = None
    early_stopping: bool = None
    calibration: str = None


@dataclass
class NestedCVConfig(BaseConfig):
    n_repeats: int
    n_jobs: int
    outer_cv: CVConfig
    inner_cv: CVConfig
    n_interp_points_roc_pr: int
    n_interp_points_calibration: int
    calib_plot_hist_bin_width: float
    default_experiment: ExperimentConfig
    log_night_eval: bool
    stratify_by_dataset_if_pooled: bool
    min_patient_nights_eval: int
    load_path_cv_feature_rankings: Union[str, Path] = None
    ranking_seed: int = None  # defaults to global random state
    extra_diagnostic_metrics: bool = False
    output_predictions: bool = False


@dataclass
class FinalModelConfig(BaseConfig):
    skip_pretrain: bool
    n_jobs: int
    bayes_cv: CVConfig
    bayes_params: BayesParamsConfig
    stratify_by_dataset_if_pooled: bool
    early_stopping: EarlyStoppingConfig
    include_pretrain_merged: bool
    overwrite_final_repo_models: bool
    log_night_level: bool
    min_patient_nights_eval: int
    output_patient_csv: bool
    experiments: List[ExperimentConfig]
    load_path_feature_rankings: Union[str, Path] = None
    nested_cv_path: Union[str, Path] = None
    save_path_models: Union[str, Path, list] = None
    calibration_cv: CVConfig = None


@dataclass
class PipelineConfig(BaseConfig):
    random_state: int
    model: ModelConfig
    data: DataConfig
    nested_cv: NestedCVConfig
    final_model: FinalModelConfig


@dataclass
class ExternalTestConfig(BaseConfig):
    random_state: int
    n_jobs: int
    min_patient_nights_eval: int
    pretrained_model_dirs: List[str]
    data: DataConfig
    log_night_eval: bool
    output_patient_csv: bool
    extra_diagnostic_metrics: Optional[bool] = False


@dataclass
class DatasetConfig:
    colors: dict
    aliases: dict

    @staticmethod
    def _normalize_key(name: str) -> str:
        name = name.strip().lower()
        name = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', name)  # camelCase → camel Case
        name = re.sub(r'[\s_\-]+', ' ', name)  # unify delimiters
        return name

    @classmethod
    def from_yaml(cls, path: Path = Path(__file__).parent.joinpath('dataset_2_color.yaml')) -> 'DatasetConfig':
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        aliases = {cls._normalize_key(k): v for k, v in config.get('aliases', {}).items()}
        return cls(colors=config['datasets'], aliases=aliases)

    def resolve(self, name: str):
        key = self._normalize_key(name)
        canonical = self.aliases.get(key, name)
        return canonical, self.colors.get(canonical)
