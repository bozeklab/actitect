import argparse
import datetime
import logging
import sys
from importlib.resources import files
from pathlib import Path

from actitect import utils
from actitect.config import PipelineConfig, ExternalTestConfig
from .pipelines import TrainPipeline, TestPipeline, PooledTrainPipeline

logger = logging.getLogger(__name__)


def _run_setup():
    FLAG2MODE = dict(train_eval='train', train_pooled='trainPooled', test='test', predict='predict')

    def __parse_args():
        parser = argparse.ArgumentParser(
            prog='RBDisco',
            description='Entry point to perform RBD analysis from actigraphy data. Call --help or -h for details.',
            formatter_class=argparse.RawTextHelpFormatter)

        # arguments that define the operation mode of the script:
        group = parser.add_mutually_exclusive_group(required=False)  # to enable default
        group.add_argument('--train_eval', action='store_true',
                           help='Run the entire training pipeline on cologne data: '
                                'nested cv on train set + train model on entire train set to eval on test set. ')
        group.add_argument('--train_pooled', action='store_true',
                           help='Run a training pipeline on a pooled multi-center dataset'
                                ' (several processed-data folders jointly).')
        group.add_argument('--test', action='store_true',
                           help='Use pretrained models to test on unseen data. (Default)')
        group.add_argument('--predict', action='store_true', help='Only run inference, no testing.')

        # Additional arguments for options within modes
        parser.add_argument('-r', '--root_dir', type=str, metavar='DIR',
                            default=str(utils.get_experiment_root()),
                            help="The root directory containing 'data/' and 'ActiTect/'.")

        parser.add_argument('-c', '--config_file', type=str, metavar='FILE',  # rel. to root_dir
                            default='./ActiTect/libs/actitect-rbdisco/src/actitect/rbdisco/configs/external_test.yaml',
                            help='full path (rel. to root or abs. path) of the config .yaml file '
                                 'defining preprocessing settings.')

        parser.add_argument('-d', '--processed_data_dir', type=str, nargs='+', metavar='DIR',
                            default=['./data/processed/'],  # rel. to root dir
                            help='directory (rel. to root or abs. path) containing the pre-processed data. '
                                 'Each patient may have subdirectories corresponding to multiple recordings.')

        parser.add_argument('--output_dir', type=str, default=None,
                            help="If set, write results exactly to this directory (no auto-generated timestamp).")

        parser.add_argument('-m', '--meta_file', type=str, metavar='FILE',
                            default='./data/raw/meta/metadata.csv',  # rel. to root dir
                            help="meta data file path (rel. to root or abs. path) that contains list of patients."
                                 " Has to contain the class labels except for execution in 'predict' mode. ")

        parser.add_argument('-ds', '--dataset_tag', type=str, default='external',
                            help="Tag used to identify the dataset (used only in test mode, e.g. 'cologne', 'oxford',"
                                 " ...).")

        _args = parser.parse_args()

        # Set default operation mode:
        if not any([_args.train_eval, _args.train_pooled, _args.test, _args.predict]):
            _args.test = True

        # cast path-like args into PosixPath types
        _args.root_dir = Path(_args.root_dir).expanduser()

        _args.config_file = utils.resolve_root_or_abs_path(_args.root_dir, _args.config_file)
        _args.processed_data_dir = [
            utils.resolve_root_or_abs_path(_args.root_dir, _dir) for _dir in _args.processed_data_dir]
        _args.meta_file = utils.resolve_root_or_abs_path(_args.root_dir, _args.meta_file)
        if _args.output_dir is not None:
            _args.output_dir = utils.resolve_root_or_abs_path(_args.root_dir, _args.output_dir)

        for p in _args.processed_data_dir:
            assert p.is_dir(), f"Processed-data dir not found: '{p}'"
        if len(_args.processed_data_dir) == 1:
            _args.processed_data_dir = _args.processed_data_dir[0]

        assert (_args.root_dir.is_dir() and _args.root_dir.joinpath('data').is_dir() and
                _args.root_dir.joinpath('ActiTect').is_dir()), \
            (f"'--root_dir' ('-r') ({_args.root_dir}) is not a dir "
             f"or does not contain sub-dirs './data' and './ActiTect'")
        assert _args.config_file.is_file(), \
            f"'--config_file' (-'c') is not a file. ({_args.config_file})"

        return _args

    args = __parse_args()

    run_mode = next(m for f, m in FLAG2MODE.items() if getattr(args, f))
    if args.output_dir:
        root_save_path = utils.check_make_dir(Path(args.output_dir))
        # experiment_id = root_save_path.name
        logger.warning("Overriding default results directory. Writing to: %s", root_save_path)
    else:
        ts = datetime.datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')
        experiment_id = f"{run_mode}_{ts}"
        root_save_path = utils.check_make_dir(
            Path(args.root_dir).joinpath(f"results/pipeline/run_{experiment_id}"))

    utils.setup_logging(log_file_path=root_save_path.joinpath('log'))
    if args.output_dir:
        logger.warning("Overriding default results directory. Writing to: %s", root_save_path)

    if args.train_eval or args.train_pooled:
        config = PipelineConfig.from_yaml(which=None, yaml_path=args.config_file)
    elif args.test or args.predict:
        config = ExternalTestConfig.from_yaml(which=None, yaml_path=args.config_file)
        config.pretrained_model_dirs = [args.root_dir.joinpath(_dir) for _dir in config.pretrained_model_dirs]
    else:
        raise ValueError(f"must choose one of args.train_eval, args.test or args.predict.")

    _case_ident = f"{config.data.loader.feature_dir.replace('/features', '')}" + "_" + "_".join(
        config.data.loader.aggregation) + f"/{sys.platform}" + f"/scale_{config.data.processing.scaling_order}"
    if config.data.processing.scaling_order == 'before_ranking':
        _case_ident += f"/{config.data.processing.scaler}_scaler"

    if args.train_eval or args.train_pooled:
        if not config.nested_cv.load_path_cv_feature_rankings:
            _cv_dir = 'cv' if args.train_eval else 'cv_pooled'
            config.nested_cv.load_path_cv_feature_rankings = args.root_dir.joinpath(
                f"results/feat_rank/{_cv_dir}/{_case_ident}")
        else:
            config.nested_cv.load_path_cv_feature_rankings = Path(config.nested_cv.load_path_cv_feature_rankings)

        if not config.final_model.load_path_feature_rankings:
            config.final_model.load_path_feature_rankings = args.root_dir.joinpath(
                f"results/feat_rank/full/{_case_ident}")
        else:
            config.final_model.load_path_feature_rankings = Path(config.final_model.load_path_feature_rankings)

        if not config.final_model.nested_cv_path:
            config.final_model.nested_cv_path = root_save_path.joinpath('nested_cv')
        else:
            config.final_model.nested_cv_path = Path(config.final_model.nested_cv_path)

        if not config.final_model.save_path_models:
            config.final_model.save_path_models = root_save_path.joinpath('models')
            if config.final_model.overwrite_final_repo_models:
                config.final_model.save_path_models = [config.final_model.save_path_models]
                config.final_model.save_path_models.append(
                    files('actitect.rbdisco').joinpath('models/pretrained/'))
        else:
            config.final_model.save_path_models = Path(config.final_model.save_path_models)

    # save the settings
    _dump_dict = {'command line options': vars(args), 'config': config.dict()}
    utils.dump_to_json(_dump_dict, root_save_path.joinpath('settings.json'))

    return args, config, root_save_path


def main():
    # setup
    args, config, save_path = _run_setup()

    # define and run pipeline
    pipeline_cls = TrainPipeline if args.train_eval else PooledTrainPipeline if args.train_pooled else TestPipeline
    pipeline = pipeline_cls(args, config, save_path)
    pipeline.run()


if __name__ == '__main__':
    main()
