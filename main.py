from argparse import ArgumentParser
from prediction import get_predictions
from tagger import TagHandler, TagConfig
import yaml


def add_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--root-path', '-d', type=str, default=None, help='Root path of images to be tagged')
    parser.add_argument('--model', '-m', type=str, default=None, help='Model to use for predictions')
    parser.add_argument('--categories-path', '-mc', type=str, default=None, help='Model config to use for predictions')
    parser.add_argument('--default-threshold', '-t', type=float, default=None, help='Default threshold for tags')
    parser.add_argument('--custom-threshold-tags', '-ctag', type=str, nargs='+', default=None, help='Tags to apply custom thresholds to. Must be in the same order as custom-thresholds')
    parser.add_argument('--custom-thresholds', '-cthres', type=float, nargs='+', default=None, help='Thresholds for custom tags. Must be in the same order as custom_threshold_tags')
    parser.add_argument('--escape-specials', type=bool, default=None, help='Escape special characters in tags')
    parser.add_argument('--remove-underscores', type=bool, default=None, help='Remove underscores from tags when necessary')
    parser.add_argument('--hide-tags', type=str, nargs='+', default=None, help='Tags to hide from the tag list')
    parser.add_argument('--include-categories', type=str, nargs='+', default=None, help='Categories to include in the tag list')
    parser.add_argument('--prepend-tags', type=str, nargs='+', default=None, help='Tags to prepend to all files')
    parser.add_argument('--append-tags', type=str, nargs='+', default=None, help='Tags to append to all files')
    parser.add_argument('--add-tags', type=str, nargs='+', default=None, help='Tags to add to all files, sorted by priority. Only allowed if order is "alphabetical"')
    parser.add_argument('--overwrite-mode', type=str, default=None, help='How to treat existing tags. Options: "prepend", "append", "replace", "ignore"')
    parser.add_argument('--order', type=str, default=None, help='How to sort tags. Options: "alphabetical", "probability", "probability-reweighted"')
    parser.add_argument('--batch-size', '-bs', type=int, default=None, help='Batch size for predictions')
    parser.add_argument('--nproc', '-j', type=int, default=None, help='Number of workers for dataloader')

    args = parser.parse_args()
    return args


def create_config(args):
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = {'tagger': {}, 'predictor': {}}
    tagger_conf = config['tagger']
    predictor_conf = config['predictor']
    predictor_conf['batch_size'] = args.batch_size if args.batch_size is not None else predictor_conf['batch_size']
    predictor_conf['model_path'] = args.model if args.model is not None else predictor_conf['model_path']
    predictor_conf['nproc'] = args.nproc if args.nproc is not None else predictor_conf['nproc']
    tagger_conf['categories_path'] = args.categories_path if args.categories_path is not None else tagger_conf['categories_path']

    predictor_conf['root_path'] = args.root_path if args.root_path is not None else config['root_path']
    tagger_conf['root_path'] = args.root_path if args.root_path is not None else config['root_path']

    tagger_conf['default_threshold'] = args.default_threshold if args.default_threshold is not None else tagger_conf['default_threshold']
    tagger_conf['custom_threshold_tags'] = args.custom_threshold_tags if args.custom_threshold_tags is not None else tagger_conf['custom_threshold_tags']
    tagger_conf['custom_thresholds'] = args.custom_thresholds if args.custom_thresholds is not None else tagger_conf['custom_thresholds']
    tagger_conf['escape_specials'] = args.escape_specials if args.escape_specials is not None else tagger_conf['escape_specials']
    tagger_conf['remove_underscores'] = args.remove_underscores if args.remove_underscores is not None else tagger_conf['remove_underscores']
    tagger_conf['hide_tags'] = args.hide_tags if args.hide_tags is not None else tagger_conf['hide_tags']
    tagger_conf['include_categories'] = args.include_categories if args.include_categories is not None else tagger_conf['include_categories']
    tagger_conf['prepend_tags'] = args.prepend_tags if args.prepend_tags is not None else tagger_conf['prepend_tags']
    tagger_conf['append_tags'] = args.append_tags if args.append_tags is not None else tagger_conf['append_tags']
    tagger_conf['add_tags'] = args.add_tags if args.add_tags is not None else tagger_conf['add_tags']
    tagger_conf['overwrite_mode'] = args.overwrite_mode if args.overwrite_mode is not None else tagger_conf['overwrite_mode']
    tagger_conf['order'] = args.order if args.order is not None else tagger_conf['order']
    tagger_conf['categories_path'] = args.categories_path if args.categories_path is not None else tagger_conf['categories_path']

    return tagger_conf, predictor_conf


if __name__ == '__main__':
    args = add_args()
    tagger_conf, predictor_conf = create_config(args)
    pred, paths = get_predictions(**predictor_conf)
    tag_config = TagConfig(**tagger_conf)
    tag_handler = TagHandler(tag_config, paths, pred)
    tag_handler.update_all()
