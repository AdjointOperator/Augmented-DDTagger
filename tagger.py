import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Tuple
from pathlib import Path

import h5py as h5
import numpy as np
from transformers import CLIPTokenizer  # type: ignore

from utils import invsigmoid, sigmoid


@dataclass
class TagConfig:
    """Configuration for the Tags class

    Attributes:
        root_path (str): Path to the root directory of the dataset
        tags_path (str): Path to the text file containing the tags
        categories_path (str): Path to the json file containing the categories
        custom_threshold_tags (list): List of tags for which to use custom thresholds
        custom_thresholds (list): List of custom thresholds for the tags in custom_threshold_tags
        overwrite_mode (str): How to handle tags that are already present in the dataset. One of add, prepend, append, replace, ignore
        default_threshold (float): Default threshold for tags that are not in the custom_thresholds list
        escape_specials (bool): Whether to escape special characters in the tags
        remove_underscores (bool): Whether to remove underscores from the tags when necessary
        include_categories (tuple): Categories to include in the dataset. All other categories will be ignored. Default is ('General',), other options are ('General', 'Character', 'System')
        add_dir_to_tags (bool): Whether to add the name of the directory to the tags. Not implemented yet.
        add_fname_tags (bool): Whether to add the name of the file to the tags. Not implemented yet.
        hide_tags (tuple): Tags in this list will never be outputted by the tagger
        order (str): How to order the tags. One of alphabetical, probability, probability-reweighted
        prepend_tags (list): Tags to prepend to the list of tags
        append_tags (list): Tags to append to the list of tags
        add_tags (list): Tags to add to the list of tags, only supported for order=alphabetical
        separator (str): Separator to use for tags with multiple words
        token_limit (int): Maximum number of tokens outputted by the tagger, does `NOT` include prepend_tags, append_tags, or add_tags
    """
    root_path: str
    backend: str
    custom_threshold_tags: List[str]
    custom_thresholds: List[float]
    overwrite_mode: str = 'ignore'
    default_threshold: float = 0.6
    escape_specials: bool = True
    remove_underscores: bool = True
    include_categories: Tuple[str] = ('General',)
    add_dir_to_tags: bool = False
    add_fname_tags: bool = False
    hide_tags: Tuple[str] = tuple()
    order: str = 'alphabetical'
    categories_path: str = ''
    tags_path: str = ''
    prepend_tags: List[str] = field(default_factory=list)
    append_tags: List[str] = field(default_factory=list)
    add_tags: List[str] = field(default_factory=list)
    separator: str = ','
    token_limit: int = 70

    tags_raw: np.ndarray = field(init=False)
    tags: np.ndarray = field(init=False)
    thresholds: np.ndarray = field(init=False)
    mask: np.ndarray = field(init=False)
    token_lengths: np.ndarray = field(init=False)

    def __post_init__(self):
        assert self.order in ('alphabetical', 'probability', 'probability-reweighted')
        _custom_thresholds = np.array(self.custom_thresholds)
        assert np.all((0 <= _custom_thresholds) & (_custom_thresholds <= 127)), 'thresholds should be in [0,127] or (0,1)'
        assert len(self.custom_threshold_tags) == len(self.custom_thresholds), 'custom_tags and custom_thresholds should have the same length'
        assert self.overwrite_mode in ('add', 'prepend', 'append', 'replace', 'ignore'), 'overwrite_mode should be one of prepend, append, replace, ignore'
        assert not (self.add_tags and self.order != 'alphabetical'), 'add_tags is only supported for order=alphabetical'
        assert not (self.overwrite_mode == 'add' and self.order != 'alphabetical'), 'overwrite_mode=add is only supported for order=alphabetical'
        self.load_tags()
        self.generate_mask()
        self.generate_thresholds()
        tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained('models/clip_tokenizer')
        tokens = tokenizer(list(self.tags), add_special_tokens=False, padding=False, truncation=False)
        self.token_lengths = np.array([len(token) for token in tokens['input_ids']]) # type: ignore
        print(self.token_lengths[:100])
    def load_tags(self):
        # load tags and do preprocessing
        if self.backend == 'DeepDanbooru':
            self.tags_raw = np.loadtxt(self.tags_path, dtype=str)
        else:
            tags_raw = np.loadtxt(self.tags_path, dtype=str, delimiter=',')[1:]
            self.tags_raw = tags_raw[:, 1]
            self.wd_convnext_raw_tags = tags_raw

        tags = self.tags_raw
        if self.remove_underscores:
            re_under = re.compile(r'(?<=[a-zA-Z\()])_(?=[a-zA-Z\(])')
            tags = [re_under.sub(r' ', tag) for tag in tags]
        if self.escape_specials:
            re_special = re.compile(r'([\\():])')
            tags = [re_special.sub(r'\\\1', tag) for tag in tags]
        self.tags = np.array(tags)

    def generate_mask_deepdanbooru(self):
        with open(self.categories_path, 'r') as f:
            cate_list = json.load(f)
        cate_list = sorted(cate_list, key=lambda k: k['start_index'])
        cate_names = [cate['name'] for cate in cate_list]
        cate_idx = [cate['start_index'] for cate in cate_list]
        assert len(cate_idx) == len(cate_names), 'categories.json is not valid: len(cate_idx) != len(cate_names)'

        cate_idx = cate_idx + [None]
        mask = np.zeros((len(self.tags),), dtype=bool)
        left = 0
        for name, right in zip(cate_names, cate_idx[1:]):
            if name in self.include_categories:
                mask[left:right] = True
            left = right
        self.mask = mask

        # apply hide tags
        if len(self.hide_tags) > 0:
            for tag in self.hide_tags:
                idx = np.where(self.tags_raw == tag)[0]
                assert len(idx) > 0, f'No tag {tag} found for hide tags'
                self.mask[idx] = False

    def generate_mask_ConvNext(self):
        categories = {'General': 0, 'System': 9}
        mask = np.zeros((len(self.tags),), dtype=bool)
        for name in self.include_categories:
            mask[self.wd_convnext_raw_tags[:, 2].astype(np.int32) == categories[name]] = True
        self.mask = mask

        # apply hide tags
        if len(self.hide_tags) > 0:
            for tag in self.hide_tags:
                idx = np.where(self.tags_raw == tag)[0]
                assert len(idx) > 0, f'No tag {tag} found for hide tags'
                self.mask[idx] = False

    def generate_mask(self):
        if self.backend == 'DeepDanbooru':
            self.generate_mask_deepdanbooru()
        else:
            self.generate_mask_ConvNext()

    def generate_thresholds(self):
        # generate thresholds
        th = self.default_threshold
        th = invsigmoid(th) if 0 < th < 1 else th
        thresholds = np.full((len(self.tags),), th, dtype=np.float16)
        thresholds[~self.mask] = 127

        # apply custom thresholds
        if len(self.custom_threshold_tags) > 0:
            for tag, m_thres in zip(self.custom_threshold_tags, self.custom_thresholds):
                idx = np.where(self.tags_raw == tag)[0]
                assert len(idx) > 0, f'No tag {tag} found for custom threshold tags'
                m_thres = invsigmoid(m_thres) if 0 < m_thres < 1 else m_thres
                thresholds[idx] = m_thres
        self.thresholds = thresholds


class TagHandler:
    def __init__(self, tag_config: TagConfig, paths: List[Path] | List[str], preds: np.ndarray):
        """Master class for handling tags

        Args:
            tag_config (TagConfig)
            paths (List[str]): List of paths to the images
            preds (np.ndarray): Array of predictions
        """
        self.config = tag_config
        self.backend = tag_config.backend
        self.tags = tag_config.tags
        self.thresholds = tag_config.thresholds
        self.paths = [tag_config.root_path / Path(path) for path in paths]
        self.preds = preds
        self.separator = tag_config.separator

    def trim_tags(self, tags: List[str], preds: np.ndarray, thres: np.ndarray) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Trim new_tags to fit the token limit

        Args:
            tags
            preds: Original predictions. Sigmoid is not applied
            thres: Thresholds for each tag. Sigmoid is not applied

        Returns:
            trimmed tags
        """
        if self.config.token_limit < 0:
            return tags, preds, thres
        t = sigmoid(thres)
        p = sigmoid(preds)
        order = np.argsort((p - t) / (1 - t))[::-1]
        token_sizes = self.config.token_lengths[order]
        # Take the separator size into account
        token_sizes[1:] += len(re.findall(r'[^\s]', self.separator))
        cum_size = np.cumsum(token_sizes)
        cut = np.searchsorted(cum_size, self.config.token_limit - 2, side='left')
        mask = order[:cut]
        return np.array(tags)[mask].tolist(), preds[mask], thres[mask]

    def sort_tags(self, tags, preds: np.ndarray, thres: np.ndarray):
        """Sort tags according to the config. For probability-reweighted, the order is based on (p-thres)/(1-thres).

        Args:
            tags
            preds (np.ndarray): Original predictions. Sigmoid is not applied
            thres (np.ndarray): Thresholds for each tag. Sigmoid is not applied

        Raises:
            Exception: _description_

        Returns:
            sorted tags
        """
        if self.config.order == 'alphabetical':
            tags = sorted(tags)
        else:
            if self.config.order == 'probability':
                order = np.argsort(preds)[::-1]
            elif self.config.order == 'probability-reweighted':
                t = sigmoid(thres)
                p = sigmoid(preds)
                order = np.argsort((p - t) / (1 - t))[::-1]
            else:
                raise Exception('Unknown ordering mode. WTF have you done?')
            tags = np.array(tags)[order].tolist()
        return tags

    def process_tag(self, prepend: List[str], append: List[str], new_tags: List[str], old_tags: List[str] | None, preds: np.ndarray, thres: np.ndarray):
        """Interfacing function to apply all the tag processing steps

        Args:
            prepend: tags to prepend
            append: tags to append
            new_tags: List of tags predicted by the model
            old_tags: List of tags already present in the image
            preds (np.ndarray): Raw predictions given by the model
            thres (np.ndarray): Raw thresholds

        Returns:
            processed tags
        """
        new_tags, preds, thres = self.trim_tags(new_tags, preds, thres)
        if self.config.overwrite_mode == 'replace' or old_tags is None:
            tags = self.sort_tags(new_tags, preds, thres)
        elif self.config.overwrite_mode == 'prepend':
            tags = self.sort_tags(new_tags, preds, thres) + old_tags
        elif self.config.overwrite_mode == 'append':
            tags = old_tags + self.sort_tags(new_tags, preds, thres)
        elif self.config.overwrite_mode == 'ignore':
            tags = old_tags

        elif self.config.overwrite_mode == 'add':
            tags = list(set(old_tags + new_tags))
            tags = self.sort_tags(tags, preds, thres)
        else:
            raise Exception('Unknown overwrite mode. WTF have you done?')
        tags = prepend + tags + append
        return self.separator.join(tags)

    def update_tags(self, idx):
        """Update tags for a single image

        Args:
            idx: Index of the image to be updated. The corresponding path and prediction are taken from self.paths and self.preds
        """

        path = self.paths[idx]
        pred_all_tags = self.preds[idx]
        prepend = self.config.prepend_tags
        append = self.config.append_tags
        add_tags = self.config.add_tags

        txt_path = Path(path).with_suffix('.txt')
        if os.path.exists(txt_path):
            with open(txt_path) as f:
                old_tags = f.read().split(self.separator)
        else:
            old_tags = None
        mask = pred_all_tags > self.thresholds
        new_tags = self.tags[mask].tolist()
        preds = pred_all_tags[mask]
        thres = self.thresholds[mask]
        if add_tags:
            new_tags = list(set(new_tags + add_tags))
        tag_str = self.process_tag(prepend, append, new_tags, old_tags, preds, thres)
        with open(txt_path, 'w') as f:
            f.write(tag_str)

    def update_all(self):
        """Update tags for all images"""
        for i in range(len(self.paths)):
            self.update_tags(i)
