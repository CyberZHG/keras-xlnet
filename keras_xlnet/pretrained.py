import os
import shutil
from collections import namedtuple

from .backend import keras

__all__ = ['PretrainedInfo', 'PretrainedList', 'get_pretrained_paths']


PretrainedInfo = namedtuple('PretrainedInfo', ['url', 'extract_name', 'target_name'])


class PretrainedList(object):

    __test__ = PretrainedInfo(
        'https://github.com/CyberZHG/keras-xlnet/archive/master.zip',
        'keras-xlnet-master',
        'keras-xlnet',
    )

    en_cased_base = PretrainedInfo(
        'https://storage.googleapis.com/xlnet/released_models/cased_L-12_H-768_A-12.zip',
        'xlnet_cased_L-12_H-768_A-12',
        'xlnet_cased_L-12_H-768_A-12',
    )
    en_cased_large = PretrainedInfo(
        'https://storage.googleapis.com/xlnet/released_models/cased_L-24_H-1024_A-16.zip',
        'xlnet_cased_L-24_H-1024_A-16',
        'xlnet_cased_L-24_H-1024_A-16',
    )


def get_pretrained_paths(info):
    path = info
    if isinstance(info, PretrainedInfo):
        path = info.url
    path = keras.utils.get_file(fname=os.path.split(path)[-1], origin=path, extract=True)
    base_part, file_part = os.path.split(path)
    file_part = file_part.split('.')[0]
    if isinstance(info, PretrainedInfo):
        extract_path = os.path.join(base_part, info.extract_name)
        target_path = os.path.join(base_part, info.target_name)
        if not os.path.exists(target_path):
            shutil.move(extract_path, target_path)
        file_part = info.target_name
    path = os.path.join(base_part, file_part)
    PretrainedPaths = namedtuple('PretrainedPaths', ['config', 'model', 'vocab'])
    config_path = os.path.join(path, 'xlnet_config.json')
    model_path = os.path.join(path, 'xlnet_model.ckpt')
    vocab_path = os.path.join(path, 'spiece.model')
    return PretrainedPaths(config_path, model_path, vocab_path)
