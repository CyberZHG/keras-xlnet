import os
import tempfile
from unittest import TestCase

import numpy as np

from keras_xlnet.backend import keras
from keras_xlnet import load_trained_model_from_checkpoint


class TestLoader(TestCase):

    def test_load_not_training(self):
        current_path = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(current_path, 'test_checkpoint')
        model = load_trained_model_from_checkpoint(
            config_path=os.path.join(checkpoint_path, 'xlnet_config.json'),
            checkpoint_path=os.path.join(checkpoint_path, 'xlnet_model.ckpt'),
            batch_size=2,
            memory_len=5,
            target_len=5,
            in_train_phase=False,
        )
        model.summary()

        def _load_numpy(name):
            return np.load(os.path.join(checkpoint_path, name + '.npy'))

    def test_load_training(self):
        current_path = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(current_path, 'test_checkpoint')
        model = load_trained_model_from_checkpoint(
            config_path=os.path.join(checkpoint_path, 'xlnet_config.json'),
            checkpoint_path=os.path.join(checkpoint_path, 'xlnet_model.ckpt'),
            batch_size=2,
            memory_len=5,
            target_len=5,
            in_train_phase=True,
        )
        model.summary()

        def _load_numpy(name):
            return np.load(os.path.join(checkpoint_path, name + '.npy'))
