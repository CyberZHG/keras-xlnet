import os
import tempfile
from unittest import TestCase

import numpy as np

from keras_xlnet.backend import keras
from keras_xlnet.backend import backend as K
from keras_xlnet import load_trained_model_from_checkpoint


class TestLoader(TestCase):

    @staticmethod
    def _update_memory(model, name, weight):
        layer = model.get_layer(name)
        memory = K.get_value(layer.weights[0])
        memory[:weight.shape[0], -weight.shape[1]:, :] = weight
        layer.set_weights([memory])

    @staticmethod
    def _get_memory(model, name, length):
        layer = model.get_layer(name)
        memory = K.get_value(layer.weights[0])
        return memory[:, -length:, :]

    def test_load_not_training(self):
        current_path = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(current_path, 'test_checkpoint_tune')
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

        input_ids = _load_numpy('input_ids')
        seg_ids = _load_numpy('seg_ids')
        mems_0 = _load_numpy('mems_0')
        mems_1 = _load_numpy('mems_1')
        tune_output = _load_numpy('tune_output')
        tune_new_mems_0 = _load_numpy('tune_new_mems_0')
        tune_new_mems_1 = _load_numpy('tune_new_mems_1')

        inputs = [input_ids, seg_ids, np.ones((2, 1)) * 5]
        self._update_memory(model, 'Memory-0', mems_0)
        self._update_memory(model, 'Memory-1', mems_1)
        output = model.predict_on_batch(inputs)
        self.assertTrue(np.allclose(tune_new_mems_0, self._get_memory(model, 'Memory-0', 5), atol=1e-6))
        self.assertTrue(np.allclose(tune_new_mems_1, self._get_memory(model, 'Memory-1', 5), atol=1e-6))
        self.assertTrue(np.allclose(tune_output, output, atol=1e-6))

    def _test_load_training(self):
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
