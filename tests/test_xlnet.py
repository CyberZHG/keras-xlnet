import os
import tempfile
from unittest import TestCase

import numpy as np

from keras_xlnet.backend import keras
from keras_xlnet import build_xlnet, set_custom_objects


class TestXLNet(TestCase):

    def test_build(self):
        model = build_xlnet(
            units=6,
            num_token=31,
            num_block=2,
            num_head=2,
            hidden_dim=12,
            batch_size=2,
            memory_len=5,
            target_len=5,
            dropout=0.1,
            attention_dropout=0.1,
        )
        set_custom_objects()
        model_path = os.path.join(tempfile.gettempdir(), 'test_xlnet_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path)
        model.summary()