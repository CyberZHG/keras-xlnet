import os
import tempfile
from unittest import TestCase

import numpy as np

from keras_xlnet.backend import keras
from keras_xlnet import build_xlnet, get_custom_objects, ATTENTION_TYPE_BI


class TestXLNet(TestCase):

    def test_build_training(self):
        model = build_xlnet(
            units=6,
            training=True,
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
        model_path = os.path.join(tempfile.gettempdir(), 'test_xlnet_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects=get_custom_objects())
        model.summary()
        try:
            current_path = os.path.dirname(os.path.abspath(__file__))
            visual_path = os.path.join(current_path, 'test_build_training.jpg')
            keras.utils.vis_utils.plot_model(model, visual_path, show_shapes=True)
        except Exception as e:
            pass

    def test_build_not_training(self):
        model = build_xlnet(
            units=6,
            training=False,
            num_token=31,
            num_block=2,
            num_head=2,
            hidden_dim=12,
            batch_size=2,
            memory_len=5,
            target_len=5,
            dropout=0.1,
            attention_dropout=0.1,
            attention_type=ATTENTION_TYPE_BI,
        )
        model_path = os.path.join(tempfile.gettempdir(), 'test_xlnet_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects=get_custom_objects())
        model.summary()
        try:
            current_path = os.path.dirname(os.path.abspath(__file__))
            visual_path = os.path.join(current_path, 'test_build_not_training.jpg')
            keras.utils.vis_utils.plot_model(model, visual_path, show_shapes=True)
        except Exception as e:
            pass

    def test_fit_batch_changes(self):
        model = build_xlnet(
            units=4,
            training=False,
            num_token=2,
            num_block=1,
            num_head=1,
            hidden_dim=4,
            batch_size=4,
            memory_len=0,
            target_len=5,
            permute=True,
            attention_type=ATTENTION_TYPE_BI,
            clamp_len=100,
        )
        model.compile('adam', 'mse')
        model.summary()

        def gen():
            while True:
                yield [np.ones((4, 5)), np.zeros((4, 5)), np.zeros((4, 1))], np.zeros((4, 5, 4))
                yield [np.ones((3, 5)), np.zeros((3, 5)), np.zeros((3, 1))], np.zeros((3, 5, 4))
        model.fit_generator(gen(), steps_per_epoch=2)
