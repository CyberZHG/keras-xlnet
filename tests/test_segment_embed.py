from unittest import TestCase

import numpy as np

from keras_xlnet.backend import keras
from keras_xlnet.backend import backend as K
from keras_xlnet import RelativeSegmentEmbedding


class TestSegmentEmbed(TestCase):

    def test_relative_segment(self):
        inputs_segment = keras.layers.Input(shape=(6,))
        inputs_memory = keras.layers.Input(shape=(3, 3))
        inputs = [inputs_segment, inputs_memory]
        embed_layer = RelativeSegmentEmbedding(units=3)
        outputs = embed_layer(inputs)
        weights = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ])
        embed_layer.set_weights([weights])
        func = K.function(inputs, outputs)
        query = [0, 1, 2, 2, 1, 0]
        key = [0, 0, 0] + query
        inputs = [np.array([query]), np.zeros((1, 3, 3))]
        outputs = func(inputs)
        self.assertEqual((1, 6, 9, 2), outputs[0].shape)
        for i in range(6):
            for j in range(9):
                if query[i] == key[j]:
                    self.assertEqual([1, 0], outputs[0][0, i, j].tolist())
                else:
                    self.assertEqual([0, 1], outputs[0][0, i, j].tolist())
        self.assertTrue(np.allclose(weights, outputs[1]))
