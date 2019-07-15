from unittest import TestCase

import numpy as np

from keras_xlnet.backend import keras, EAGER_MODE
from keras_xlnet.backend import backend as K
from keras_xlnet import PermutationMask


class TestPermutation(TestCase):

    def test_permutation_mask(self):
        inputs_segment = keras.layers.Input(shape=(6, 3))
        inputs_memory = keras.layers.Input(shape=(3, 3))
        inputs = [inputs_segment, inputs_memory]
        outputs = PermutationMask()(inputs, training=K.learning_phase())
        learning_phase = K.learning_phase()
        if EAGER_MODE:
            learning_phase = K.symbolic_learning_phase()
        func = K.function(inputs + [learning_phase], outputs)
        inputs = [np.zeros((2, 6, 3)), np.zeros((2, 3, 3)), 0]
        if EAGER_MODE:
            inputs[-1] = False
        outputs = func(inputs)
        self.assertEqual((2, 6, 9), outputs[0].shape)
        self.assertEqual((2, 6, 9), outputs[1].shape)
        expect = [
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]
        self.assertEqual(expect, outputs[0][0].tolist())
        expect = [
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        ]
        self.assertEqual(expect, outputs[1][0].tolist())
        inputs = [np.zeros((2, 6, 3)), np.zeros((2, 3, 3)), 1]
        if EAGER_MODE:
            inputs[-1] = True
        outputs = func(inputs)
        self.assertEqual((2, 6, 9), outputs[0].shape)
        self.assertEqual((2, 6, 9), outputs[1].shape)
        counts = [0] * 6
        for i in range(6):
            counts[int(outputs[0][0][i].sum()) - 4] += 1
        self.assertEqual(6, sum(counts))
        expect = outputs[0][0].tolist()
        for i in range(6):
            expect[i][i + 3] = 0.0
        self.assertEqual(expect, outputs[1][0].tolist())
