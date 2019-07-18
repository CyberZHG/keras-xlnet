from unittest import TestCase

from keras_xlnet.backend import keras
from keras_transformer_xl import RelativeBias
from keras_xlnet import RelativePartialMultiHeadSelfAttention, SegmentBias


class TestAttention(TestCase):

    def test_with_bias(self):
        sequence_length = 5
        previous_length = 15
        units = 12

        token_input = keras.layers.Input(shape=(sequence_length, units))
        content_input = keras.layers.Input(shape=(sequence_length, units))
        memory_input = keras.layers.Input(shape=(previous_length, units))
        segment_mat_input = keras.layers.Input(shape=(sequence_length, previous_length + sequence_length, 2))
        segment_embed_input = keras.layers.Input(shape=(2, units))
        position_input = keras.layers.Input(shape=(previous_length + sequence_length, units))
        permutation_input = keras.layers.Input(shape=(sequence_length, previous_length + sequence_length))

        relative_bias = RelativeBias(units=units)(token_input)
        segment_bias = SegmentBias(units=units)(token_input)

        RelativePartialMultiHeadSelfAttention(units=units, num_head=3, use_bias=True)([
            token_input, content_input, memory_input,
            segment_mat_input, segment_embed_input, position_input,
            relative_bias[0], relative_bias[1], segment_bias,
            permutation_input,
        ])
