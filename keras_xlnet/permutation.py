import tensorflow as tf

from .backend import keras
from .backend import backend as K

__all__ = ['PermutationMask']


class PermutationMask(keras.layers.Layer):
    """Generate random permutation masks during training.

    # Input shape
        Inputs, 3D tensor with shape: `(batch_size, seq_len, units)`.
        Memory, 3D tensor with shape: `(batch_size, mem_len, units)`.

    # Output shape
        Content mask, 3D tensor with shape: `(batch_size, seq_len, mem_len + seq_len)`.
        Query mask, 3D tensor with shape: `(batch_size, seq_len, mem_len + seq_len)`.

    # References
        - [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/pdf/1906.08237)
    """

    def __init__(self, **kwargs):
        super(PermutationMask, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        input_shape, memory_shape = input_shape
        seq_len = input_shape[1]
        mem_len = memory_shape[1]
        key_len = None
        if mem_len is not None and seq_len is not None:
            key_len = mem_len + seq_len
        return [(input_shape[0], seq_len, key_len)] * 2

    def compute_mask(self, inputs, mask=None):
        return [None, None]

    def call(self, inputs, training=None, **kwargs):
        inputs, memory = inputs
        batch_size = K.shape(inputs)[0]
        seq_len = K.shape(inputs)[1]
        mem_mask = K.tile(K.ones_like(memory[:, :, :1], dtype=K.floatx()), [1, 1, seq_len])

        # Build content mask with random permutation
        ranges = K.tile(K.expand_dims(K.arange(0, seq_len), axis=-1), [1, batch_size])
        shuffled = K.in_train_phase(tf.random.shuffle(ranges), ranges, training)
        ranges = K.expand_dims(K.permute_dimensions(ranges, [1, 0]), axis=-1)
        shuffled = K.expand_dims(K.permute_dimensions(shuffled, [1, 0]), axis=1)
        content_mask = K.cast(ranges <= shuffled, dtype=K.floatx())

        # Build query mask based on content mask
        ranges = K.arange(0, seq_len)
        eye = K.equal(K.expand_dims(ranges, axis=0), K.expand_dims(ranges, axis=-1))
        eye = K.expand_dims(K.cast(eye, dtype=K.floatx()), axis=0)
        query_mask = content_mask * (1.0 - eye)

        content_mask = K.concatenate([mem_mask, content_mask], axis=1)
        query_mask = K.concatenate([mem_mask, query_mask], axis=1)
        return [
            K.permute_dimensions(content_mask, [0, 2, 1]),
            K.permute_dimensions(query_mask, [0, 2, 1]),
        ]
