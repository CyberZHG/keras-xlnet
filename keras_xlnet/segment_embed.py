from .backend import keras
from .backend import backend as K

__all__ = ['RelativeSegmentEmbedding']


class RelativeSegmentEmbedding(keras.layers.Embedding):
    """Embeddings for relative segments.

    # Input shape
        Segment IDs, 2D tensor with shape: `(batch_size, seq_len)`.
        Memory, 3D tensor with shape: `(batch_size, mem_len, units)`.

    # Output shape
        4D tensor with shape: `(batch_size, seq_len, mem_len + seq_len, 2)`.
        2D tensor with shape: `(2, units)`.

    # References
        - [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/pdf/1906.08237)
    """

    def __init__(self, units, **kwargs):
        kwargs['input_dim'] = 2
        kwargs['output_dim'] = units
        super(RelativeSegmentEmbedding, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units

    def compute_output_shape(self, input_shape):
        segment_shape, memory_shape = input_shape
        mem_len = None
        if segment_shape[1] is not None and memory_shape[1] is not None:
            mem_len = segment_shape[1] + memory_shape[1]
        return [(segment_shape[0], segment_shape[1], mem_len, 2), (2, memory_shape[2])]

    def compute_mask(self, inputs, mask=None):
        return [None, None]

    def call(self, inputs):
        segment, memory = inputs
        full = K.concatenate([K.zeros_like(memory[:, :, 0]), segment], axis=1)
        relative = K.not_equal(K.expand_dims(segment, axis=-1), K.expand_dims(full, axis=1))
        relative = K.one_hot(K.cast(relative, 'uint8'), 2)
        return [relative, K.identity(self.embeddings)]

    def get_config(self):
        config = {
            'units': self.units,
        }
        base_config = super(RelativeSegmentEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
