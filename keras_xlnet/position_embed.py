from .backend import keras
from .backend import backend as K

__all__ = ['PositionalEmbedding']


class PositionalEmbedding(keras.layers.Layer):
    """Positional embeddings.

    # Arguments
        output_dim: int >= 0. Dimension of the embedding. Should be even.
        clamp_len: int > 0. Maximum length of positions.
        directional: boolean. Whether the input is directional.

    # Input shape
        2D tensor with shape: `(batch_size, sequence_length)`.
        3D tensor with shape: `(batch_size, memory_length, output_dim)`.

    # Output shape
        3D tensor with shape: `(batch_size, sequence_length + memory_length, output_dim)`.

    # References
        - [Transformer-XL](https://arxiv.org/pdf/1901.02860.pdf)
    """

    def __init__(self, output_dim, clamp_len=None, directional=True, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.supports_masking = True
        self.output_dim = output_dim
        self.clamp_len = clamp_len
        self.directional = directional

    def compute_output_shape(self, input_shape):
        input_shape, memory_shape = input_shape
        mem_len = None
        if input_shape[1] is not None and memory_shape[1] is not None:
            mem_len = input_shape[1] + memory_shape[1]
        return input_shape[0],  mem_len, memory_shape[2]

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        return mask[0]

    def call(self, inputs, **kwargs):
        q_len, m_len = K.shape(inputs[0])[1], K.shape(inputs[1])[1]
        k_len = q_len + m_len
        start, stop = k_len, -1
        if not self.directional:
            stop = -q_len
        inputs = K.tile(
            K.expand_dims(K.arange(start, stop, -1, dtype=K.floatx()), axis=0),
            [K.shape(inputs[0])[0], 1],
        )
        if self.clamp_len is not None:
            inputs = K.clip(inputs, min_value=0, max_value=self.clamp_len)
        inputs = K.expand_dims(inputs, axis=-1)
        output_dim = K.cast(self.output_dim, K.floatx())
        ranges = K.expand_dims(K.arange(0.0, self.output_dim, 2.0), axis=0) / output_dim
        inverse = 1.0 / K.pow(10000.0, ranges)
        positions = inputs * inverse
        return K.concatenate([K.sin(positions), K.cos(positions)], axis=-1)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'clamp_len': self.clamp_len,
            'directional': self.directional,
        }
        base_config = super(PositionalEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
