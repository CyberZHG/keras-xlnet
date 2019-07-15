from .backend import keras, initializers, regularizers, constraints
from .backend import backend as K

__all__ = ['MaskEmbedding']


class MaskEmbedding(keras.layers.Layer):
    """Embedding for query tokens.

    # Arguments
        units: int >= 0. Number of hidden units.

    # Input shape
        Token embeddings, 3D tensor with shape: `(batch_size, seq_len, units)`.
        Query input, 2D tensor with shape: `(batch_size, seq_len)`.

    # Output shape
        3D tensor with shape: `(batch_size, seq_len, units)`.

    # References
        - [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/pdf/1906.08237)
    """

    def __init__(self,
                 units,
                 initializer='uniform',
                 regularizer=None,
                 constraint=None,
                 **kwargs):
        super(MaskEmbedding, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.initializer = initializers.get(initializer)
        self.regularizer = regularizers.get(regularizer)
        self.constraint = constraints.get(constraint)

        self.embeddings = None

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(1, 1, self.units),
            initializer=self.initializer,
            regularizer=self.regularizer,
            constraint=self.constraint,
            name='embeddings',
        )
        super(MaskEmbedding, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def compute_mask(self, inputs, mask=None):
        output_mask = None
        if mask is not None:
            output_mask = mask[0]
        return output_mask

    def call(self, inputs, **kwargs):
        token_embed, query = inputs
        query = K.expand_dims(K.cast(query, dtype=K.floatx()), axis=-1)
        return query * self.embeddings + (1.0 - query) * token_embed

    def get_config(self):
        config = {
            'units': self.units,
            'initializer': initializers.serialize(self.initializer),
            'regularizer': regularizers.serialize(self.regularizer),
            'constraint': constraints.serialize(self.constraint),
        }
        base_config = super(MaskEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
