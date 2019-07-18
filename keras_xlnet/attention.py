import tensorflow as tf
from .backend import keras, activations, initializers, regularizers, constraints, TF_KERAS
from .backend import backend as K

__all__ = ['RelativePartialMultiHeadSelfAttention']


class RelativePartialMultiHeadSelfAttention(keras.layers.Layer):
    """Positional embeddings.

    # Arguments
        units: int >= 0. Dimensions of all tensors.
        num_head: int >= 0. Number of heads. Should divide units.
        use_bias: Boolean. Whether to use bias term.
        attention_dropout: 0.0 < float < 1.0. Dropout rate for attention weights.

    # Input shape
        Input feature, 3D tensor with shape: `(batch_size, sequence_length, units)`.
        Content feature, 3D tensor with shape: `(batch_size, sequence_length, units)`.
        Memory feature, 3D tensor with shape: `(batch_size, previous_length, units)`.
        Segment matrix, 4D tensor with shape: `(batch_size, sequence_length, previous_length + sequence_length, 2)`.
        Segment embedding, 2D tensor with shape: `(2, units)`.
        Positional embedding, 3D tensor with shape: `(batch_size, previous_length + sequence_length, units)`.
        Context bias, 1D tensor with shape: `(units,)`.
        Relative bias, 1D tensor with shape: `(units,)`.
        Segment bias, 1D tensor with shape: `(units,)`.
        Permutation mask, 3D tensor with shape: `(batch_size, sequence_length, previous_length + sequence_length)`.

    # Output shape
        3D tensor with shape: `(batch_size, sequence_length, units)`.

    # References
        - [Transformer-XL](https://arxiv.org/pdf/1901.02860.pdf)
    """

    def __init__(self,
                 units,
                 num_head,
                 activation=None,
                 use_bias=False,
                 attention_dropout=0.0,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(RelativePartialMultiHeadSelfAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.num_head = num_head
        self.units_head = units // num_head
        self.activation = activation
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.attention_dropout = attention_dropout
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.kernel, self.bias = None, None
        self.att_drop_layer = None

    def compute_mask(self, inputs, mask=None):
        return mask

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(self.units, self.units * 5),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='kernel',
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units * 5,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='bias',
            )

        if 0.0 < self.attention_dropout < 1.0:
            self.att_drop_layer = keras.layers.Dropout(self.attention_dropout)
        super(RelativePartialMultiHeadSelfAttention, self).build(input_shape)

    def _reshape_to_batches(self, x):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        x = K.reshape(x, (batch_size, seq_len, self.num_head, self.units_head))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batch_size * self.num_head, seq_len, self.units_head))

    def _reshape_from_batches(self, x):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        x = K.reshape(x, (batch_size // self.num_head, self.num_head, seq_len, feature_dim))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batch_size // self.num_head, seq_len, feature_dim * self.num_head))

    def _reshape_mask(self, mask):
        seq_len = K.shape(mask)[1]
        mask = K.expand_dims(mask, axis=1)
        mask = K.tile(mask, [1, self.num_head, 1])
        return K.reshape(mask, (-1, seq_len))

    @staticmethod
    def _relative_shift(x):
        batch_size, q_len, k_len = K.shape(x)[0], K.shape(x)[1], K.shape(x)[2]
        x = tf.pad(x, [[0, 0], [0, 0], [1, 0]])               # (batch * n_head, seq_len, prev_len + seq_len + 1)
        x = K.reshape(x, (batch_size, k_len + 1, q_len))      # (batch * n_head, prev_len + seq_len + 1, seq_len)
        x = x[:, 1:, :]                                       # (batch * n_head, prev_len + seq_len, seq_len)
        return K.reshape(x, (batch_size, q_len, k_len))       # (batch * n_head, seq_len, prev_len + seq_len)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, inputs, mask=None, training=None):
        (inputs, content, memories,
         segment_mat, segment_embed, relatives,
         bias_context, bias_relative, bias_segment,
         permutation) = inputs
        full = K.concatenate([memories, content], axis=1)     # (batch, prev_len + seq_len, units)

        kernel_q = self.kernel[:, :self.units]
        kernel_kv = self.kernel[:, self.units:self.units * 3]
        kernel_r = self.kernel[:, self.units * 3:self.units * 4]
        kernel_o = self.kernel[:, self.units * 4:self.units * 5]

        bias_q, bias_kv, bias_r, bias_o = (None,) * 4
        if self.use_bias:
            bias_q = self.bias[:self.units]
            bias_kv = self.bias[self.units:self.units * 3]
            bias_r = self.bias[self.units * 3:self.units * 4]
            bias_o = self.bias[self.units * 4:self.units * 5]

        w_q = K.dot(inputs, kernel_q)                    # (batch, seq_len, units)
        w_kv = K.dot(full, kernel_kv)                    # (batch, prev_len + seq_len, units * 2)
        w_r = K.dot(relatives, kernel_r)                 # (batch, prev_len + seq_len, units)
        if self.use_bias:
            w_q = K.bias_add(w_q, bias_q)
            w_kv = K.bias_add(w_kv, bias_kv)
            w_r = K.bias_add(w_r, bias_r)
        if self.activation is not None:
            w_q = self.activation(w_q)
            w_kv = self.activation(w_kv)
            w_r = self.activation(w_r)

        w_k = w_kv[:, :, :self.units]                    # (batch, prev_len + seq_len, units)
        w_v = w_kv[:, :, self.units:]                    # (batch, prev_len + seq_len, units)
        batch_size, q_len, k_len = K.shape(inputs)[0], K.shape(w_q)[1], K.shape(w_k)[1]

        w_qc = K.bias_add(w_q, bias_context)
        w_qc = self._reshape_to_batches(w_qc)            # (batch * n_head, seq_len, units_head)
        w_k = self._reshape_to_batches(w_k)              # (batch * n_head, prev_len + seq_len, units_head)
        a_context = K.batch_dot(w_qc, w_k, axes=2)       # (batch * n_head, seq_len, prev_len + seq_len)

        w_qr = K.bias_add(w_q, bias_relative)
        w_qr = self._reshape_to_batches(w_qr)            # (batch * n_head, seq_len, units_head)
        w_r = self._reshape_to_batches(w_r)              # (batch * n_head, prev_len + seq_len, units_head)
        a_relative = K.batch_dot(w_qr, w_r, axes=2)      # (batch * n_head, seq_len, prev_len + seq_len)
        a_relative = self._relative_shift(a_relative)    # (batch * n_head, seq_len, prev_len + seq_len)

        w_qs = K.bias_add(w_q, bias_segment)
        w_qs = K.reshape(w_qs, (-1, q_len, self.num_head, self.units_head))
        w_qs = K.permute_dimensions(w_qs, (2, 0, 1, 3))               # (n_head, batch, seq_len, units_head)
        segment_embed = K.reshape(K.transpose(segment_embed), (self.num_head, 1, self.units_head, 2))
        segment_embed = K.tile(segment_embed, (1, batch_size, 1, 1))
        a_segment = K.batch_dot(w_qs, segment_embed, axes=(3, 2))     # (n_head, batch, seq_len, 2)
        a_segment = K.permute_dimensions(a_segment, (1, 2, 3, 0))     # (batch, seq_len, 2, n_head)
        a_segment = K.batch_dot(segment_mat, a_segment, axes=(3, 2))  # (batch, seq_len, prev_len + seq_len, n_head)
        a_segment = K.reshape(K.permute_dimensions(a_segment, (0, 3, 1, 2)), (-1, q_len, k_len))

        att = (a_context + a_relative + a_segment) / K.sqrt(K.constant(self.units_head, dtype=K.floatx()))
        exp = K.exp(att - K.max(att, axis=-1, keepdims=True))

        permutation = K.tile(K.expand_dims(permutation, axis=1), [1, self.num_head, 1, 1])
        permutation = K.reshape(permutation, (-1, q_len, k_len))
        exp *= permutation
        if mask is not None and mask[0] is not None:
            mask = K.cast(mask[0], K.floatx())
            mask = K.concatenate([K.ones_like(memories[:, :, 0]), mask], axis=1)
            exp *= K.expand_dims(self._reshape_mask(mask), axis=1)

        att = exp / K.sum(exp, axis=-1, keepdims=True)
        if self.att_drop_layer is not None:
            att = self.att_drop_layer(att, training=training)
        w_v = self._reshape_to_batches(w_v)                   # (batch * n_head, prev_len + seq_len, units_head)
        w_o = K.batch_dot(att, w_v)                           # (batch * n_head, seq_len, units_head)

        w_o = self._reshape_from_batches(w_o)                 # (batch, seq_len, units)
        w_o = K.dot(w_o, kernel_o)                       # (batch, seq_len, units)
        if self.use_bias:
            w_o = K.bias_add(w_o, bias_o)
        if self.activation is not None:
            w_o = self.activation(w_o)

        if TF_KERAS:
            # Add shape information to tensor when using `tf.keras`
            input_shape = K.int_shape(inputs)
            if input_shape[1] is not None:
                w_o = K.reshape(w_o, (-1,) + input_shape[1:])
        return w_o

    def get_config(self):
        config = {
            'units': self.units,
            'num_head': self.num_head,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'attention_dropout': self.attention_dropout,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
        }
        base_config = super(RelativePartialMultiHeadSelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
