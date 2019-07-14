from .backend import keras
from .backend import backend as K

from keras_embed_sim import EmbeddingRet, EmbeddingSim
from keras_transformer import gelu
from keras_transformer_xl import RelativeBias, Memory, PositionalEmbedding
from keras_layer_normalization import LayerNormalization
from keras_position_wise_feed_forward import FeedForward

from .segment_bias import SegmentBias
from .segment_embed import RelativeSegmentEmbedding
from .attention import RelativePartialMultiHeadSelfAttention as Attention

__all__ = [
    'get_custom_objects', 'set_custom_objects', 'build_xlnet',
]


def get_custom_objects() -> dict:
    return {
        'gelu': gelu,
        'EmbeddingRet': EmbeddingRet,
        'EmbeddingSim': EmbeddingSim,
        'PositionalEmbedding': PositionalEmbedding,
        'RelativeBias': RelativeBias,
        'SegmentBias': SegmentBias,
        'RelativeSegmentEmbedding': RelativeSegmentEmbedding,
        'Memory': Memory,
        'LayerNormalization': LayerNormalization,
        'RelativePartialMultiHeadSelfAttention': Attention,
        'FeedForward': FeedForward,
    }


def set_custom_objects() -> None:
    for key, val in get_custom_objects().items():
        keras.utils.get_custom_objects()[key] = val


def build_xlnet(units,
                training,
                num_token,
                num_block,
                num_head,
                hidden_dim,
                batch_size,
                memory_len,
                target_len,
                dropout=0.0,
                attention_dropout=0.0,
                clamp_len=None,
                shared_biases=True):
    """Build XLNet.

    :param units: Hidden dimensions throughout the model.
    :param training: Whether in training mode.
    :param num_token: Number of distinct tokens.
    :param num_block: Number of basic encoder blocks.
    :param num_head: Number of heads for attention.
    :param hidden_dim: Dimension inside position-wise feed-forward layer.
    :param batch_size: Maximum batch size.
    :param memory_len: The maximum length of memories.
    :param target_len: The length of prediction block.
    :param dropout: General dropout rate.
    :param attention_dropout: Dropout rate inside attention layer.
    :param clamp_len: The maximum value of relative position.
    :param shared_biases: Whether to use the same biases for all layers.
    :return: The built model.
    """
    token_input = keras.layers.Input(
        shape=(target_len,),
        name='Input-Token',
    )
    seg_input = keras.layers.Input(
        shape=(target_len,),
        name='Input-Segment',
    )
    memory_length_input = keras.layers.Input(
        shape=(1,),
        name='Input-Memory-Length',
    )
    perm_input = keras.layers.Input(
        shape=(target_len, target_len),
        name='Input-Perm',
    )
    token_embed, embed_weights = EmbeddingRet(
        input_dim=num_token,
        output_dim=units,
        mask_zero=True,
        name='Embed-Token',
    )(token_input)
    if 0.0 < dropout < 1.0:
        token_embed = keras.layers.Dropout(
            rate=dropout,
            name='Embed-Token-Dropout'
        )(token_embed)

    memories = []
    for i in range(num_block):
        memories.append(Memory(
            batch_size=batch_size,
            memory_len=memory_len,
            target_len=target_len,
            output_dim=units,
            name='Memory-Content-{}'.format(i + 1),
        )([token_embed, memory_length_input]))

    pos_embed = PositionalEmbedding(
        output_dim=units,
        clamp_len=clamp_len,
        name='Embed-Pos',
    )([token_embed, memories[0]])

    if shared_biases:
        relative_biases = RelativeBias(
            units,
            name='Relative-Bias',
        )(memories[0])
        segment_biases = SegmentBias(
            units,
            name='Segment-Bias',
        )(memories[0])
    else:
        relative_biases, segment_biases = [], []
        for i in range(num_block):
            relative_biases.append(RelativeBias(
                units,
                name='Relative-Bias-{}'.format(i + 1),
            )(memories[i]))
            segment_biases.append(RelativeBias(
                units,
                name='Segment-Bias-{}'.format(i + 1),
            )(memories[i]))

    content_output, query_output = token_embed, token_embed
    for i in range(num_block):
        segment_embed = RelativeSegmentEmbedding(
            units=units,
            name='Embed-Segment-{}'.format(i + 1),
        )([seg_input, memories[i]])

        def _build_block(name, query, content):
            attention_input = query
            if shared_biases:
                context_bias, relative_bias, segment_bias = relative_biases[0], relative_biases[1], segment_biases
            else:
                context_bias, relative_bias  = relative_biases[i][0], relative_biases[i][1]
                segment_bias = segment_biases[i]
            _output = Attention(
                units=units,
                num_head=num_head,
                use_bias=False,
                attention_dropout=attention_dropout,
                name='Attention-{}-{}'.format(name, i + 1),
            )([
                query, content, memories[i],
                segment_embed, pos_embed,
                context_bias, relative_bias, segment_bias,
            ])
            if 0.0 < dropout < 1.0:
                _output = keras.layers.Dropout(
                    rate=dropout,
                    name='Attention-{}-Dropout-{}'.format(name, i + 1),
                )(_output)
            _output = keras.layers.Add(
                name='Attention-{}-Residual-{}'.format(name, i + 1),
            )([attention_input, _output])
            _output = LayerNormalization(
                name='Attention-{}-Norm-{}'.format(name, i + 1)
            )(_output)

            feed_forward_input = _output
            _output = FeedForward(
                units=hidden_dim,
                dropout_rate=dropout,
                name='FeedForward-{}-{}'.format(name, i + 1),
            )(_output)
            if 0.0 < dropout < 1.0:
                _output = keras.layers.Dropout(
                    rate=dropout,
                    name='FeedForward-Dropout-{}-{}'.format(name, i + 1),
                )(_output)
            _output = keras.layers.Add(
                name='FeedForward-Residual-{}-{}'.format(name, i + 1),
            )([feed_forward_input, _output])
            _output = LayerNormalization(
                name='FeedForward-Normal-{}-{}'.format(name, i + 1),
            )(_output)
            return _output

        content = content_output
        content_output = _build_block('Content', content_output, content)
        if training:
            query_output = _build_block('Query', query_output, content)

    if training:
        output = EmbeddingSim(name='Softmax')([query_output, embed_weights])
    else:
        output = content_output
    model = keras.models.Model(
        inputs=[token_input, seg_input, memory_length_input, perm_input],
        outputs=output
    )
    return model
