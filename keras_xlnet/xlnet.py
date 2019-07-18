from .backend import keras

from keras_embed_sim import EmbeddingRet, EmbeddingSim
from keras_transformer import gelu
from keras_transformer_xl import RelativeBias, Memory, PositionalEmbedding
from keras_layer_normalization import LayerNormalization
from keras_position_wise_feed_forward import FeedForward

from .segment_bias import SegmentBias
from .segment_embed import RelativeSegmentEmbedding
from .permutation import PermutationMask
from .mask_embed import MaskEmbedding
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
        'PermutationMask': PermutationMask,
        'MaskEmbedding': MaskEmbedding,
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
    inputs = [token_input, seg_input, memory_length_input]
    if training:
        query_input = keras.layers.Input(
            shape=(target_len,),
            name='Input-Mask',
        )
        inputs.append(query_input)
    else:
        query_input = None
    token_embed, embed_weights = EmbeddingRet(
        input_dim=num_token,
        output_dim=units,
        mask_zero=True,
        name='Embed-Token',
    )(token_input)
    if training:
        mask_embed = MaskEmbedding(
            units=units,
            name='Embed-Mask'
        )([token_embed, query_input])
    else:
        mask_embed = None
    if 0.0 < dropout < 1.0:
        token_embed = keras.layers.Dropout(
            rate=dropout,
            name='Embed-Token-Dropout'
        )(token_embed)
        if training:
            mask_embed = keras.layers.Dropout(
                rate=dropout,
                name='Embed-Mask-Dropout'
            )(mask_embed)

    memories = [Memory(
        batch_size=batch_size,
        memory_len=memory_len,
        target_len=target_len,
        output_dim=units,
        name='Memory-0',
    )([token_embed, memory_length_input])]

    pos_embed = PositionalEmbedding(
        output_dim=units,
        clamp_len=clamp_len,
        name='Embed-Pos',
    )([token_embed, memories[0]])

    content_mask, query_mask = PermutationMask(
        name='Permutation',
    )([token_embed, memories[0]])

    context_bias, relative_bias, segment_bias = None, None, None
    if shared_biases:
        context_bias, relative_bias = RelativeBias(
            units,
            name='Relative-Bias',
        )(memories[0])
        segment_bias = SegmentBias(
            units,
            name='Segment-Bias',
        )(memories[0])

    content_output, query_output = token_embed, None
    if training:
        query_output = mask_embed

    for i in range(num_block):
        if not shared_biases:
            context_bias, relative_bias = RelativeBias(
                units,
                name='Relative-Bias-{}'.format(i + 1),
            )(memories[i])
            segment_bias = SegmentBias(
                units,
                name='Segment-Bias-{}'.format(i + 1),
            )(memories[i])

        segment_mat, segment_embed = RelativeSegmentEmbedding(
            units=units,
            name='Embed-Segment-{}'.format(i + 1),
        )([seg_input, memories[i]])

        attention = Attention(
            units=units,
            num_head=num_head,
            use_bias=False,
            attention_dropout=attention_dropout,
            name='Attention-{}'.format(i + 1),
        )
        if 0.0 < dropout < 1.0:
            attention_dropout_layer = keras.layers.Dropout(
                rate=dropout,
                name='Attention-Dropout-{}'.format(i + 1),
            )
        else:
            attention_dropout_layer = None
        attention_add = keras.layers.Add(name='Attention-Residual-{}'.format(i + 1))
        attention_layer_norm = LayerNormalization(name='Attention-Normal-{}'.format(i + 1))

        feed_forward = FeedForward(
            units=hidden_dim,
            dropout_rate=dropout,
            activation=gelu,
            name='FeedForward-{}'.format(i + 1),
        )
        if 0.0 < dropout < 1.0:
            feed_forward_dropout = keras.layers.Dropout(
                rate=dropout,
                name='FeedForward-Dropout-{}'.format(i + 1),
            )
        else:
            feed_forward_dropout = None
        feed_forward_add = keras.layers.Add(name='FeedForward-Residual-{}'.format(i + 1))
        feed_forward_layer_norm = LayerNormalization(name='FeedForward-Normal-{}'.format(i + 1))

        content = content_output

        def _build_output(query, mask):
            attention_input = query
            _output = attention([
                query, content, memories[i],
                segment_mat, segment_embed, pos_embed,
                context_bias, relative_bias, segment_bias,
                mask,
            ])
            if attention_dropout_layer is not None:
                _output = attention_dropout_layer(_output)
            _output = attention_add([attention_input, _output])
            _output = attention_layer_norm(_output)

            feed_forward_input = _output
            _output = feed_forward(_output)
            if feed_forward_dropout is not None:
                _output = feed_forward_dropout(_output)
            _output = feed_forward_add([feed_forward_input, _output])
            _output = feed_forward_layer_norm(_output)
            return _output

        content_output = _build_output(content_output, content_mask)
        if training:
            query_output = _build_output(query_output, query_mask)

        if i < num_block - 1:
            memories.append(Memory(
                batch_size=batch_size,
                memory_len=memory_len,
                target_len=target_len,
                output_dim=units,
                name='Memory-{}'.format(i + 1),
            )([content_output, memory_length_input]))

    if training:
        output = EmbeddingSim(name='Softmax')([query_output, embed_weights])
    else:
        output = content_output
    model = keras.models.Model(
        inputs=inputs,
        outputs=output
    )
    return model
