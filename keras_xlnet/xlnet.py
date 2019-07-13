from .backend import keras
from .backend import backend as K
from keras_embed_sim import EmbeddingRet, EmbeddingSim
from keras_transformer import gelu

__all__ = [
    'get_custom_objects', 'set_custom_objects',
]


def get_custom_objects() -> dict:
    return {
        'gelu': gelu,
        'EmbeddingRet': EmbeddingRet,
        'EmbeddingSim': EmbeddingSim,
    }


def set_custom_objects() -> None:
    for key, val in get_custom_objects().items():
        keras.utils.get_custom_objects()[key] = val


def build_xlnet(units,
                num_token):
    """Build XLNet.

    :param units: Hidden dimensions throughout the model.
    :param num_token: Number of distinct tokens.
    :return: The built model.
    """
    token_input = keras.layers.Input(
        shape=(None,),
        name='Input-Token',
    )
    token_embed, embed_weights = EmbeddingRet(
        input_dim=num_token,
        output_dim=units,
        name='Embed-Token',
    )(token_input)
