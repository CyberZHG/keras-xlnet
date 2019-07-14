import json
import numpy as np
import tensorflow as tf
from .xlnet import build_xlnet


__all__ = [
    'build_model_from_config',
    'load_model_weights_from_checkpoint',
    'load_trained_model_from_checkpoint',
]


def checkpoint_loader(checkpoint_file):
    def _loader(name):
        return tf.train.load_variable(checkpoint_file, name)
    return _loader


def build_model_from_config(config_path,
                            batch_size,
                            memory_len,
                            target_len,
                            in_train_phase):
    """Build the model from config file.

    :param config_path: The path to the JSON configuration file.
    :param batch_size: Batch size.
    :param memory_len: Maximum size of memory.
    :param target_len: Length of target.
    :param in_train_phase: Whether in training phase.
    :return: model and config
    """
    with open(config_path, 'r') as reader:
        config = json.loads(reader.read())
    model = build_xlnet(
        units=config['d_model'],
        training=in_train_phase,
        num_token=config['n_token'],
        num_block=config['n_layer'],
        num_head=config['n_head'],
        hidden_dim=config['d_inner'],
        batch_size=batch_size,
        memory_len=memory_len,
        target_len=target_len,
        dropout=0.0,
        attention_dropout=0.0,
        clamp_len=None,
        shared_biases=not config['untie_r'],
    )
    return model, config


def load_model_weights_from_checkpoint(model,
                                       config,
                                       checkpoint_path,
                                       in_train_phase):
    """Load trained official model from checkpoint.

    :param model: Built keras model.
    :param config: Loaded configuration file.
    :param checkpoint_path: The path to the checkpoint files, should end with '.ckpt'.
    :param in_train_phase: Whether in training phase.
    """
    units = config['d_model']
    loader = checkpoint_loader(checkpoint_path)

    model.get_layer(name='Embed-Token').set_weights([
        loader('model/transformer/word_embedding/lookup_table'),
    ])

    if in_train_phase:
        model.get_layer(name='Embed-Mask').set_weights([
            loader('model/transformer/mask_emb/mask_emb'),
        ])

    r_w_bias = loader('model/transformer/r_w_bias')
    r_r_bias = loader('model/transformer/r_r_bias')
    r_s_bias = loader('model/transformer/r_s_bias')
    segment_embed = loader('model/transformer/seg_embed')
    if config.get('untie_r', False):
        for i in range(config['n_layer']):
            model.get_layer(name='Relative-Bias-{}'.format(i + 1)).set_weights([
                r_w_bias[i].flatten(),
                r_r_bias[i].flatten(),
            ])
            model.get_layer(name='Segment-Bias-{}'.format(i + 1)).set_weights([
                r_s_bias[i].flatten(),
            ])
    else:
        model.get_layer(name='Relative-Bias').set_weights([
            r_w_bias.flatten(),
            r_r_bias.flatten(),
        ])
        model.get_layer(name='Segment-Bias').set_weights([
            r_s_bias.flatten(),
        ])

    for i in range(config['n_layer']):
        model.get_layer(name='Embed-Segment-{}'.format(i + 1)).set_weights([
            segment_embed[i].reshape((2, units))
        ])

        att_kernel_name = 'model/transformer/layer_{}/rel_attn/{}/kernel'
        model.get_layer(name='Attention-{}'.format(i + 1)).set_weights([
            np.concatenate(
                [
                    loader(att_kernel_name.format(i, 'q')).reshape((units, units)),
                    loader(att_kernel_name.format(i, 'k')).reshape((units, units)),
                    loader(att_kernel_name.format(i, 'v')).reshape((units, units)),
                    loader(att_kernel_name.format(i, 'r')).reshape((units, units)),
                    loader(att_kernel_name.format(i, 'o')).reshape((units, units)).transpose(),
                ],
                axis=1,
            ),
        ])
        model.get_layer(name='Attention-Normal-{}'.format(i + 1)).set_weights([
            loader('model/transformer/layer_{}/rel_attn/LayerNorm/gamma'.format(i)),
            loader('model/transformer/layer_{}/rel_attn/LayerNorm/beta'.format(i)),
        ])
        model.get_layer(name='FeedForward-{}'.format(i + 1)).set_weights([
            loader('model/transformer/layer_{}/ff/layer_1/kernel'.format(i)),
            loader('model/transformer/layer_{}/ff/layer_1/bias'.format(i)),
            loader('model/transformer/layer_{}/ff/layer_2/kernel'.format(i)),
            loader('model/transformer/layer_{}/ff/layer_2/bias'.format(i)),
        ])
        model.get_layer(name='FeedForward-Normal-{}'.format(i + 1)).set_weights([
            loader('model/transformer/layer_{}/ff/LayerNorm/gamma'.format(i)),
            loader('model/transformer/layer_{}/ff/LayerNorm/beta'.format(i)),
        ])

    if in_train_phase:
        model.get_layer(name='Softmax').set_weights([
            loader('model/lm_loss/bias'),
        ])


def load_trained_model_from_checkpoint(config_path,
                                       checkpoint_path,
                                       batch_size,
                                       memory_len,
                                       target_len,
                                       in_train_phase=False):
    """Load trained official model from checkpoint.

    :param config_path: The path to the JSON configuration file.
    :param checkpoint_path: The path to the checkpoint files, should end with '.ckpt'.
    :param batch_size: Batch size.
    :param memory_len: Maximum size of memory.
    :param target_len: Length of target.
    :param in_train_phase: Whether in training phase.
    :return: model
    """
    model, config = build_model_from_config(
        config_path=config_path,
        batch_size=batch_size,
        memory_len=memory_len,
        target_len=target_len,
        in_train_phase=in_train_phase,
    )
    load_model_weights_from_checkpoint(
        model=model,
        config=config,
        checkpoint_path=checkpoint_path,
        in_train_phase=in_train_phase,
    )
    return model
