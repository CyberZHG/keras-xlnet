import os
import sys

import numpy as np

from keras_xlnet import PretrainedList, get_pretrained_paths
from keras_xlnet import Tokenizer, load_trained_model_from_checkpoint
from keras_xlnet import ATTENTION_TYPE_UNI


if len(sys.argv) == 2:
    checkpoint_path = sys.argv[1]
    vocab_path = os.path.join(checkpoint_path, 'spiece.model')
    config_path = os.path.join(checkpoint_path, 'xlnet_config.json')
    model_path = os.path.join(checkpoint_path, 'xlnet_model.ckpt')
else:
    checkpoint_path = get_pretrained_paths(PretrainedList.en_cased_base)
    vocab_path = checkpoint_path.vocab
    config_path = checkpoint_path.config
    model_path = checkpoint_path.model

# Tokenize inputs
tokenizer = Tokenizer(vocab_path)
text = "All's right with the world"
tokens = tokenizer.encode(text)

target_len = 3
total_len = len(tokens)

# Load pre-trained model
model = load_trained_model_from_checkpoint(
    config_path=config_path,
    checkpoint_path=model_path,
    batch_size=1,
    memory_len=total_len,
    target_len=target_len,
    in_train_phase=False,
    attention_type=ATTENTION_TYPE_UNI,
)

# Predict
for mem_len in range(0, total_len, target_len):
    index = mem_len // target_len
    sub_tokens = tokens[index * target_len:(index + 1) * target_len]
    token_input = np.expand_dims(np.array(sub_tokens + [Tokenizer.SYM_PAD] * (target_len - len(sub_tokens))), axis=0)
    segment_input = np.zeros_like(token_input)
    memory_length_input = np.array([[mem_len]])
    results = model.predict_on_batch([token_input, segment_input, memory_length_input])
    for i in range(3):
        if index * 3 + i < 7:
            print(results[0, i, :5])
    print('')
"""
Official outputs of [0, i, :5]:

  '_All': [ 1.3914602   0.47409844 -0.18970338 -1.9293687  -0.97905093]
     "'": [-1.1784189   1.5238011   1.129552    0.6578493  -4.228884  ]
     's': [-0.26101297  2.4856389   0.2463414   0.7937627  -4.5178328 ]
'_right': [ 0.14573663  3.205772    1.0440648  -0.6510392  -3.068475  ]
 '_with': [ 0.51798135  0.81651264  1.1682358  -0.66321874 -2.7442    ]
  '_the': [-0.17506915  0.6740285  -0.17924197 -0.8452157  -0.30235714]
'_world': [-0.17312089  1.0867603   0.79385513  0.6352485   0.17119849]
"""
