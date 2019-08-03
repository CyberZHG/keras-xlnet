import os
import sys

import numpy as np

from keras_xlnet import PretrainedList, get_pretrained_paths
from keras_xlnet import Tokenizer, load_trained_model_from_checkpoint
from keras_xlnet import ATTENTION_TYPE_UNI, ATTENTION_TYPE_BI


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

token_input = np.expand_dims(np.array(tokens), axis=0)
segment_input = np.zeros_like(token_input)
memory_length_input = np.zeros((1, 1))

# Load pre-trained model
model = load_trained_model_from_checkpoint(
    config_path=config_path,
    checkpoint_path=model_path,
    batch_size=1,
    memory_len=0,
    target_len=7,
    in_train_phase=False,
    attention_type=ATTENTION_TYPE_UNI,
)

# Predict
results = model.predict_on_batch([token_input, segment_input, memory_length_input])
print('# Uni-directional')
for i in range(len(tokens)):
    print(results[0, i, :5])

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

# Load pre-trained model
model = load_trained_model_from_checkpoint(
    config_path=config_path,
    checkpoint_path=model_path,
    batch_size=1,
    memory_len=0,
    target_len=7,
    in_train_phase=False,
    attention_type=ATTENTION_TYPE_BI,
)

# Predict
results = model.predict_on_batch([token_input, segment_input, memory_length_input])
print('# Bi-directional')
for i in range(len(tokens)):
    print(results[0, i, :5])

"""
Official outputs of [0, i, :5]:

  '_All': [-1.5559932  -3.7393584   1.081502   -2.0768185  -4.210291  ]
     "'": [-1.347696   -2.761841    0.28881213  0.1324176  -4.2987413 ]
     's': [-0.6636667  -1.0754938   0.63279617  0.8230225  -2.6985726 ]
'_right': [ 0.34975937 -1.767821    0.6679799   0.7689314  -2.206827  ]
 '_with': [-0.12003238 -0.89179885  0.17190406  0.6927716  -1.2259011 ]
  '_the': [-0.56934553  1.3299885  -0.36877036  0.5785186  -1.033815  ]
'_world': [-0.93081206  0.04857514 -0.83456343  0.05322461  0.18784925]
"""
