import os
import sys

import numpy as np

from keras_xlnet import Tokenizer, load_trained_model_from_checkpoint, ATTENTION_TYPE_BI


if len(sys.argv) == 2:
    '''Can be found at https://github.com/ymcui/Chinese-PreTrained-XLNet'''
    checkpoint_path = sys.argv[1]
    vocab_path = os.path.join(checkpoint_path, 'spiece.model')
    config_path = os.path.join(checkpoint_path, 'xlnet_config.json')
    model_path = os.path.join(checkpoint_path, 'xlnet_model.ckpt')
else:
    print('python3 token_embed.py CHECPOINT_PATH')
    sys.exit(-1)

# Tokenize inputs
tokenizer = Tokenizer(vocab_path)
text = "给岁月以文明"
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
    target_len=6,
    in_train_phase=False,
    attention_type=ATTENTION_TYPE_BI,
)

# Predict
results = model.predict_on_batch([token_input, segment_input, memory_length_input])
for i in range(len(tokens)):
    print(results[0, i, :5])

"""
Official outputs of [0, i, :5]:

  '给': [ 1.896126    3.493831     0.38272348  -5.808503   3.8789816 ]
  '岁': [ 2.2447402   3.1403003    2.1649847   -5.424841   1.3610269 ]
  '月': [-0.03201458  3.808922     0.86165166  -0.6400312  1.1946323 ]
  '以': [-3.048748   -0.15033042  -0.9607934   -1.5968797  1.0801126 ]
  '文': [-3.489529    2.4176004   -1.0115104   -4.0798264  0.07743317]
  '明': [-0.7437068   3.883186    -1.3612015   -6.2936883  0.8162358 ]
"""
