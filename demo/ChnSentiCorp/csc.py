import os
import sys
from collections import namedtuple

import numpy as np
import pandas as pd

from keras_xlnet.backend import keras
from keras_bert.layers import Extract
from keras_xlnet import Tokenizer, load_trained_model_from_checkpoint, ATTENTION_TYPE_BI
from keras_radam import RAdam

EPOCH = 10
BATCH_SIZE = 4
SEQ_LEN = 256
MODEL_NAME = 'ChnSentiCorp.h5'


if len(sys.argv) != 2:
    print('python csc.py PRETRAINED_PATH')


pretrained_path = sys.argv[1]
PretrainedPaths = namedtuple('PretrainedPaths', ['config', 'model', 'vocab'])
config_path = os.path.join(pretrained_path, 'xlnet_config.json')
model_path = os.path.join(pretrained_path, 'xlnet_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'spiece.model')
paths = PretrainedPaths(config_path, model_path, vocab_path)
tokenizer = Tokenizer(paths.vocab)


# Read data
class DataSequence(keras.utils.Sequence):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return (len(self.y) + BATCH_SIZE - 1) // BATCH_SIZE

    def __getitem__(self, index):
        s = slice(index * BATCH_SIZE, (index + 1) * BATCH_SIZE)
        return [item[s] for item in self.x], self.y[s]


def generate_sequence(path):
    tokens, classes = [], []
    df = pd.read_csv(path, sep='\t', error_bad_lines=False)
    for _, row in df.iterrows():
        text, cls = row['text_a'], row['label']
        encoded = tokenizer.encode(text)[:SEQ_LEN - 1]
        encoded = [tokenizer.SYM_PAD] * (SEQ_LEN - 1 - len(encoded)) + encoded + [tokenizer.SYM_CLS]
        tokens.append(encoded)
        classes.append(int(cls))
    tokens, classes = np.array(tokens), np.array(classes)
    segments = np.zeros_like(tokens)
    segments[:, -1] = 1
    lengths = np.zeros_like(tokens[:, :1])
    return DataSequence([tokens, segments, lengths], classes)


current_path = os.path.dirname(os.path.abspath(__file__))
train_seq = generate_sequence(os.path.join(current_path, 'train.tsv'))
dev_seq = generate_sequence(os.path.join(current_path, 'dev.tsv'))
test_seq = generate_sequence(os.path.join(current_path, 'test.tsv'))


# Load pretrained model
model = load_trained_model_from_checkpoint(
    config_path=paths.config,
    checkpoint_path=paths.model,
    batch_size=BATCH_SIZE,
    memory_len=0,
    target_len=SEQ_LEN,
    in_train_phase=False,
    attention_type=ATTENTION_TYPE_BI,
)


# Build classification model
last = model.output
extract = Extract(index=-1, name='Extract')(last)
dense = keras.layers.Dense(units=768, name='Dense')(extract)
norm = keras.layers.BatchNormalization(name='Normal')(dense)
output = keras.layers.Dense(units=2, activation='softmax', name='Softmax')(norm)
model = keras.models.Model(inputs=model.inputs, outputs=output)
model.summary()


# Fit model
if os.path.exists(MODEL_NAME):
    model.load_weights(MODEL_NAME)

model.compile(
    optimizer=RAdam(lr=2e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'],
)

model.fit_generator(
    generator=train_seq,
    validation_data=dev_seq,
    epochs=EPOCH,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_sparse_categorical_accuracy',
            restore_best_weights=True,
            patience=3,),
    ],
)

model.save_weights(MODEL_NAME)

# Evaluation
results = model.predict_generator(test_seq, verbose=True).argmax(axis=-1)
tp, fp, fn, tn = 0, 0, 0, 0
for i in range(len(results)):
    if results[i] == 1:
        if test_seq.y[i] == 1:
            tp += 1
        else:
            fp += 1
    else:
        if test_seq.y[i] == 1:
            fn += 1
        else:
            tn += 1

print('Confusion:')
print('[{}, {}]'.format(tp, fp))
print('[{}, {}]'.format(fn, tn))

print('Accuracy: %.4f' % ((tp + tn) / (tp + fp + fn + tn)))
print('Precision: %.2f' % (100.0 * tp / (tp + fp + 1e-8)))
print('Recall: %.2f' % (100.0 * tp / (tp + fn + 1e-8)))
print('F1-Score: %.2f' % (100.0 * (2.0 * tp) / (2.0 * tp + fp + fn)))
