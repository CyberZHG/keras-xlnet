import os

import numpy as np

from keras_xlnet.backend import keras
from keras_xlnet.backend import backend as K
from keras_bert.layers import Extract
from keras_xlnet import PretrainedList, get_pretrained_paths, Tokenizer, load_trained_model_from_checkpoint

EPOCH = 10
BATCH_SIZE = 64
SEQ_LEN = 32
MODEL_NAME = 'CoLA.h5'


# Load pretrained model
paths = get_pretrained_paths(PretrainedList.en_cased_base)
tokenizer = Tokenizer(paths.vocab)
model = load_trained_model_from_checkpoint(
    config_path=paths.config,
    checkpoint_path=paths.model,
    batch_size=BATCH_SIZE,
    memory_len=0,
    target_len=SEQ_LEN,
    in_train_phase=False,
    attention_type='bi',
)


# Build classification model
last = Extract(index=-1, name='Extract')(model.output)
dense = keras.layers.Dense(units=768, activation='tanh', name='Dense')(last)
dropout = keras.layers.Dropout(rate=0.1, name='Dropout')(dense)
output = keras.layers.Dense(units=2, activation='softmax', name='Softmax')(dropout)
model = keras.models.Model(inputs=model.inputs, outputs=output)
model.summary()


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
    with open(path) as reader:
        for line in reader:
            line = line.strip()
            parts = line.split('\t')
            encoded = tokenizer.encode(parts[-1])[:SEQ_LEN - 1]
            encoded = [tokenizer.SYM_PAD] * (SEQ_LEN - 1 - len(encoded)) + encoded + [tokenizer.SYM_CLS]
            tokens.append(encoded)
            if parts[1] == '0':
                classes.append(1)
            else:
                classes.append(0)
    tokens, classes = np.array(tokens), np.array(classes)
    segments = np.zeros_like(tokens)
    segments[:, -1] = 1
    lengths = np.zeros_like(tokens[:, :1])
    return DataSequence([tokens, segments, lengths], classes)


current_path = os.path.dirname(os.path.abspath(__file__))
train_seq = generate_sequence(os.path.join(current_path, 'train.tsv'))
dev_seq = generate_sequence(os.path.join(current_path, 'dev.tsv'))


# Fit model
if os.path.exists(MODEL_NAME):
    model.load_weights(MODEL_NAME)

model.compile(
    optimizer=keras.optimizers.Adam(lr=2.5e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'],
)

model.fit_generator(
    generator=train_seq,
    validation_data=dev_seq,
    epochs=EPOCH,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=3)],
)

model.save_weights(MODEL_NAME)

# Evaluation
# Use dev set because the results of test set is unknown
results = model.predict_generator(dev_seq, verbose=True).argmax(axis=-1)
tp, fp, fn, tn = 0, 0, 0, 0
for i in range(len(results)):
    if results[i] == 1:
        if dev_seq.y[i] == 1:
            tp += 1
        else:
            fp += 1
    else:
        if dev_seq.y[i] == 1:
            fn += 1
        else:
            tn += 1

print('Confusion:')
print('[{}, {}]'.format(tp, fp))
print('[{}, {}]'.format(fn, tn))

print('Accuracy: %.4f' % ((tp + tn) / (tp + fp + fn + tn)))

mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + K.epsilon())
print('MCC: %.4f' % mcc)
