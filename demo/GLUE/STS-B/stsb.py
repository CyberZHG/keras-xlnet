import os

import numpy as np
from scipy.stats import pearsonr, spearmanr

from keras_xlnet.backend import keras
from keras_bert.layers import Extract
from keras_xlnet import PretrainedList, get_pretrained_paths
from keras_xlnet import Tokenizer, load_trained_model_from_checkpoint, ATTENTION_TYPE_BI

EPOCH = 10
BATCH_SIZE = 32
SEQ_LEN = 140
MODEL_NAME = 'STS-B.h5'

current_path = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(current_path, 'train.tsv')
dev_path = os.path.join(current_path, 'dev.tsv')

paths = get_pretrained_paths(PretrainedList.en_cased_base)
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
    tokens, classes, scores = [], [], []
    max_len = 0
    with open(path) as reader:
        reader.readline()
        for line in reader:
            line = line.strip()
            parts = line.split('\t')
            encoded_a, encoded_b = tokenizer.encode(parts[7]), tokenizer.encode(parts[8])
            encoded = encoded_a + [tokenizer.SYM_SEP] + encoded_b + [tokenizer.SYM_SEP]
            max_len = max(max_len, len(encoded))
            encoded = [tokenizer.SYM_PAD] * (SEQ_LEN - 1 - len(encoded)) + encoded + [tokenizer.SYM_CLS]
            tokens.append(encoded)
            classes.append(round(float(parts[9])))
            scores.append(float(parts[9]))
    tokens, classes = np.array(tokens), np.array(classes)
    segments = np.zeros_like(tokens)
    segments[:, -1] = 1
    lengths = np.zeros_like(tokens[:, :1])
    return DataSequence([tokens, segments, lengths], classes), scores


current_path = os.path.dirname(os.path.abspath(__file__))
train_seq, _ = generate_sequence(train_path)
dev_seq, scores = generate_sequence(dev_path)

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
last = Extract(index=-1, name='Extract')(model.output)
dense = keras.layers.Dense(units=768, activation='tanh', name='Dense')(last)
dropout = keras.layers.Dropout(rate=0.1, name='Dropout')(dense)
output = keras.layers.Dense(units=6, activation='softmax', name='Softmax')(dropout)
model = keras.models.Model(inputs=model.inputs, outputs=output)
model.summary()


# Fit model
if os.path.exists(MODEL_NAME):
    model.load_weights(MODEL_NAME)

model.compile(
    optimizer=keras.optimizers.Adam(lr=3e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'],
)

model.fit_generator(
    generator=train_seq,
    validation_data=dev_seq,
    epochs=EPOCH,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)],
)

model.save_weights(MODEL_NAME)

# Evaluation
# Use dev set because the results of test set is unknown
classes = np.array([[0], [1], [2], [3], [4], [5]])
results = np.dot(model.predict_generator(dev_seq, verbose=True), classes).squeeze(axis=-1)
print('Pearson: %.2f' % (100.0 * pearsonr(results, scores)[0]))
print('Spearman: %.2f' % (100.0 * spearmanr(results, scores)[0]))
