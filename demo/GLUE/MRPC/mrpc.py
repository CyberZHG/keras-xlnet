import os

import numpy as np

from keras_xlnet.backend import keras
from keras_bert.layers import Extract
from keras_xlnet import PretrainedList, get_pretrained_paths
from keras_xlnet import Tokenizer, load_trained_model_from_checkpoint, ATTENTION_TYPE_BI

EPOCH = 10
BATCH_SIZE = 32
SEQ_LEN = 120
MODEL_NAME = 'MRPC.h5'

current_path = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(current_path, 'msr_paraphrase_train.txt')
dev_path = os.path.join(current_path, 'mrpc_dev_ids.tsv')
test_path = os.path.join(current_path, 'msr_paraphrase_test.txt')

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


with open(dev_path) as dev_reader:
    dev_ids = set([tuple(line.strip().split('\t')) for line in dev_reader])


def generate_sequence(path, dev_ids=None, select_dev=False):
    tokens, classes = [], []
    with open(path) as reader:
        reader.readline()
        for line in reader:
            line = line.strip()
            parts = line.split('\t')
            ids = (parts[1], parts[2])
            if dev_ids is not None:
                if select_dev:
                    if ids not in dev_ids:
                        continue
                else:
                    if ids in dev_ids:
                        continue
            encoded_a, encoded_b = tokenizer.encode(parts[3]), tokenizer.encode(parts[4])
            encoded = encoded_a + [tokenizer.SYM_SEP] + encoded_b + [tokenizer.SYM_SEP]
            encoded = [tokenizer.SYM_PAD] * (SEQ_LEN - 1 - len(encoded)) + encoded + [tokenizer.SYM_CLS]
            tokens.append(encoded)
            classes.append(int(parts[0], 10))
    tokens, classes = np.array(tokens), np.array(classes)
    segments = np.zeros_like(tokens)
    segments[:, -1] = 1
    lengths = np.zeros_like(tokens[:, :1])
    return DataSequence([tokens, segments, lengths], classes)


current_path = os.path.dirname(os.path.abspath(__file__))
train_seq = generate_sequence(train_path, dev_ids=dev_ids, select_dev=False)
dev_seq = generate_sequence(train_path, dev_ids=dev_ids, select_dev=True)
test_seq = generate_sequence(test_path)


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
output = keras.layers.Dense(units=2, activation='softmax', name='Softmax')(dropout)
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
for seq in [dev_seq, test_seq]:
    results = model.predict_generator(seq, verbose=True).argmax(axis=-1)
    tp, fp, fn, tn = 0, 0, 0, 0
    for i in range(len(results)):
        if results[i] == 1:
            if seq.y[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if seq.y[i] == 1:
                fn += 1
            else:
                tn += 1

    print('Confusion:')
    print('[{}, {}]'.format(tp, fp))
    print('[{}, {}]'.format(fn, tn))

    print('Accuracy: %.2f' % (100.0 * (tp + tn) / len(results)))
    print('Precision: %.2f' % (100.0 * tp / (tp + fp)))
    print('Recall: %.2f' % (100.0 * tp / (tp + fn)))
    print('F1-Score: %.2f' % (100.0 * (2.0 * tp) / (2.0 * tp + fp + fn)))
