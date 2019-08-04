# Keras XLNet

[![Travis](https://travis-ci.org/CyberZHG/keras-xlnet.svg)](https://travis-ci.org/CyberZHG/keras-xlnet)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-xlnet/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-xlnet)
[![Version](https://img.shields.io/pypi/v/keras-xlnet.svg)](https://pypi.org/project/keras-xlnet/)
![Downloads](https://img.shields.io/pypi/dm/keras-xlnet.svg)
![License](https://img.shields.io/pypi/l/keras-xlnet.svg)

![](https://img.shields.io/badge/keras-tensorflow-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras/eager-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras/2.0_beta-blue.svg)

\[[中文](https://github.com/CyberZHG/keras-xlnet/blob/master/README.zh-CN.md)|[English](https://github.com/CyberZHG/keras-xlnet/blob/master/README.md)\]

Unofficial implementation of [XLNet](https://arxiv.org/pdf/1906.08237). [Embedding extraction](demo/extract/token_embeddings.py) and [embedding extract with memory](demo/extract/token_embeddings_with_memory.py) show how to get the outputs of the last transformer layer using pre-trained checkpoints.

## Install

```bash
pip install keras-xlnet
```

## Usage

### Fine-tuning on GLUE

Click the task name to see the demos with base model:

|Task Name                       |Metrics                       |Approximate Results on Dev Set|
|:-------------------------------|:----------------------------:|----:|
|[CoLA](demo/GLUE/CoLA/cola.py)  |Matthew Corr.                 |52   |
|[SST-2](demo/GLUE/SST-2/sst2.py)|Accuracy                      |93   |
|[MRPC](demo/GLUE/MRPC/mrpc.py)  |Accuracy/F1                   |86/89|
|[STS-B](demo/GLUE/STS-B/stsb.py)|Pearson Corr. / Spearman Corr.|86/87|
|[QQP](demo/GLUE/QQP/qqp.py)     |Accuracy/F1                   |90/86|
|[MNLI](demo/GLUE/MNLI/mnli.py)  |Accuracy                      |84/84|
|[QNLI](demo/GLUE/QNLI/qnli.py)  |Accuracy                      |86   |
|[RTE](demo/GLUE/RTE/rte.py)     |Accuracy                      |64   |
|[WNLI](demo/GLUE/WNLI/wnli.py)  |Accuracy                      |56   |

(Only 0s are predicted in WNLI dataset)

### Load Pretrained Checkpoints

```python
import os
from keras_xlnet import Tokenizer, load_trained_model_from_checkpoint, ATTENTION_TYPE_BI

checkpoint_path = '.../xlnet_cased_L-24_H-1024_A-16'

tokenizer = Tokenizer(os.path.join(checkpoint_path, 'spiece.model'))
model = load_trained_model_from_checkpoint(
    config_path=os.path.join(checkpoint_path, 'xlnet_config.json'),
    checkpoint_path=os.path.join(checkpoint_path, 'xlnet_model.ckpt'),
    batch_size=16,
    memory_len=512,
    target_len=128,
    in_train_phase=False,
    attention_type=ATTENTION_TYPE_BI,
)
model.summary()
```

Arguments `batch_size`, `memory_len` and `target_len` are maximum sizes used for initialization of memories. The model used for training a language model is returned if `in_train_phase` is `True`, otherwise a model used for fine-tuning will be returned.

### About I/O

**Note that** `shuffle` should be `False` in either `fit` or `fit_generator` if memories are used. 

#### `in_train_phase` is `False`

3 inputs:

* IDs of tokens, with shape `(batch_size, target_len)`.
* IDs of segments, with shape `(batch_size, target_len)`.
* Length of memories, with shape `(batch_size, 1)`.

1 output:

* The feature for each token, with shape `(batch_size, target_len, units)`.

#### `in_train_phase` is `True`

4 inputs:

* IDs of tokens, with shape `(batch_size, target_len)`.
* IDs of segments, with shape `(batch_size, target_len)`.
* Length of memories, with shape `(batch_size, 1)`.
* Masks of tokens, with shape `(batch_size, target_len)`.

1 output:

* The probability of each token in each position, with shape `(batch_size, target_len, num_token)`.
