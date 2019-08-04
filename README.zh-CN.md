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

\[[中文](https://github.com/CyberZHG/keras-xlnet/blob/master/README.zh-CN.md)|[English](https://github.com/CyberZHG/keras-xlnet/blob/master/README.md)|[通用问题](https://github.com/CyberZHG/summary/blob/master/QA.md)\]

[XLNet](https://arxiv.org/pdf/1906.08237)的非官方实现。[嵌入提取](demo/extract/token_embeddings.py)和[有记忆的嵌入提取](demo/extract/token_embeddings_with_memory.py)展示了如何加载预训练检查点并得到transformer的输出特征。

## 安装

```bash
pip install keras-xlnet
```

## 使用

### GLUE微调

点击任务名可以查看基础模型的训练样例：

|任务名                           |指标                          |验证集上大致结果|
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

（注意：WNLI数据集上只输出了0，不是一个正常结果）

### 加载预训练检查点

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

参数`batch_size`、`memory_len`、`target_len`用于初始化记忆单元，代表最大尺寸，实际属于可以小于对应数值。如果`in_train_phase`是`True`会返回一个用于训练语言模型的模型，否则返回一个用于fine-tuning的模型。

### 关于输入输出

**注意**：依赖记忆时输入有序，一定不能打乱输入顺序，`fit`或`fit_generator`的`shuffle`应该为`False`。

#### `in_train_phase`是`False`

3个输入：

* 词的ID，形状为`(batch_size, target_len)`。
* 段落的ID，形状为`(batch_size, target_len)`。
* 历史记忆的长度，形状为`(batch_size, 1)`。

1个输出：

* 每个词的特征，形状为`(batch_size, target_len, units)`。

#### `in_train_phase`是`True`

4个输入，前三个和`in_train_phase`为`False`时相同：

* 词的ID，形状为`(batch_size, target_len)`。
* 段落的ID，形状为`(batch_size, target_len)`。
* 历史记忆的长度，形状为`(batch_size, 1)`。
* 被遮罩的词的蒙版，形状为`(batch_size, target_len)`。

1个输出：

* 每个位置每个词的概率，形状为`(batch_size, target_len, num_token)`。
