# FlyVec
> Flybrain-inspired Sparse Binary Word Embeddings


Code based on the ICLR 2021 paper [Can a Fruit Fly Learn Word Embeddings?](https://openreview.net/forum?id=xfmSoxdxFCG ). A work in progress.

## Install

`pip install flyvec`

## How to use

### Basic Usage

```
import numpy as np
from flyvec import FlyVec

model = FlyVec.load()
embed_info = model.get_sparse_embedding("market"); embed_info
```




    {'token': 'market',
     'id': 1180,
     'embedding': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0], dtype=int8)}



### Changing the Hash Length

```
small_embed = model.get_sparse_embedding("market", 4); np.sum(small_embed['embedding'])
```




    4



### Handling "unknown" tokens

FlyVec uses a simple, word-based tokenizer with to isolate concepts. The provided model uses a tokenizer with about 20,000 words, all lower-cased, with special tokens for numbers (`<NUM>`) and unknown words (`<UNK>`). Unknown tokens have the token id of `0`, so we can use this to filter unknown tokens.

```
unk_embed = model.get_sparse_embedding("DefNotAWord")
if unk_embed['id'] == 0:
    print("I AM THE UNKNOWN TOKEN DON'T USE ME FOR ANYTHING IMPORTANT")
```

    I AM THE UNKNOWN TOKEN DON'T USE ME FOR ANYTHING IMPORTANT


### Batch generating word embeddings

```
sentence = "Supreme Court dismissed the criminal charges."
tokens = model.tokenize(sentence)
embedding_info = [model.get_sparse_embedding(t) for t in tokens]
embeddings = np.array([e['embedding'] for e in embedding_info])
print("TOKENS: ", [e['token'] for e in embedding_info])
print("EMBEDDINGS: ", embeddings)
```

    TOKENS:  ['supreme', 'court', 'dismissed', 'the', 'criminal', 'charges']
    EMBEDDINGS:  [[0 1 0 ... 0 0 0]
     [0 0 0 ... 0 0 0]
     [0 0 0 ... 0 1 0]
     [0 0 0 ... 0 0 0]
     [0 0 0 ... 0 1 0]
     [0 0 0 ... 0 1 0]]


### Viewing the vocabulary

The vocabulary under the hood uses the gensim `Dictionary` and can be accessed by either IDs (`int`s) or Tokens (`str`s)

```
# The tokens in the vocabulary
print(model.token_vocab[:5])

# The IDs that correspond to those tokens
print(model.vocab[:5])

# The dictionary object itself
model.dictionary;
```

    ['properties', 'a', 'among', 'and', 'any']
    [2, 3, 4, 5, 6]


### Training

Please note that training `flyvec` on your own custom corpus is not currently supported. 

# Citation

If you use this in your work, please cite:

```
@inproceedings{
liang2021can,
title={Can a Fruit Fly Learn Word Embeddings?},
author={Yuchen Liang and Chaitanya Ryali and Benjamin Hoover and Saket Navlakha and Leopold Grinberg and Mohammed J Zaki and Dmitry Krotov},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=xfmSoxdxFCG}
}
```
