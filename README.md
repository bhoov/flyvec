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


### Creating dense representations

We encourage usage of the sparse word embeddings which is calculated by selecting the top `H` activated [Kenyon Cells](https://en.wikipedia.org/wiki/Kenyon_cell) in our model. However, if you need a dense representation of the word embeddings, you can get the raw `softmax`ed activations by running:

```
# Generate dense word embeddings
dense_embed = model.get_dense_embedding("incredible"); 
print(f"First 10 entries of the dense embedding:\n {dense_embed['embedding'][:10]}")
```

    First 10 entries of the dense embedding:
     [1.8123710e-05 6.1162762e-05 7.3589981e-05 3.7589352e-04 1.0641745e-04
     1.6521414e-04 3.7847902e-05 9.5790623e-05 1.2732553e-04 5.6038076e-05]


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



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-13-74d5f287910c> in <module>
          3 
          4 # The IDs that correspond to those tokens
    ----> 5 print(model.vocab[:5])
          6 
          7 # The dictionary object itself


    TypeError: 'set' object is not subscriptable


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
