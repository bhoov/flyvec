# FlyVec
> Flybrain-inspired Sparse Binary Word Embeddings


Code based on the ICLR 2021 paper [Can a Fruit Fly Learn Word Embeddings?](https://openreview.net/forum?id=xfmSoxdxFCG ). A work in progress.

## Install

`pip install flyvec`

## How to use

```python
#hide_output
import numpy as np
from flyvec import FlyVec

model = FlyVec.load()
embed_info = model.get_sparse_embedding("market")
```

FlyVec uses a simple, word-based tokenizer with to isolate concepts. The provided model uses a tokenizer with about 40,000 words, all lower-cased, with special tokens for numbers (`<NUM>`) and unknown words (`<UNK>`). See `Tokenizer` for details.

```python
# Batch generate word embeddings
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

