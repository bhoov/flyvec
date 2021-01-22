# FlyVec
> Flybrain-inspired Sparse Binary Word Embeddings


Code based on the ICLR 2021 paper [Can a Fruit Fly Learn Word Embeddings?](https://openreview.net/forum?id=xfmSoxdxFCG ). A work in progress.

## Install

```
pip install flyvec

```

## How to use

```
model = FlyVec.from_config("../data/model_config.yaml")
sentence = "Stock market plunged on Tuesday following the analyst reports"
ids = model.tokenizer.encode(sentence)
print("IDS: ", ids)
```

    Loading Tokenizer...
    No phraser specified. Proceeding without phrases
    IDS:  [5769, 1180, 9659, 321, 3206, 1160, 32, 10370, 880]


```
model.tokenizer.tokenize(sentence)
```




    ['stock',
     'market',
     'plunged',
     'on',
     'tuesday',
     'following',
     'the',
     'analyst',
     'reports']



```
# Decoding without stop words
go_words = [model.tokenizer.id2token(id) for id in ids if id not in model.stop_words]
print(go_words)
```

    Loading stop words...
    ['stock', 'market', 'plunged', 'tuesday', 'following', 'analyst', 'reports']

