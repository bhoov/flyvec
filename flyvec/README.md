# FlyVec Training

This is a wrapper around the original code that was used to train FlyVec. Originally coded to work on on the PowerPC architecture, it now supports any architecture with CUDA.

## How to use

**Prerequisites**
You need a python environment with `numpy` installed, a system that supports CUDA, `nvcc`, and `g++`.

**Building the Source Files**

`bash src/short_make`

Note that you will see some warnings. This is expected.

**Training**
`python train_flyvec.py path/to/encodings.npy path/to/offsets.npy -o save/checkpoints/in/this/directory`

**Description of Inputs**
- `encodings.npy` -- An `np.int32` array representing the tokenized vocabulary-IDs of the input corpus, of shape `(N,)` where `N` is the number of tokens in the corpus
- `offsets.npy` -- An `np.uint64` array of shape `(C,)` where `C` is the number of chunks in the corpus. Each each value represents the index that starts a new chunk within `encodings.npy`. 
    (Chunks can be thought of as sentences or paragraphs within the corpus; boundaries over which the sliding window does not cross.)

**Description of Outputs**
- `model_X.npy` -- Stores checkpoints after every epoch within the specified output directory

See `python train_flyvec.py --help` for more options.
