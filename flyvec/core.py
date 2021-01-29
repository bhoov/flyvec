# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/00_core.ipynb (unless otherwise specified).

__all__ = ['softmax', 'normalize_synapses', 'FlyVec']

# Cell
import yaml
import os
import numpy as np
from .tokenizer import GensimTokenizer
from pathlib import Path
from cached_property import cached_property
from typing import *
from fastcore.test import *
from .downloader import prepare_flyvec_data, get_config_dir, get_model_dir

# Cell
def softmax(x: np.array, beta=1.0):
    """Take the softmax of 1-D vector `x` according to inverse temperature `beta`. Returns a vector of the same length as x"""
    v = np.exp(beta*x)
    return v / np.sum(v)

# Cell
def normalize_synapses(syn: np.array, prec=1.0e-32, p=2):
    """Normalize the synapses

    Args:
        syn: The matrix of learned synapses
        prec: Noise to prevent division by 0
        p: Of the p-norm

    Returns:
        Normalized array of the given synapses
    """
    K, N = syn.shape
    nc = np.power(np.sum(syn**p,axis=1),1/p).reshape(K,1)
    return syn / np.tile(nc + prec, (1, N))

# Cell
class FlyVec:
    """A class wrapper around a tokenizer, stop words, and synapse weights for hashing words"""

    def __init__(self, synapse_file: Union[Path, str], tokenizer_file: Union[Path, str], stopword_file: Optional[Union[Path, str]]=None, phrases_file: Optional[Union[Path, str]]=None, normalize_synapses: bool=True):

        self.synapse_file = str(synapse_file)
        self.tokenizer_file = str(tokenizer_file)
        self.stopword_file = str(stopword_file) if stopword_file is not None else None
        self.phrases_file = str(phrases_file) if phrases_file is not None else None
        self.normalize_synapses = normalize_synapses

    @classmethod
    def load(cls, force_redownload=False):
        """Load the default configuration for the FlyVec model. If the data is not present in the local
        flyvec configuration, download it

        Args:
            force_redownload: Pass `True` to redownload models. Useful if the data gets corrupted
        """
        prepare_flyvec_data(force=force_redownload)
        return cls.from_config(get_model_dir() / "config.yaml")


    @classmethod
    def from_config(cls, fname: Union[Path, str]):
        """Create an instance of this class from the configuration present in the `fname` yaml file"""
        fpath = Path(fname)
        ref_dir = fpath.parent

        with open(fname, "r") as fp:
            conf = yaml.load(fp, Loader=yaml.FullLoader)

        synapse_file = ref_dir / conf["synapses"]
        tokenizer_file = ref_dir / conf["tokenizer"]
        phrases_file = ref_dir / conf["phrases"] if "phrases" in conf.keys() else None
        stopword_file = ref_dir / conf["stop_words"] if "stop_words" in conf.keys() else None
        normalize_synapses = conf.get("normalize_synapses", False)

        return cls(synapse_file, tokenizer_file, stopword_file=stopword_file, phrases_file=phrases_file, normalize_synapses=normalize_synapses)

    @cached_property
    def n_neurons(self): return self.synapses.shape[0]

    @cached_property
    def synapses(self):
        """The primary weights learned by the model"""
        syn = np.load(self.synapse_file)

        if self.normalize_synapses: return normalize_synapses(syn)
        return syn

    @cached_property
    def tokenizer(self):
        return GensimTokenizer.from_file(self.tokenizer_file, self.phrases_file)

    @cached_property
    def dictionary(self):
        return self.tokenizer.dictionary

    @cached_property
    def vocab(self):
        return self.tokenizer.vocab

    @cached_property
    def token_vocab(self):
        return self.tokenizer.token_vocab

    @cached_property
    def n_vocab(self): return self.tokenizer.n_vocab()

    @cached_property
    def stop_words(self):
        """Words the model should not respond to"""
        return set(np.load(self.stopword_file))

    @cached_property
    def unknown_embedding_info(self):
        return {
            "token": "<UNK>",
            "id": 0,
            "embedding": np.zeros(self.n_neurons).astype(np.uint8)
        }

    def tokenize(self, sentence:str):
        return self.tokenizer.tokenize(sentence)

    def is_unknown_token(self, token:str):
        """Check if a token is unknown (return false) or bad (raise ValueError)"""
        if len(token) == 0:
            raise ValueError("Token cannot be the empty string")

        tok = self.tokenizer.tokenize(token)[0]
        tok_id = self.tokenizer.token2id(tok)
        return tok_id == 0

    def get_sparse_embedding(self, word: str, hash_length: int=32):
        """Get a context-independent word embedding for a given word.
        If, when tokenized, the word is composed of multiple tokens, return the embedding of the first.

        Args:
            token: A token (in the vocabulary) to get the word-embedding of
            hash_length: The number of non-zero entries in the sparse embedding
            normalize_first: If true, preprocess the token to be lowercase, no punctuation, etc.

        Returns:
            `np.ndarray` of shape (self.n_neurons,) and dtype np.int8
        """
        if self.is_unknown_token(word): return self.unknown_embedding_info

        dense_info = self.get_dense_embedding(word)
        act = dense_info['embedding']
        i_sorted = np.argsort(-act)
        act_sort = act[i_sorted]
        thr = (act_sort[hash_length - 1] + act_sort[hash_length]) / 2.0
        binary = (act > thr).astype(np.int8)
        return {
            "token": dense_info['token'],
            "id": dense_info['id'],
            "embedding": binary
        }

    def get_dense_embedding(self, word: str):
        """Get a context-independent word embedding for a given token

        Args:
            token: A token (in the vocabulary) to get the word-embedding of
            hash_length: The number of non-zero entries in the sparse embedding

        Returns:
            `np.ndarray` of shape (self.n_neurons,) and dtype np.float64
        """
        if self.is_unknown_token(word): return self.unknown_embedding_info

        token = self.tokenizer.tokenize(word)[0]
        tok_id = self.tokenizer.token2id(token)
        activation_scores = self.synapses[:, self.n_vocab + tok_id] # Target word embedding is stored in second compartment of matrix

        return {
            "token": token,
            "id": tok_id,
            "embedding": activation_scores
        }