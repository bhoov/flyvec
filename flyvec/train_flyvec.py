# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/03_Train Flyvec.ipynb (unless otherwise specified).

__all__ = ['BIN', 'model_descriptor', 'model_arrays', 'cuda_helpers', 'model', 'prune_input', 'init']

# Cell
import os
from pathlib import Path
import ctypes
from ctypes import *
import array as arr
import sys
import argparse
import time
import numpy
import flyvec.path_fixes as pf
import subprocess as sp

# Cell
# Load binaries
BIN = pf.CU_BIN

model_descriptor = ctypes.CDLL(str(BIN / 'model_descriptor.so'))
model_arrays = ctypes.CDLL(str(BIN / 'model_arrays.so'))
cuda_helpers = ctypes.CDLL(str(BIN / 'cuda_helpers.so'))
model = ctypes.CDLL(str(BIN / 'cuda_funcs.so'))
prune_input = ctypes.CDLL(str(BIN / 'prune_input.so'))

# Cell
from fastcore.script import *

@call_parse
def init():
    sp.call(["sh", str(BIN / "short_make")], cwd=str(BIN))