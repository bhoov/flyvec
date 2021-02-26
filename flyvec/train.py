import os
from pathlib import Path
import ctypes
from ctypes import *
import array as arr
import sys
import argparse
import time
import numpy


# Load binaries
ME = Path(os.path.abspath(__file__)).parent
SRC = ME / "src"
BIN = SRC

model_descriptor = ctypes.CDLL(str(BIN / 'model_descriptor.so'))
model_arrays = ctypes.CDLL(str(BIN / 'model_arrays.so'))
cuda_helpers = ctypes.CDLL(str(BIN / 'cuda_helpers.so'))
model = ctypes.CDLL(str(BIN / 'cuda_funcs.so'))
prune_input = ctypes.CDLL(str(BIN / 'prune_input.so'))

# Assign values to memory
py_prune_input_data = prune_input.prune_input_data
py_prune_input_data.rgtypes=[c_void_p,c_uint64,c_uint64]

py_getnum_samples = prune_input.getnum_samples
py_getnum_samples.rgtypes=[c_void_p,c_uint64,c_uint64]
py_getnum_samples.restype = c_uint64

py_compute_offset_phrases = prune_input.compute_offset_phrases
py_compute_offset_phrases.rgtypes=[c_void_p, c_void_p, c_uint64, c_uint64]

py_model_create_descriptor = model_descriptor.model_create_descriptor
py_model_create_descriptor.rgtypes=[c_int32]
py_model_create_descriptor.restype = ctypes.c_void_p

py_set_model_param_int = model_descriptor.set_model_param_int
py_set_model_param_float = model_descriptor.set_model_param_float
py_compute_model_derived_parameters = model_descriptor.compute_model_derived_parameters

py_set_model_param_int.rgtypes=[c_uint64,c_char_p,c_void_p]
py_set_model_param_float.rgtypes=[c_float,c_char_p,c_void_p]

py_compute_model_derived_parameters.rgtypes=[c_void_p]

py_copy_model_params = model_descriptor.copy_model_params
py_copy_model_params.rgtypes=[c_void_p,c_int32,c_int32]


py_model_create_arrays = model_arrays.model_create_arrays
py_model_create_arrays.restype = ctypes.c_void_p
py_model_create_arrays.rgtypes=[c_int32]

py_model_arrays_allocate_memory = model_arrays.model_arrays_allocate_memory
py_model_arrays_allocate_memory.rgtypes=[c_void_p,c_void_p,c_int32]


py_model_arrays_reshuffle_indices = model_arrays.do_reshuffle_indices
py_model_arrays_reshuffle_indices.rgtypes=[c_void_p,c_void_p,c_int32]

py_compute_inverse_word_frequency = model_arrays.compute_inverse_word_frequency
py_compute_inverse_word_frequency.rgtypes=[c_void_p,c_void_p,c_void_p,c_uint64,c_int32]

python_get_cuda_pinned_memory = cuda_helpers.do_gpu_cudaHostAlloc
python_get_cuda_pinned_memory.restype = ctypes.c_void_p

python_get_cuda_managed_memory = cuda_helpers.do_gpu_cudaMallocManaged
python_get_cuda_managed_memory.restype = ctypes.c_void_p


py_run_epoch_INPUT_AS_IMAGE = model.launch_epoch_INPUT_AS_IMAGE
py_run_epoch_INPUT_AS_IMAGE.rgtypes=[c_void_p,c_void_p,c_int32,c_uint64,c_uint64]

py_run_epoch_INPUT_AS_FLOAT = model.launch_epoch_INPUT_AS_FLOAT
py_run_epoch_INPUT_AS_FLOAT.rgtypes=[c_void_p,c_void_p,c_int32,c_uint64,c_uint64]


py_run_epoch_INPUT_AS_INT = model.launch_epoch_INPUT_AS_INT
py_run_epoch_INPUT_AS_INT.rgtypes=[c_void_p,c_void_p,c_int32,c_uint64,c_uint64]

py_model_arrays_get_data_pointer = model_arrays.get_data_pointer
py_model_arrays_get_data_pointer.restype = ctypes.c_void_p
py_model_arrays_get_data_pointer.rgtypes = [c_char_p, c_void_p, c_int32]

num_gpus = cuda_helpers.get_cuda_num_devices()
print('num_gpus=', num_gpus)

print("py_model_create_arrays and descriptor...")
MODEL_DSCR = py_model_create_descriptor( c_int32(num_gpus) )
MODEL_DATA = py_model_create_arrays( c_int32(num_gpus) )

parser = argparse.ArgumentParser()
parser.add_argument("encodings_source", type=str, help="""Path to the tokens encoded as an ID. (eltype should be
        np.int32)""")
parser.add_argument("offsets_source", type=str, help="""Path to the offsets array indicating where each value of the
array indicates where a chunk of text should start. (eltype should be np.uint64, first value should be 0 if)""")
parser.add_argument("--output_dir", "-o", default=".", type=str, help="""Directory in which to save the checkpoints.
        Created if it does not exist""")
parser.add_argument("--save_every", "-s", type=int, default=1, help="""How many epochs to run before saving a checkpoint""")
parser.add_argument("--ckpt_prefix", default="flyvec_model_", type=str, help="""Prefix to name each checkpoint.
        Additional parameter choices are inserted into the checkpoint name.""")
parser.add_argument("--starting_checkpoint", type=str, default=None, help="""Path to .npy file of saved checkpoint""")
parser.add_argument("--W", default=11, type=int, help="Size of the W-gram sliding window used to train the word vectors")
parser.add_argument("--hid", default=400, type=int, help="Number of hidden units (neurons). Do not change")
parser.add_argument("--initial_learning_rate", default=0.0002, type=float, help="Initial learning rate")
parser.add_argument("--delta", default=0, type=float, help="From equation")
parser.add_argument("--mu", default=0, type=float, help="""If no checkpoint provided, use this as mean for random normal initialization of synapses""")
parser.add_argument("--sigma", default=0, type=float, help="""If no checkpoint provided, use this as stdev for random normal initialization of synapses""")
parser.add_argument("--Nep", default=15, type=int, help="Maximum number of epochs, fewer if starting from a checkpoint")
parser.add_argument("--batch_size", "-b", default=10000, type=int, help="Minibatch size")
parser.add_argument("--prec", default=1.0E-30, type=float, help="Precision, avoid dividing by 0")

args = parser.parse_args()

# Obsolete Parameters
stride = int(1)
IM_HEIGHT = int(1)
IM_WIDTH = int(11)
Nchannels = int(1)
# m = 2
# p = 2
# parser.add_argument("--stride", default=1, type=int, help="Stride. Do not change.")
# parser.add_argument("--IM_HEIGHT", default=1, type=int, help="Height of the image data. Obsolete.")
# parser.add_argument("--IM_WIDTH", default=11, type=int, help="Width of the image data. Obsolete.")
# parser.add_argument("--Nchannels", default=1, type=int, help="Number of channels in the image data. Obsolete.")
# parser.add_argument("--m", default=2, type=int, help="Parameter from equation. Obsolete.")
# parser.add_argument("--p", default=2, type=int, help="Parameter from equation. Obsolete.")

# Do not change
frequency_scaling = 1 # 1 or 0, true or false
Lmid = 1.0
Lbase = 1.0

sparse_input = 1 # 1 or 0, true or false, make input sparse and only store column


# Create output directory
output_dir = Path(args.output_dir)
if not output_dir.exists(): output_dir.mkdir(parents=True)
OUTPUT_NAME=f"{args.ckpt_prefix}_H_{args.hid}_W_{args.W}_LR_{args.initial_learning_rate}_"
OUTPUT = str(output_dir / OUTPUT_NAME)

# Allocate memory for initial encodings
input_data_on_disk_encoding = numpy.load(args.encodings_source, mmap_mode='r')
Ns_1 = input_data_on_disk_encoding.shape[0]

# Cannot push to GPU
INPUT = python_get_cuda_pinned_memory(ctypes.c_uint64(Ns_1*ctypes.sizeof(ctypes.c_int32)))

#create numpy array to store input data, use memory allocated for INPUT
INPUT_data_pointer = ctypes.cast(INPUT,ctypes.POINTER(ctypes.c_int32))
INPUT_np_array = numpy.ctypeslib.as_array(INPUT_data_pointer,shape=(Ns_1,))

print('Copying input...')
numpy.copyto(INPUT_np_array,input_data_on_disk_encoding)

vocabulary_size = numpy.uint64(numpy.max(INPUT_np_array)+1)
N = numpy.uint64(vocabulary_size*2)
print(f"Determined a vocabulary size of {vocabulary_size}")

input_data_on_disk_offsets = numpy.load(args.offsets_source,mmap_mode='r')
Number_of_sentences = input_data_on_disk_offsets.shape[0] - 1

Number_of_phrases = prune_input.getnum_samples(ctypes.c_void_p(input_data_on_disk_offsets.__array_interface__['data'][0]),
        c_uint64(Number_of_sentences), c_uint64(args.W) )
print('Number of phrases: ', Number_of_phrases)

#allocate memory for offsets for phrases of size W
INPUT_phrases_offsets = python_get_cuda_pinned_memory(ctypes.c_uint64((Number_of_phrases+1)*ctypes.sizeof(ctypes.c_int64) ))

#compute offsets
py_compute_offset_phrases(c_void_p(INPUT_phrases_offsets),
        c_void_p(input_data_on_disk_offsets.__array_interface__['data'][0]), c_uint64(Number_of_sentences),
        c_uint64(args.W))

Ns_1 = Number_of_phrases

model_descriptor.print_model_params(ctypes.c_void_p(MODEL_DSCR))

py_set_model_param_int(IM_HEIGHT,b'IM_HEIGHT',ctypes.c_void_p(MODEL_DSCR))

py_set_model_param_int(IM_WIDTH,b'IM_WIDTH',ctypes.c_void_p(MODEL_DSCR))
py_set_model_param_int(Nchannels,b'Nchannels',ctypes.c_void_p(MODEL_DSCR))
py_set_model_param_int(ctypes.c_uint64(Ns_1),b'Ns_1',ctypes.c_void_p(MODEL_DSCR))
#
py_set_model_param_int(args.W, b'W', ctypes.c_void_p(MODEL_DSCR))
py_set_model_param_int(stride, b'ST', ctypes.c_void_p(MODEL_DSCR))
py_set_model_param_int(args.hid, b'hid', ctypes.c_void_p(MODEL_DSCR))
py_set_model_param_int(args.batch_size, b'Num', ctypes.c_void_p(MODEL_DSCR))

py_set_model_param_int(c_uint64(vocabulary_size), b'vocabulary_size',ctypes.c_void_p(MODEL_DSCR))
py_set_model_param_int(sparse_input, b'sparse_input',ctypes.c_void_p(MODEL_DSCR))
py_set_model_param_int(frequency_scaling, b'frequency_scaling',ctypes.c_void_p(MODEL_DSCR))

py_set_model_param_float(c_float(args.initial_learning_rate), b'initial_learning_rate', ctypes.c_void_p(MODEL_DSCR))
py_set_model_param_float(c_float(args.delta), b'delta', ctypes.c_void_p(MODEL_DSCR))
py_set_model_param_float(c_float(args.prec), b'prec', ctypes.c_void_p(MODEL_DSCR))
py_set_model_param_float(c_float(args.mu), b'mu', ctypes.c_void_p(MODEL_DSCR))
py_set_model_param_float(c_float(args.sigma), b'sigma', ctypes.c_void_p(MODEL_DSCR))
py_set_model_param_float(c_float(Lmid), b'Lmid', ctypes.c_void_p(MODEL_DSCR))
py_set_model_param_float(c_float(Lbase), b'Lbase', ctypes.c_void_p(MODEL_DSCR))
py_compute_model_derived_parameters(ctypes.c_void_p(MODEL_DSCR))

for gpu in range(1,num_gpus,1):
    py_copy_model_params( ctypes.c_void_p(MODEL_DSCR), c_int32(0), c_int32(gpu) )

model_descriptor.print_model_params(ctypes.c_void_p(MODEL_DSCR))

py_model_arrays_allocate_memory(ctypes.c_void_p(MODEL_DSCR), ctypes.c_void_p(MODEL_DATA),c_int32(num_gpus))

print('setting INPUT pointer')
for gpu in range(0,num_gpus,1):
    model_arrays.set_up_INPUT_pointer(b'INPUT', ctypes.c_void_p(MODEL_DATA), ctypes.c_void_p(MODEL_DSCR), ctypes.c_void_p(INPUT),ctypes.c_void_p(INPUT_phrases_offsets), b'i4',c_int32(gpu))

if frequency_scaling==1 :
    print('Computing inverse word frequency...')
    for gpu in range(0,num_gpus,1):
        py_compute_inverse_word_frequency(ctypes.c_void_p(MODEL_DATA), ctypes.c_void_p(MODEL_DSCR), ctypes.c_void_p(INPUT), c_uint64(input_data_on_disk_encoding.shape[0]),  c_int32(gpu) )

#model_arrays.push_INPUT_memory_to_GPU(ctypes.c_void_p(MODEL_DATA), ctypes.c_void_p(MODEL_DSCR),c_int32(0), b'i4')


#initialize SYNAPSES
print('Setting initial weights...')
epoch_start = 0
if args.starting_checkpoint is not None:
    R=numpy.load(args.starting_checkpoint,mmap_mode='r')
    # Assume that the epoch_starting_number is the last value of the checkpoint name
    epoch_start = int(args.starting_checkpoint.split("_")[-1])
else:
    R = numpy.float32(numpy.random.normal(args.mu, args.sigma, (args.hid,N)))

#push the same initial model data into all GPUs
for gpu in range(0,num_gpus,1):
    SYNAPSES_data_pointer = ctypes.cast(py_model_arrays_get_data_pointer(b'synapses',ctypes.c_void_p(MODEL_DATA),c_int32(gpu)),ctypes.POINTER(ctypes.c_float))
    SYNAPSES_np_array = numpy.ctypeslib.as_array(SYNAPSES_data_pointer,shape=(args.hid,N))
    numpy.copyto(SYNAPSES_np_array,R)

SYNAPSES_data_pointer = ctypes.cast(py_model_arrays_get_data_pointer(b'synapses',ctypes.c_void_p(MODEL_DATA),c_int32(0)),ctypes.POINTER(ctypes.c_float))
SYNAPSES_np_array = numpy.ctypeslib.as_array(SYNAPSES_data_pointer,shape=(args.hid,N))

#push INPUT data to the GPU (depending on available memory)
for i, ep in enumerate(range(epoch_start,args.Nep)):
    print('epoch ID = ',ep)
    t1 = time.time()
    py_model_arrays_reshuffle_indices(ctypes.c_void_p(MODEL_DSCR), ctypes.c_void_p(MODEL_DATA),c_int32(0))
    t11 = time.time()
    py_run_epoch_INPUT_AS_INT(ctypes.c_void_p(MODEL_DSCR), ctypes.c_void_p(MODEL_DATA), c_int32(num_gpus),
            c_uint64(ep), c_uint64(args.Nep) )
    t2 = time.time()
    print('time per epoch = ',t2-t1,'[s]', '  py_model_arrays_reshuffle_indices time = ',t11-t1,'[s]')
    if ((i+1) % args.save_every) == 0 or i == (args.Nep - 1):
        numpy.save(OUTPUT+str(ep),SYNAPSES_np_array)
