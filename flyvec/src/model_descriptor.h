#ifndef MODEL_DSCR_H
#define MODEL_DSCR_H


//for definition of uint64_t
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <curand.h>


class model_descriptor {

   public:

   //input data dimensions
   uint64_t IM_HEIGHT;
   uint64_t IM_WIDTH;
   uint64_t Nchannels;
   uint64_t Ns_1;

   //input data patch parameters
   uint64_t W;
   uint64_t ST;

   //input data  derived parameters
   //number of patches in row and column dimensions
   uint64_t HEIGHT_W_ST_block;
   uint64_t WIDTH_W_ST_block;
   uint64_t Ns; //total number of patches


   //number of hidden layers in input data
   uint64_t hid;
   //number of elements in each layer
   uint64_t N;  
 
   uint64_t Num;  //minibatch size
   uint64_t m;
   uint64_t p;


   float initial_learning_rate;
   float delta;
   float prec;

   float mu;
   float sigma;

   float Lmid;
   float Lbase;


   //for input data stored as a sparse operator
   uint64_t sparse_input; 
   uint64_t vocabulary_size;

   // normalized inverse frequency for words
   uint64_t frequency_scaling;

   uint64_t compute_num_patches(uint64_t IM_DIM, uint64_t Window_size, uint64_t Stride);

   curandGenerator_t cu_rand_gen_handle;

};

#endif

