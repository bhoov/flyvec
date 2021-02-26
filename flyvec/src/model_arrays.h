#ifndef MODEL_ARRAYS_H
#define MODEL_ARRAYS_H

#include "model_descriptor.h"
#include <cuda.h>



class model_arrays {

   public:
     float **synapses; //weights,output
     float **ds;   //local data

     float **tot_input; //local data
     float **tot_input_raw; //additional storage for tot_input - needed with word_inv_frequency scaling;
     float **input;  //local data/minibatch
     int **input_sparse_indx; //local data - for sparse input, indices 
     float *xx;      //local data
     float *m_max_counter; //local data
     int **max_IDs;       //local data

     unsigned long long *rand_vals; //random values      
     uint64_t *indices; //indices [0 Ns_1-1] that will be reshuffled 

     //data can be provided as unsigned char or fp32
     float **INPUT_DATA_float;
     unsigned char **INPUT_DATA_uchar;
     int ** INPUT_DATA_int;
     
     float *word_inv_frequency;

     double report_model_memory_required(model_descriptor descr);

     void allocate_model_memory (model_descriptor descr, int device_ID);
     void free_model_memory (model_descriptor descr);

     void initialize_model_memory(model_descriptor descr, int device_ID);
     void push_model_memory_to_GPU(model_descriptor descr, int device_ID);

     void reshuffle_indices(model_descriptor descr, int deviceID);

};



#endif

