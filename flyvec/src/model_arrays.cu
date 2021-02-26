#include "model_descriptor.h"
#include "model_arrays.h"
#include "memory_allocator.h"




//acc_cuda.cu
void fill( uint64_t N, float* data, float value);
void fill( uint64_t N, uint64_t* data, uint64_t value);
void fill( uint64_t N, int* data, int value);


extern "C"
void * model_create_arrays(int num_gpus){
  model_arrays * MA = new model_arrays[num_gpus];
  return (void*) MA;
}


extern "C"
void model_arrays_allocate_memory (void * descr_in, void *MA_in, int num_gpus){
  model_descriptor * descr = (model_descriptor*) descr_in;
  model_arrays * MA = (model_arrays*) MA_in;

  for (int gpu = 0; gpu < num_gpus; ++gpu){

    double GB = MA[gpu].report_model_memory_required(descr[gpu]);
    fprintf(stderr,"model_arrays_allocate_memory: GPU ID = %d,  allocating  %g [GB] using CUDA MANAGED MEMORY\n",gpu, GB);
    MA[gpu].allocate_model_memory(descr[gpu], gpu);
    fprintf(stderr,"allocate_model_memory - done\n");
    MA[gpu].initialize_model_memory(descr[gpu], gpu);
    fprintf(stderr,"initialize_model_memory - done\n");
    MA[gpu].push_model_memory_to_GPU(descr[gpu], gpu);
    fprintf(stderr,"push_model_memory_to_GPU - done\n");
  }  
  //return to GPU 0
  cudaSetDevice(0);
  
}


double model_arrays::report_model_memory_required(model_descriptor descr){

  double GB;
  double memory_needed_GB = 0;
  uint64_t hid = descr.hid;
  uint64_t Num = descr.Num;


  uint64_t N, N_1;
  if (descr.sparse_input == 0){
     N = descr.W * descr.W * descr.Nchannels;
  }
  else{
    N_1 =  descr.W;
    N = 2 * descr.vocabulary_size;
  }



  //synapses
  GB = (double)(hid*sizeof(float*) + hid*N*sizeof(float)) / (1024.0*1024.0*1024.0);
  fprintf(stderr,"memory for synapses: %g [GB]\n",GB);
  memory_needed_GB += GB;
  //ds
  GB = (double)(hid*sizeof(float*) + hid*N*sizeof(float)) / (1024.0*1024.0*1024.0);
  fprintf(stderr,"memory for ds: %g [GB]\n",GB);
  memory_needed_GB += GB;
  //tot_input
  GB = (double)(Num*sizeof(float*) + hid*Num*sizeof(float)) / (1024.0*1024.0*1024.0);
  fprintf(stderr,"memory for tot_input: %g [GB]\n",GB);
  memory_needed_GB += GB;
  if (descr.frequency_scaling == 1){
    fprintf(stderr,"memory for tot_input_raw: %g [GB]\n",GB);
    memory_needed_GB += GB;
  }

  //input
  if (descr.sparse_input == 0)
    GB = (double)(Num*sizeof(float*) + Num*N*sizeof(float)) / (1024.0*1024.0*1024.0);
  else
    GB = (double)(Num*sizeof(int*) + Num*N_1*sizeof(int)) / (1024.0*1024.0*1024.0);
  fprintf(stderr,"memory for input: %g [GB]\n",GB);
  memory_needed_GB += GB;
    //max_IDs
  GB = (double)(Num*sizeof(int*) + Num*2*sizeof(int)) / (1024.0*1024.0*1024.0);
  fprintf(stderr,"memory for max_IDs: %g [GB]\n",GB);
  memory_needed_GB += GB;
   //xx
  GB = (double)(hid*sizeof(float)) / (1024.0*1024.0*1024.0);
  fprintf(stderr,"memory for xx: %g [GB]\n",GB);
  memory_needed_GB += GB;

  GB = (double)(hid*sizeof(float)) / (1024.0*1024.0*1024.0);
  fprintf(stderr,"memory for m_max_counter : %g [GB]\n",GB);
  memory_needed_GB += GB;

  if (descr.frequency_scaling == 1){
    GB = (double)(descr.vocabulary_size * sizeof(float)) / (1024.0*1024.0*1024.0);
    fprintf(stderr,"memory for word_inv_frequency : %g [GB]\n",GB);
    memory_needed_GB += GB;
  }

  return memory_needed_GB;
}


void model_arrays::allocate_model_memory (model_descriptor descr, int deviceID){

     cudaSetDevice(deviceID);
    
     uint64_t N = descr.N;

     printf("model_arrays::allocate_model_memory:  N = %lu\n",N);
     uint64_t two = 2;
     synapses  = mem_alloc2D_cuda_managed<float,uint64_t>(descr.hid, N);
     ds        = mem_alloc2D_cuda_managed<float,uint64_t>(descr.hid, N);
     tot_input = mem_alloc2D_cuda_managed<float,uint64_t>(descr.Num, descr.hid);
     if (descr.sparse_input == 0)
        input     = mem_alloc2D_cuda_managed<float,uint64_t>(descr.Num, N);
     else
        input_sparse_indx     = mem_alloc2D_cuda_managed<int,uint64_t>(descr.Num, descr.W);

     if (descr.frequency_scaling == 1){
        word_inv_frequency = mem_alloc1D_cuda_managed<float,uint64_t>(descr.vocabulary_size);
        tot_input_raw = mem_alloc2D_cuda_managed<float,uint64_t>(descr.Num, descr.hid);
     }

     max_IDs   = mem_alloc2D_cuda_managed<int  ,uint64_t>(descr.Num, two);
     xx        = mem_alloc1D_cuda_managed<float,uint64_t>(descr.hid);
     m_max_counter = mem_alloc1D_cuda_managed<float,uint64_t>(descr.hid);
     if (deviceID == 0){
       rand_vals = mem_alloc1D_cuda_host_registered<unsigned long long,uint64_t>(descr.Ns_1);
       indices   = mem_alloc1D_cuda_host_registered<uint64_t,uint64_t>(descr.Ns_1);
     }
     else {
       rand_vals = NULL;
       indices = NULL;
     }
}

void model_arrays::free_model_memory (model_descriptor descr){
     mem_free2D_cuda_managed(synapses); 
     mem_free2D_cuda_managed(ds);
     mem_free2D_cuda_managed(tot_input);
     mem_free2D_cuda_managed(input);
     mem_free2D_cuda_managed(max_IDs);
     mem_free1D_cuda_managed(xx);
     mem_free1D_cuda_managed(m_max_counter);
     if (word_inv_frequency != NULL){
       mem_free1D_cuda_managed(word_inv_frequency); 
       mem_free2D_cuda_managed(tot_input_raw);
     }
     if (rand_vals != NULL) 
        mem_free1D_host_registered(rand_vals);
     if (indices != NULL) 
        mem_free1D_host_registered(indices); 
}

void model_arrays::initialize_model_memory(model_descriptor descr, int deviceID){

    cudaSetDevice(deviceID);
    uint64_t N = descr.N;
    uint64_t two = 2;

    fill( descr.hid * N, synapses[0], 0.0);
    fill( descr.hid * N, ds[0], 0.0);
    fill( descr.Num * descr.hid, tot_input[0], 0.0);
    if (word_inv_frequency != NULL)
      fill( descr.Num * descr.hid, tot_input_raw[0], 0.0);
    if (descr.sparse_input == 0)
       fill( descr.Num * N, input[0], 0.0);
    else
       fill( descr.Num * descr.W, input_sparse_indx[0], 0);

    fill( descr.Num * two, max_IDs[0], 0);
    fill( descr.hid, xx, 0.0);
    fill( descr.hid, m_max_counter, 0.0);
    if (indices != NULL){
       for (uint64_t i = 0; i < descr.Ns_1; ++i) indices[i] = i;
      

      // fprintf(stderr,"line=%d\n",__LINE__);
      if (deviceID == 0){
        curandGenerateLongLong(descr.cu_rand_gen_handle, rand_vals, descr.Ns_1);
        cudaDeviceSynchronize();
      }
      // fprintf(stderr,"line=%d\n",__LINE__);

   
      for (uint64_t i = 0; i < descr.Ns_1; ++i){
        uint64_t itmp = indices[i];
        uint64_t ii = rand_vals[i]%descr.Ns_1;
        indices[i] = indices[ii];
        indices[ii] = itmp;
      }
    }


}
template <class Tptr, class Tdim>
void prefetch_to_device(Tptr **DATA, Tdim dim1,  Tdim dim2, int deviceID){
  cudaMemPrefetchAsync (DATA, dim1*sizeof(Tptr*), deviceID );
  cudaMemPrefetchAsync (DATA[0], dim1*dim2*sizeof(Tptr), deviceID );
  cudaDeviceSynchronize();
  #ifdef USE_MANAGED
  cudaMemAdvise(DATA, dim1*sizeof(Tptr*),cudaMemAdviseSetPreferredLocation,deviceID);
  cudaMemAdvise(DATA[0],dim1*dim2*sizeof(Tptr),cudaMemAdviseSetPreferredLocation,deviceID);
  #endif
}

template <class Tptr, class Tdim>
void prefetch_to_device(Tptr *DATA, Tdim dim1,  int deviceID){
  cudaMemPrefetchAsync (DATA, dim1*sizeof(Tptr), deviceID );
  cudaDeviceSynchronize();
  #ifdef USE_MANAGED
  cudaMemAdvise(DATA, dim1*sizeof(Tptr),cudaMemAdviseSetPreferredLocation,deviceID);
  #endif
}



void model_arrays::push_model_memory_to_GPU(model_descriptor descr, int deviceID){

   cudaSetDevice(deviceID);
   uint64_t N = descr.N;
   uint64_t two = 2;

   prefetch_to_device(ds, descr.hid,  N, deviceID);
   prefetch_to_device(tot_input, descr.Num,  descr.hid, deviceID);
   if (descr.sparse_input == 0)
     prefetch_to_device(input, descr.Num,  N, deviceID);
   else
     prefetch_to_device(input_sparse_indx, descr.Num, descr.W, deviceID);

   prefetch_to_device(synapses, descr.hid,  N, deviceID);
   prefetch_to_device(xx, descr.hid, deviceID);
   prefetch_to_device(max_IDs, descr.Num, two, deviceID);
   if (word_inv_frequency != NULL){
     prefetch_to_device(word_inv_frequency, descr.vocabulary_size, deviceID); 
     prefetch_to_device(tot_input_raw, descr.Num,  descr.hid, deviceID);
   }

   //we are not prefetching m_max_counter, is it will not be needed if delta == 0
   //if delta is non-zero , this array will be paged to a GPU on demand.

}
extern "C"
void push_INPUT_memory_to_GPU(void * MA_in,  void  *descr_in, int deviceID,const char* type ){

  model_descriptor * DSCR = (model_descriptor*) descr_in;
  model_arrays * MA = (model_arrays*) MA_in;

  double GB = MA[0].report_model_memory_required(DSCR[0]);
  double GPU_MEMORY_GB = 16.0; //GB
  double SAFETY_MARGINS = 0.5; //GB  
  double available_memory = GPU_MEMORY_GB - SAFETY_MARGINS - GB;
  fprintf(stderr,"GPU_MEMORY_GB=%f, SAFETY_MARGINS=%g, GB=%g\n",GPU_MEMORY_GB,SAFETY_MARGINS,GB);


  double page_size = 2.0/1024.0;  //in GB
  double INPUT_size = 0.0;
  uint64_t dim1 = DSCR[0].Ns_1;
  uint64_t dim2 = DSCR[0].IM_HEIGHT * DSCR[0].IM_WIDTH * DSCR[0].Nchannels;
  uint64_t INPUT_length = dim1*dim2;
  fprintf(stderr,"dim1 = %lu, dim2 = %lu, INPUT_length = %lu\n",dim1, dim2,INPUT_length);


  if ( (0==strcmp(type,"F")) || (0==strcmp(type,"f")) ){
        INPUT_size = (double) INPUT_length*sizeof(float)/(1024.0*1024.0*1024.0);
        INPUT_size = INPUT_size + (double) dim1*sizeof(float*)/(1024.0*1024.0*1024.0);                
        if (INPUT_size <=  available_memory) {//prefetch everything and pin to GPU
          prefetch_to_device(MA[0].INPUT_DATA_float, dim1, dim2, deviceID);
        }
        else {
           fprintf(stderr,"INPUT_size (%f) > available_memory (%f)\n",INPUT_size,available_memory);
        }
  }
  else if ( (0==strcmp(type,"I4")) || (0==strcmp(type,"i4")) ){

        INPUT_size = (double) INPUT_length*sizeof(int)/(1024.0*1024.0*1024.0);
        INPUT_size = INPUT_size +  (double) dim1*sizeof(int*)/(1024.0*1024.0*1024.0);
        if (INPUT_size <=  available_memory) {//prefetch everything and pin to GPU
          prefetch_to_device(MA[0].INPUT_DATA_int, dim1, dim2, deviceID);
        }
        else {
           fprintf(stderr,"INPUT_size (%f) > available_memory (%f)\n",INPUT_size,available_memory);
        }
  }
  else if ( (0==strcmp(type,"U1")) || (0==strcmp(type,"u1")) ){
        INPUT_size = (double) INPUT_length*sizeof(unsigned char)/(1024.0*1024.0*1024.0);
        INPUT_size = INPUT_size + (double) dim1*sizeof(unsigned char*)/(1024.0*1024.0*1024.0);
        if (INPUT_size <=  available_memory) {//prefetch everything and pin to GPU
          prefetch_to_device(MA[0].INPUT_DATA_uchar, dim1, dim2, deviceID);
        }
        else {
           fprintf(stderr,"INPUT_size (%f) > available_memory (%f)\n",INPUT_size,available_memory);
        }
   }

}

extern "C"
void* get_data_pointer(const char *s, void* MA_in, int deviceID){
  model_arrays * MA = (model_arrays*) MA_in;
  if (0==strcmp(s,"synapses")){
      // fprintf(stderr,"get_data_pointer:synapses[0] = %p\n",MA[deviceID].synapses[0]);
      return (void*) ( MA[deviceID].synapses[0]);
  }
  else {
    // printf("get_data_pointer: unknown array %s\n",s);
    return NULL;
  }
}



extern "C"
void set_data_pointer(const char *s, void* MA_in, void * descr_in, void *ptr, const char *type, int deviceID){
  model_arrays * MA = (model_arrays*) MA_in;
  model_descriptor * DSCR = (model_descriptor*) descr_in;

  if (0==strcmp(s,"INPUT")){
     uint64_t N = DSCR[deviceID].IM_HEIGHT * DSCR[deviceID].IM_WIDTH * DSCR[deviceID].Nchannels;
     if ( (0==strcmp(type,"F")) || (0==strcmp(type,"f")) ){
        MA[deviceID].INPUT_DATA_float = mem_alloc1D_cuda_managed<float*,uint64_t>(DSCR[deviceID].Ns_1);
        MA[deviceID].INPUT_DATA_float[0] = (float*) ptr;
        for (uint64_t i = 1; i < DSCR[deviceID].Ns_1; ++i)
            MA[deviceID].INPUT_DATA_float[i] = MA[deviceID].INPUT_DATA_float[0] + i*N;
        prefetch_to_device(MA[deviceID].INPUT_DATA_float,DSCR[deviceID].Ns_1, deviceID); 

     }
     else if ( (0==strcmp(type,"I4")) || (0==strcmp(type,"i4")) ){
        MA[deviceID].INPUT_DATA_int = mem_alloc1D_cuda_managed<int*,uint64_t>(DSCR[deviceID].Ns_1);
        MA[deviceID].INPUT_DATA_int[0] = (int*) ptr;
        for (uint64_t i = 1; i < DSCR[deviceID].Ns_1; ++i)
            MA[deviceID].INPUT_DATA_int[i] = MA[deviceID].INPUT_DATA_int[0] + i*N;
        prefetch_to_device(MA[deviceID].INPUT_DATA_int,DSCR[deviceID].Ns_1, deviceID);
     }
     else if ( (0==strcmp(type,"U1")) || (0==strcmp(type,"u1")) ){
        MA[deviceID].INPUT_DATA_uchar = mem_alloc1D_cuda_managed<unsigned char*,uint64_t>(DSCR[deviceID].Ns_1);
        MA[deviceID].INPUT_DATA_uchar[0] = (unsigned char*) ptr;
        for (uint64_t i = 1; i < DSCR[deviceID].Ns_1; ++i)
            MA[deviceID].INPUT_DATA_uchar[i] = MA[deviceID].INPUT_DATA_uchar[0] + i*N;
        prefetch_to_device(MA[deviceID].INPUT_DATA_uchar,DSCR[deviceID].Ns_1, deviceID);

     }
     else
       printf("set_data_pointer: unknown array type %s\n",s);
  }
  else {
    printf("set_data_pointer: unknown array %s\n",s);
  }
}

extern "C"
void set_up_INPUT_pointer(const char *s, void* MA_in, void * descr_in, void *ptr_in, void *ptr_offsets_in,  const char *type, int deviceID){
  model_arrays * MA = (model_arrays*) MA_in;
  model_descriptor * DSCR = (model_descriptor*) descr_in;
  uint64_t N = DSCR[deviceID].N;
  uint64_t *ptr_offsets = (uint64_t*) ptr_offsets_in;

  if ( (0==strcmp(type,"F")) || (0==strcmp(type,"f")) ){
      float *ptr = (float*) ptr_in;
      MA[deviceID].INPUT_DATA_float = mem_alloc1D_cuda_host_registered<float*,uint64_t>(DSCR[deviceID].Ns_1);
      for (uint64_t i = 0; i < DSCR[deviceID].Ns_1; ++i){
          MA[deviceID].INPUT_DATA_float[i] =  &ptr[ ptr_offsets[i] ];
      }
  }
  else if ( (0==strcmp(type,"I4")) || (0==strcmp(type,"i4")) ){

    int *ptr = (int*) ptr_in;
     if (deviceID == 0){
       MA[deviceID].INPUT_DATA_int = mem_alloc1D_cuda_host_registered<int*,uint64_t>(DSCR[deviceID].Ns_1);
       for (uint64_t i = 0; i < DSCR[deviceID].Ns_1; ++i){
          MA[deviceID].INPUT_DATA_int[i] =  &ptr[ ptr_offsets[i] ];
       }
     }
     else{
        MA[deviceID].INPUT_DATA_int = MA[0].INPUT_DATA_int;
     }
  } 
  else if ( (0==strcmp(type,"U1")) || (0==strcmp(type,"u1")) ){
     unsigned char *ptr = (unsigned char*) ptr_in;
     MA[deviceID].INPUT_DATA_uchar = mem_alloc1D_cuda_host_registered<unsigned char*,uint64_t>(DSCR[deviceID].Ns_1);
     for (uint64_t i = 0; i < DSCR[deviceID].Ns_1; ++i){
          MA[deviceID].INPUT_DATA_uchar[i] =  &ptr[ ptr_offsets[i] ];
     }
  }
  else
    printf("set_up_INPUT_pointer: unknown array type %s\n",s);
}

extern "C"
void compute_inverse_word_frequency(void *MA_in, void * descr_in,  void *input_in, uint64_t Nwords, int deviceID){

  model_arrays * MA = (model_arrays*) MA_in;
  model_descriptor * DSCR = (model_descriptor*) descr_in;
  if (DSCR[deviceID].frequency_scaling == 0) return;
  int *ptr = (int*) input_in;
 
 
  if (MA[deviceID].word_inv_frequency == NULL)
    fprintf(stderr,"compute_inverse_word_frequency:  word_inv_frequency array has not been allocated\n");
 

 
  for (uint64_t i = 0; i < DSCR[deviceID].vocabulary_size; ++i) 
    MA[deviceID].word_inv_frequency[i] = 0.0; 


  for (uint64_t i = 0; i < Nwords; ++i){
    if (ptr[i] >= 0)
      MA[deviceID].word_inv_frequency[ ptr[i] ] += 1.0;  
  }


  for (uint64_t i = 0; i < DSCR[deviceID].vocabulary_size; ++i){
    if (MA[deviceID].word_inv_frequency[i] != 0)
      MA[deviceID].word_inv_frequency[i] = 1.0/MA[deviceID].word_inv_frequency[i];
    else
      MA[deviceID].word_inv_frequency[i] = 0.0;
  }

/*
  //find max value and scale by its inverse
  float inv_max_frequency = 0;
  for (uint64_t i = 0; i < DSCR[deviceID].vocabulary_size; ++i)
    inv_max_frequency = inv_max_frequency > MA[deviceID].word_inv_frequency[i] ? inv_max_frequency : MA[deviceID].word_inv_frequency[i];

  for (uint64_t i = 0; i < DSCR[deviceID].vocabulary_size; ++i)
    MA[deviceID].word_inv_frequency[i] = MA[deviceID].word_inv_frequency[i] / inv_max_frequency;
*/

  fprintf(stderr,"compute_inverse_word_frequency line=%d\n",__LINE__);

}


extern "C"
void do_reshuffle_indices(void * descr_in, void *MA_in, int deviceID){
  model_arrays * MA = (model_arrays*) MA_in;
  if (MA[deviceID].indices == NULL) return;

  model_descriptor * DSCR = (model_descriptor*) descr_in;
  MA[deviceID].reshuffle_indices(DSCR[0],deviceID);
}


void model_arrays::reshuffle_indices(model_descriptor descr, int deviceID){

    cudaSetDevice(deviceID);

    curandGenerateLongLong(descr.cu_rand_gen_handle, rand_vals, descr.Ns_1);
    cudaDeviceSynchronize();

    for (uint64_t i = 0; i < descr.Ns_1; ++i){
      uint64_t itmp = indices[i];
      uint64_t ii = rand_vals[i]%descr.Ns_1;
      indices[i] = indices[ii];
      indices[ii] = itmp;
    }

}

