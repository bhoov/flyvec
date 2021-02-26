

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

#include "model_arrays.h"
#include "model_descriptor.h"
#include <omp.h>

//external function: cu_special_reduction.cu
float reduce_by_chunks_and_absmax(float **data,  unsigned long long *offsets, int **device_list, float **max_per_block, unsigned long long nblocks, float *max_abs_val, int num_devices);
void update_model_multiGPU(float **data,  unsigned long long *offsets, int **device_list, int num_devices, float** model, float scale_factor);

float reduce_by_chunks_and_absmax_sparse_2D(float ***data2D,  unsigned long long *offsets, int **device_list, float **max_per_block, unsigned long long nblocks, float *max_abs_val, int num_devices, uint64_t Num, float **xx2D);
void update_model_multiGPU_sparse_2D(float ***data2D,  unsigned long long *offsets, int **device_list, int num_devices, float*** model2D, float scale_factor, uint64_t Num, float **xx2D);


void do_reshuffle_indices(void * descr_in, void *MA_in, int deviceID);


cublasHandle_t handle;




#ifdef IMAGE_AS_CHAR
#define IMTYPE unsigned char
#else
#define IMTYPE float
#endif



#define CUCHECK(call) {                                                      \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",          \
              __FILE__, __LINE__, cudaGetErrorString( err) );                \
      fflush(stderr);                                                        \
      exit(EXIT_FAILURE);                                                    \
    } }


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



__constant__ float eps_prec[2];

template <class ELMNT_TYPE>
void split_to_chuncks(unsigned long long length, unsigned long long nchunks, unsigned long long *offsets);

void cuda_func1(int ii_start, int ii_end, float * __restrict__ synapses, float * __restrict__ ds, int p, int nowait);

__global__
void cuda_func1_even(int N, float * __restrict__ synapses, float * __restrict__ ds, int p_1);
__global__
void cuda_func1_odd(int N, float * __restrict__ synapses, float * __restrict__ ds, int p_1);


void cuda_daxpy(int N, float * __restrict__ y, float * __restrict__ x, float val);

__global__
void cuda_daxpy_kernel(int N, float * __restrict__ synapses, float * __restrict__ ds, float val);


__global__
void update_synapses(int N, float * __restrict__ synapses, float * __restrict__ *ds,
                     float * const nc);


void cuda_ds_xx_t_synapses(int N, int hid, float ** __restrict__ ds, float * __restrict__ xx, float ** __restrict__ synapses, int nowait);

__global__
void cuda_ds_xx_t_synapses_kernel(int N, int hid, float ** __restrict__ ds, float * __restrict__ xx, float ** __restrict__ synapses);



void get_max_and_m_max(  int Num, int hid, float **tot_input, int num_m,  int **ind_max_m_max, int nowait);

__global__
void get_max_and_m_max_kernel(  int Num, int hid, float **tot_input, int num_m,  int **ind_max_m_max);
__global__
void get_max_and_m_max_kernel_2(  int Num, int hid, float **tot_input, int num_m,  int **ind_max_m_max );

void func_ds_eq_ds_input(int Num, int N, float **ds, float **input, int **max_IDs, float delta, int nowait);

__global__
void func_ds_eq_ds_input_kernel(int Num, int N, float **ds, float **input, int **max_IDs, float delta);


void func_ds_eq_ds_input(int Num, int N, float **ds, float **input, int **max_IDs, float delta, float* m_max_counter, int nowait);
__global__
void func_ds_eq_ds_input_kernel(int Num, int N, float **ds, float **input, int **max_IDs, float delta, float* m_max_counter);

__global__
void func_ds_eq_ds_input_kernel_sparse(int Num, int N, float **ds, int W, int **input, int **max_IDs, float delta);

__global__
void func_ds_eq_ds_input_kernel_sparse(int Num, int N, float **ds, int W, int **input, int **max_IDs, float delta, float Lmid, float Lbase);



__global__
void func_ds_eq_ds_input_kernel_sparse(int Num, int N, float **ds, int W, int **input, int **max_IDs, float delta, float *inv_frequency_scaling, int vocabulary_size);

__global__
void func_ds_eq_ds_input_kernel_sparse(int Num, int N, float **ds, int W, int **input, int **max_IDs, float delta, float Lmid, float Lbase, float *inv_frequency_scaling, int vocabulary_size);



void fill_func(int N, float *data, float value, int nowait);

__global__
void fill_func_kernel(int N, float *data, float value);

void xx_atomic_update(int Num, int **max_IDs, float *xx, float **tot_input, float delta, int nowait);

__global__
void xx_atomic_update_kernel(int Num, int **max_IDs, float *xx, float **tot_input, float delta);


void xx_atomic_update(int Num, int **max_IDs, float *xx, float **tot_input, float delta, float* m_max_counter, int nowait);

__global__
void xx_atomic_update_kernel(int Num, int **max_IDs, float *xx, float **tot_input,  float* m_max_counter,  float delta);



void copy_from_M(int input_start, int input_end, uint64_t *myvector, int N, float **M, float **input, int nowait);

__global__
void copy_from_M_kernel(int input_start, int input_end, uint64_t *myvector, int N, float **M, float **input);


void inflate_from_M(int Num, int W, int L, int ST, int Nchannels, IMTYPE **M1, uint64_t L_W_ST_block, uint64_t *myvector,  float **input, int nowait);
__global__
void inflate_from_M_kernel(int Num, int W, int L, int ST, int Nchannels,  IMTYPE **M1, uint64_t L_W_ST_block, uint64_t *myvector,  float **input);


void inflate_from_M(int Num, int W, uint64_t IM_WIDTH, int ST, int Nchannels, IMTYPE **M1, uint64_t HEIGHT_W_ST_block, uint64_t WIDTH_W_ST_block, uint64_t *myvector,  float **input, int nowait);
__global__
void inflate_from_M_kernel(int Num, int W, uint64_t IM_WIDTH, int ST, int Nchannels,  IMTYPE **M1, uint64_t HEIGHT_W_ST_block, uint64_t WIDTH_W_ST_block, uint64_t *myvector,  float **input);

template <class IM_Ptr>
__global__
void inflate_from_M_kernel_2(uint64_t Num, int W, uint64_t IM_WIDTH, int ST, int Nchannels, IM_Ptr **M1, uint64_t HEIGHT_W_ST_block, uint64_t WIDTH_W_ST_block, uint64_t *rand_image,  float **input);


template <class IM_Ptr>
__global__
void inflate_from_M_kernel_4(uint64_t Num, int W, uint64_t IM_WIDTH, uint64_t  vocabulary_size, IM_Ptr **M1, uint64_t *rand_image, int **input);

template <class IM_Ptr>
__global__
void inflate_from_M_kernel_4_and_prune(uint64_t Num, int W, uint64_t IM_WIDTH, uint64_t  vocabulary_size, IM_Ptr **M1, uint64_t *rand_image, int **input);

void  max_of_AbsVal(int N, float *data, float *result);
__global__
void max_of_AbsVal_kernel(int N, float *data, float *max_per_block);
__global__
void max_of_AbsVal_kernel_2(int N, float *data, float *max_per_block);
__global__
void max_of_AbsVal_kernel_finish(int N, float *data, float *result);



__global__
void dense_sparse_matmul_special(uint64_t Num, uint64_t hid, uint64_t N, int W, float **ds, int** input, float **tot_input);

__global__
void dense_sparse_matmul_special(uint64_t Num, uint64_t hid, uint64_t N, int W, float **ds, int** input, float **tot_input, float *inv_frequency_scaling, int vocabulary_size);

__global__
void dense_sparse_matmul_special(uint64_t Num, uint64_t hid, uint64_t N, int W, float **ds, int** input, float **tot_input, float Lmid, float Lbase);

__global__
void dense_sparse_matmul_special(uint64_t Num, uint64_t hid, uint64_t N, int W, float **ds, int** input, float **tot_input, float Lmid, float Lbase, float *inv_frequency_scaling, int vocabulary_size);


template<class IM_Ptr, class INPUT_Ptr, unsigned char input_scalling_factor>
void run_epoch(uint64_t Ns, int Num, uint64_t N, int hid, int p, int m, int W, int L, int ST, uint64_t L_W_ST_block, float delta, float prec, float eps,
                int Nchannels, uint64_t *myvector, float **input, int **input_sparse_indx, float **synapses,float **ds,float **tot_input,float *xx, int **max_IDs, IM_Ptr **M1, int sparse_input);

template<class IM_Ptr, class INPUT_Ptr, unsigned char input_scalling_factor>
void run_epoch(uint64_t Ns, int Num, uint64_t N, int hid, int p, int m, int W, uint64_t IM_WIDTH, int ST, uint64_t HEIGHT_W_ST_block, uint64_t WIDTH_W_ST_block, float delta, float prec, float eps, int Nchannels, uint64_t *myvector, float **input, int **input_sparse_indx, float **synapses,float **ds,float **tot_input,float *xx, int **max_IDs, IM_Ptr **M1, int sparse_input);


template<class IM_Ptr, class INPUT_Ptr, unsigned char input_scalling_factor>
void run_epoch(model_descriptor * DSCR,  model_arrays * MA, int ngpus, int *list_gpus, float delta,  float eps);


void cuda_func1(int ii_start, int ii_end, float * __restrict__ synapses, float * __restrict__ ds, int p, int nowait){

    int p_1 = p-1;

    size_t nthreads = 256;
    size_t nblocks = (ii_end-ii_start + nthreads - 1)/nthreads;
    if (nblocks > 64*1024) nblocks = 64*1024;

    if (p_1%2 == 0)
     cuda_func1_even<<<nblocks,nthreads>>>(ii_end-ii_start, &synapses[ii_start], &ds[ii_start],p_1);
    else
     cuda_func1_odd<<<nblocks,nthreads>>>(ii_end-ii_start, &synapses[ii_start], &ds[ii_start],p_1);

    if (nowait == 0) cudaDeviceSynchronize();

}

__global__
void cuda_func1_even(int N, float * __restrict__ synapses, float * __restrict__ ds, int p_1){
   int start_index = blockIdx.x * blockDim.x + threadIdx.x;
   int offset = blockDim.x*gridDim.x;

   for (int ii = start_index; ii < N; ii+=offset){
       double snps = synapses[ii];
       double sig = (snps < 0) ? -1.0 : 1.0;
       double pwr=1.0;
       for (int pp=0; pp < p_1; ++pp) pwr *= snps;
       ds[ii] = (float) (sig*pwr);
     }
}

__global__
void cuda_func1_odd(int N, float * __restrict__ synapses, float * __restrict__ ds, int p_1){
   int start_index = blockIdx.x * blockDim.x + threadIdx.x;
   int offset = blockDim.x*gridDim.x;

   for (int ii = start_index; ii < N; ii+=offset){
     double snps = synapses[ii];
     double pwr=1.0;
     for (int pp=0; pp < p_1; ++pp) pwr *= snps;
     ds[ii] = (float) pwr;
   }
}


__global__
void dense_sparse_matmul_special(uint64_t Num, uint64_t hid, uint64_t N, int W, float **ds, int** input, float **tot_input){

    for (uint64_t row = blockIdx.x;  row < Num; row += gridDim.x){
      for (uint64_t col = threadIdx.x; col < hid; col += blockDim.x){
        double sum = 0.0; 
        for (uint64_t k = 0; k < W; ++k){
            int indx = input[row][k];
            if (indx != -1){ 
              sum +=  __ldg(&ds[col][indx]);
            }
        }     
        tot_input[row][col] = sum;
      } 
    }   
}     


__global__
void dense_sparse_matmul_special(uint64_t Num, uint64_t hid, uint64_t N, int W, float **ds, int** input, float **tot_input, float Lmid, float Lbase){

    const int mid_point = W/2;
    for (uint64_t row = blockIdx.x;  row < Num; row += gridDim.x){
      for (uint64_t col = threadIdx.x; col < hid; col += blockDim.x){
        double sum = 0.0;
        for (int k = 0; k < W; ++k){
            int indx = input[row][k];
            if (indx != -1){
              //float scale_factor = word_inv_freq[indx%vocabulary_size]; 
              if (mid_point == k)
                 sum +=  __ldg(&ds[col][indx])*Lmid;//*scale_factor;
              else
                 sum +=  __ldg(&ds[col][indx])*Lbase;//*scale_factor;
            }
        }
        tot_input[row][col] = sum;
      }
    }
}


__global__
void dense_sparse_matmul_special(uint64_t Num, uint64_t hid, uint64_t N, int W, float **ds, int** input, float **tot_input, float *inv_frequency_scaling, int vocabulary_size){

    for (uint64_t row = blockIdx.x;  row < Num; row += gridDim.x){
      for (uint64_t col = threadIdx.x; col < hid; col += blockDim.x){
        double sum = 0.0;
        for (uint64_t k = 0; k < W; ++k){
            int indx = input[row][k];
            if (indx != -1){
              float scale_factor = inv_frequency_scaling[indx%vocabulary_size];
              sum +=  __ldg(&ds[col][indx])*scale_factor;
            }
        }
        tot_input[row][col] = sum;
      }
    }
}

__global__
void dense_sparse_matmul_special(uint64_t Num, uint64_t hid, uint64_t N, int W, float **ds, int** input, float **tot_input, float Lmid, float Lbase, float *inv_frequency_scaling, int vocabulary_size){

    const int mid_point = W/2;
    for (uint64_t row = blockIdx.x;  row < Num; row += gridDim.x){
      for (uint64_t col = threadIdx.x; col < hid; col += blockDim.x){
        double sum = 0.0;
        for (int k = 0; k < W; ++k){
            int indx = input[row][k];
            if (indx != -1){
              float scale_factor = inv_frequency_scaling[indx%vocabulary_size]; 
              if (mid_point == k)
                 sum +=  __ldg(&ds[col][indx])*Lmid*scale_factor;
              else
                 sum +=  __ldg(&ds[col][indx])*Lbase*scale_factor;
            }
        }
        tot_input[row][col] = sum;
      }
    }
}



void cuda_daxpy(int N, float * __restrict__ y, float * __restrict__ x, float val){
    int nthreads = 256;
    int nblocks = (N + nthreads - 1) / nthreads;
    cuda_daxpy_kernel<<<nblocks,nthreads>>>(N,y,x,val);

    cudaDeviceSynchronize();


}


__global__
void cuda_daxpy_kernel(int N, float * __restrict__ y, float * __restrict__ x, float val){
   int start_index = blockIdx.x * blockDim.x + threadIdx.x;
   int offset = blockDim.x*gridDim.x;
   for (int i = start_index; i < N; i+=offset)
      y[i] += val * x[i];
}
#if 0
__global__
void update_synapses(int N, float * __restrict__ synapses, float * __restrict__ ds, 
                     float * const prec, float * const nc, float * const eps){

   float eps_inv_nc;
   if (prec[0] > nc[0]) eps_inv_nc = eps[0]/prec[0];
   else                 eps_inv_nc = eps[0]/nc[0];

   int start_index = blockIdx.x * blockDim.x + threadIdx.x;
   int offset = blockDim.x*gridDim.x;
   for (int i = start_index; i < N; i+=offset)
      synapses[i] += eps_inv_nc * ds[i];

  //if (start_index==0)
  // printf("eps_inv_nc=%e, eps=%e, prec=%e, nc = %e\n",eps_inv_nc,eps[0],prec[0],nc[0]);
}
#else
__global__
void update_synapses(int N, float * __restrict__ synapses, float * __restrict__ ds,
                     float * const nc){

   float eps_inv_nc;
   if (eps_prec[1] > nc[0]) eps_inv_nc = eps_prec[0]/eps_prec[1];
   else                     eps_inv_nc = eps_prec[0]/nc[0];

   int start_index = blockIdx.x * blockDim.x + threadIdx.x;
   int offset = blockDim.x*gridDim.x;
   for (int i = start_index; i < N; i+=offset)
      synapses[i] += eps_inv_nc * ds[i];
}
#endif


// ds[h][c] =  -xx[h]*synapses[h][c];

void cuda_ds_xx_t_synapses(int N, int hid, float ** __restrict__ ds, float * __restrict__ xx, float ** __restrict__ synapses, int nowait){

   int nthreads = N;
   if (nthreads > 512) nthreads = 512;
   int nblocks = hid;
   cuda_ds_xx_t_synapses_kernel<<<nblocks,nthreads>>>(N, hid, ds, xx, synapses);
   if (nowait==0) cudaDeviceSynchronize();
}

__global__
void cuda_ds_xx_t_synapses_kernel(int N, int hid, float ** __restrict__ ds, float * __restrict__ xx, float **  __restrict__ synapses){

   int h = blockIdx.x;
   if (xx[h] != 0){ 
     for (int c = threadIdx.x; c < N; c += blockDim.x)
       ds[h][c] =  -xx[h]*synapses[h][c];
   }
   else{
     for (int c = threadIdx.x; c < N; c += blockDim.x)
       ds[h][c] =  0.0;
   }
}


void get_max_and_m_max(  int Num, int hid, float **tot_input, int num_m,  int **ind_max_m_max, int nowait){

      int nthreads = 64;
      int nblocks = Num;
      int s_mem_size = nthreads*(sizeof(int)+sizeof(float)) + num_m*(sizeof(int)+sizeof(float)); 
      #if 0
      get_max_and_m_max_kernel<<<nblocks,nthreads,s_mem_size>>>(Num,hid,tot_input,num_m,ind_max_m_max);
      #else
      get_max_and_m_max_kernel_2<<<nblocks,nthreads,s_mem_size>>>(Num,hid,tot_input,num_m,ind_max_m_max);
      #endif
      if (nowait==0) cudaDeviceSynchronize();


}

__global__ 
void get_max_and_m_max_kernel(  int Num, int hid, float **tot_input, int num_m,  int **ind_max_m_max ){

  extern __shared__ char s[];
  int *idmax, *ind_of_max_m;
  float *val_max, *val_of_max_m;

  val_max = (float*) s;
  val_of_max_m = (float*) ( s+blockDim.x*sizeof(float) );
  idmax = (int*) (s + blockDim.x*sizeof(float) + num_m*sizeof(float)  );
  ind_of_max_m = (int*) (s + blockDim.x*sizeof(float) + num_m*sizeof(float)  + blockDim.x*sizeof(int) );

  for (int blk = blockIdx.x;  blk < Num; blk += gridDim.x){
    float *data = tot_input[blk];

    for (int m = 0; m < num_m; ++m){
      float ref_max_value = data[0];
      int  ref_index = 0;
      for (int i = threadIdx.x; i < hid; i += blockDim.x){

        if ( data[i] > ref_max_value){
          ref_max_value = data[i];
          ref_index = i;
        }
      }
      val_max[threadIdx.x] = ref_max_value;
      idmax[threadIdx.x] = ref_index;

      __syncthreads();

      if (threadIdx.x == 0){

        ref_max_value = val_max[0];
        ref_index = idmax[0];
        for (int i = 1; i < blockDim.x; ++i){
          if (val_max[i] > ref_max_value){
            ref_max_value = val_max[i];
            ref_index = idmax[i];
          }
        }
        ind_of_max_m[m] = ref_index;
        val_of_max_m[m] = ref_max_value;
        data[ref_index] = -9999.9999 ;
      }
      __syncthreads();
     }

     //output
     if (threadIdx.x == 0){
       ind_max_m_max[blk][0] = ind_of_max_m[0];
       ind_max_m_max[blk][1] = ind_of_max_m[num_m-1];
     }
     __syncthreads();

     //restore data
     for (int m = threadIdx.x; m < num_m; m+=blockDim.x)
       data[ ind_of_max_m[m] ] = val_of_max_m[m];

     __syncthreads();

   }
}


__global__
void get_max_and_m_max_kernel_2(  int Num, int hid, float **tot_input, int num_m,  int **ind_max_m_max ){

  extern __shared__ char s[];
  int *idmax, *ind_of_max_m;
  float *val_max, *val_of_max_m;

  val_max = (float*) s;
  val_of_max_m = (float*) ( s+blockDim.x*sizeof(float) );
  idmax = (int*) (s + blockDim.x*sizeof(float) + num_m*sizeof(float)  );
  ind_of_max_m = (int*) (s + blockDim.x*sizeof(float) + num_m*sizeof(float)  + blockDim.x*sizeof(int) );

  for (int blk = blockIdx.x;  blk < Num; blk += gridDim.x){
    float *data = tot_input[blk];

    for (int m = 0; m < num_m; ++m){
      float ref_max_value = data[0];
      int  ref_index = 0;

      for (int i = threadIdx.x; i < hid; i += blockDim.x){
        if ( data[i] > ref_max_value){
          ref_max_value = data[i];
          ref_index = i;
        }
      }

      //reduction within each warp 
      for (int i=16; i>=1; i/=2){
        float val = __shfl_down_sync(0xffffffff, ref_max_value, i, 32);
        int   ind = __shfl_down_sync(0xffffffff, ref_index, i, 32);
        if (val > ref_max_value){
          ref_max_value = val;
          ref_index = ind;
        }
      }

      //thread threadIdx.x%32 == 0 stores data in shared memory
      if (threadIdx.x%32 == 0){
        val_max[threadIdx.x/32] = ref_max_value;
        idmax[threadIdx.x/32] = ref_index;
      }

      __syncthreads();

      if (threadIdx.x == 0){

        ref_max_value = val_max[0];
        ref_index = idmax[0];
        for (int i = 1; i < blockDim.x/32; ++i){
          if (val_max[i] > ref_max_value){
            ref_max_value = val_max[i];
            ref_index = idmax[i];
          }
        }
        ind_of_max_m[m] = ref_index;
        val_of_max_m[m] = ref_max_value;
        data[ref_index] = -9999.9999 ;
        
    
        ind_max_m_max[blk][0] = ind_of_max_m[0];
        ind_max_m_max[blk][1] = ind_of_max_m[num_m-1];

      }
      __syncthreads();

    }
     //restore data
     for (int m = threadIdx.x; m < num_m; m+=blockDim.x)
       data[ ind_of_max_m[m] ] = val_of_max_m[m];

     __syncthreads();

   }
}



void func_ds_eq_ds_input(int Num, int N, float **ds, float **input, int **max_IDs, float delta, int nowait){

      int nthreads = 128;
      if (N < 33) nthreads=32;
      else if (N < 65) nthreads = 64;
      else if (N < 97) nthreads = 96;

      int nblocks = Num;
      func_ds_eq_ds_input_kernel<<<nblocks,nthreads>>>(Num,N,ds,input,max_IDs,delta);
      if (nowait == 0) cudaDeviceSynchronize();

}

__global__
void func_ds_eq_ds_input_kernel(int Num, int N, float **ds, float **input, int **max_IDs, float delta){

      for (int c = blockIdx.x;  c < Num; c += gridDim.x){
         int index_max  = max_IDs[c][0];
         int index_m_max  = max_IDs[c][1];
         for (int c_input = threadIdx.x; c_input < N; c_input+=blockDim.x){
           atomicAdd(&ds[index_max][c_input], input[c][c_input]);
           atomicAdd(&ds[index_m_max][c_input],(-delta*input[c][c_input]));
         }
       }
}

void func_ds_eq_ds_input(int Num, int N, float **ds, float **input, int **max_IDs, float delta, float *m_max_counter, int nowait){

      int nthreads = 128;
      if (N < 33) nthreads=32;
      else if (N < 65) nthreads = 64;
      else if (N < 97) nthreads = 96;

      int nblocks = Num;
      func_ds_eq_ds_input_kernel<<<nblocks,nthreads>>>(Num,N,ds,input,max_IDs,delta,m_max_counter);
      if (nowait == 0) cudaDeviceSynchronize();

}

__global__
void func_ds_eq_ds_input_kernel(int Num, int N, float **ds, float **input, int **max_IDs, float delta, float *m_max_counter){
      for (int c = blockIdx.x;  c < Num; c += gridDim.x){
         const int index_max  = max_IDs[c][0];
         const int index_m_max  = max_IDs[c][1];
         const float delta_scale = delta/m_max_counter[index_m_max];
         for (int c_input = threadIdx.x; c_input < N; c_input+=blockDim.x){
           atomicAdd(&ds[index_max][c_input], input[c][c_input]);
           atomicAdd(&ds[index_m_max][c_input],(-delta_scale*input[c][c_input]));
         }
       }
}


__global__
void func_ds_eq_ds_input_kernel_sparse(int Num, int N, float **ds, int W, int **input, int **max_IDs, float delta){

      for (int c = blockIdx.x;  c < Num; c += gridDim.x){
         int index_max  = max_IDs[c][0];
         int index_m_max  = max_IDs[c][1];
         for (int c_input = threadIdx.x; c_input < W; c_input+=blockDim.x){
           int idx = input[c][c_input];
           if (idx != -1){
             atomicAdd(&ds[index_max][idx], 1.0);
             if (delta != 0) atomicAdd(&ds[index_m_max][idx],(-delta));
           }
         }
       }
}

__global__
void func_ds_eq_ds_input_kernel_sparse(int Num, int N, float **ds, int W, int **input, int **max_IDs, float delta, float Lmid, float Lbase){

      const int mid_point = W/2;
      for (int c = blockIdx.x;  c < Num; c += gridDim.x){
         int index_max  = max_IDs[c][0];
         int index_m_max  = max_IDs[c][1];
         for (int c_input = threadIdx.x; c_input < W; c_input+=blockDim.x){
           int idx = input[c][c_input];
           if (idx != -1){
             if (mid_point == c_input){
               atomicAdd(&ds[index_max][idx], 1.0*Lmid);
               if (delta != 0) atomicAdd(&ds[index_m_max][idx],(-delta*Lmid));
             }
             else{
               atomicAdd(&ds[index_max][idx], 1.0*Lbase);
               if (delta != 0) atomicAdd(&ds[index_m_max][idx],(-delta*Lbase));
             }
           }
         }
       }
}

__global__
void func_ds_eq_ds_input_kernel_sparse(int Num, int N, float **ds, int W, int **input, int **max_IDs, float delta, float *inv_frequency_scaling, int vocabulary_size){

      for (int c = blockIdx.x;  c < Num; c += gridDim.x){
         int index_max  = max_IDs[c][0];
         int index_m_max  = max_IDs[c][1];
         for (int c_input = threadIdx.x; c_input < W; c_input+=blockDim.x){
           int idx = input[c][c_input];
           if (idx != -1){
             float scale_factor = inv_frequency_scaling[idx%vocabulary_size];
             atomicAdd(&ds[index_max][idx], 1.0*scale_factor);
             if (delta != 0) atomicAdd(&ds[index_m_max][idx],(-delta*scale_factor));
           }
         }
       }
}


__global__
void func_ds_eq_ds_input_kernel_sparse(int Num, int N, float **ds, int W, int **input, int **max_IDs, float delta, float Lmid, float Lbase, float *inv_frequency_scaling, int vocabulary_size){

      const int mid_point = W/2;
      for (int c = blockIdx.x;  c < Num; c += gridDim.x){
         int index_max  = max_IDs[c][0];
         int index_m_max  = max_IDs[c][1];
         for (int c_input = threadIdx.x; c_input < W; c_input+=blockDim.x){
           int idx = input[c][c_input];
           if (idx != -1){
             float scale_factor = inv_frequency_scaling[idx%vocabulary_size];
             if (mid_point == c_input){
               atomicAdd(&ds[index_max][idx], 1.0*Lmid*scale_factor);
               if (delta != 0) atomicAdd(&ds[index_m_max][idx],(-delta*Lmid*scale_factor));
             }
             else{
               atomicAdd(&ds[index_max][idx], 1.0*Lbase*scale_factor);
               if (delta != 0) atomicAdd(&ds[index_m_max][idx],(-delta*Lbase*scale_factor));
             }
           }
         }
       }
}



void fill_func(int N, float *data, float value, int nowait){
    int nthreads = 256;
    int nblocks = (N+nthreads-1)/nthreads;
    fill_func_kernel<<<nblocks,nthreads>>>(N,data,value);
    if (nowait==0) cudaDeviceSynchronize();
}

__global__
void fill_func_kernel(int N, float *data, float value){  
   int gtid = threadIdx.x + blockDim.x*blockIdx.x;
   int offset = blockDim.x*gridDim.x;
   for (int i = gtid; i < N; i+=offset)
      data[i] = value;
}  

void xx_atomic_update(int Num, int **max_IDs, float *xx, float **tot_input, float delta, int nowait){
    int nthreads = 1;
    int nblocks = Num;
    xx_atomic_update_kernel<<<nblocks,nthreads>>>(Num,max_IDs,xx,tot_input,delta);
    if (nowait==0) cudaDeviceSynchronize();
}
__global__
void xx_atomic_update_kernel(int Num, int **max_IDs, float *xx, float **tot_input, float delta){


   if (delta != 0.0){
     if (threadIdx.x==0){
       for (int c = blockIdx.x; c < Num; c+=gridDim.x ){
         int index_max  = max_IDs[c][0];
         int index_m_max  = max_IDs[c][1];
         atomicAdd( &xx[index_max],tot_input[c][index_max] );
         atomicAdd( &xx[index_m_max], (- delta*tot_input[c][index_m_max]) );
       }
     }
   }
   else{
     if (threadIdx.x==0){
       for (int c = blockIdx.x; c < Num; c+=gridDim.x ){
         int index_max  = max_IDs[c][0];
         atomicAdd( &xx[index_max],tot_input[c][index_max] );
       }
     }
   }
}

void xx_atomic_update(int Num, int **max_IDs, float *xx, float **tot_input, float delta, float *m_max_counter, int nowait){
    int nthreads = 1;
    int nblocks = Num;
    xx_atomic_update_kernel<<<nblocks,nthreads>>>(Num,max_IDs,xx,tot_input,m_max_counter,delta);
    if (nowait==0) cudaDeviceSynchronize();
}
__global__
void xx_atomic_update_kernel(int Num, int **max_IDs, float *xx, float **tot_input, float *m_max_counter, float delta){


     if (threadIdx.x==0){
       for (int c = blockIdx.x; c < Num; c+=gridDim.x ){
         int index_max  = max_IDs[c][0];
         int index_m_max  = max_IDs[c][1];
         float m_max_c = m_max_counter[index_m_max];
         if (m_max_c < 1) m_max_c = 1.0;
         atomicAdd( &xx[index_max],tot_input[c][index_max] );
         atomicAdd( &xx[index_m_max], (- delta/m_max_c*tot_input[c][index_m_max]) );
       }
     }
}






void copy_from_M(int input_start, int input_end, uint64_t *myvector, int N, float **M, float **input, int nowait){
    
   int nthreads;
   nthreads = ( (N+31)/32 ) * 32;
   if (nthreads > 256) nthreads = 256;
 
   int nblocks = input_end - input_start;
   copy_from_M_kernel<<<nblocks,nthreads>>>(input_start, input_end, myvector, N, M, input);
   if (nowait==0) cudaDeviceSynchronize();

}

__global__
void copy_from_M_kernel(int input_start, int input_end, uint64_t *myvector, int N, float **M, float **input){      

     for (int kk = input_start+blockIdx.x; kk < input_end; kk+=gridDim.x){
       float *dest = input[kk-input_start];
       float *src  = M[myvector[kk]];
       for (int i = threadIdx.x; i < N; i+=blockDim.x)
         dest[i] = src[i];
     }
}

void inflate_from_M(int Num, int W, int L, int ST, int Nchannels, IMTYPE **M1, uint64_t L_W_ST_block, uint64_t *myvector,  float **input, int nowait){

   unsigned int W_Nchannels = W*Nchannels;
   unsigned int d1;
   unsigned int d2 = 4;

   if (W_Nchannels < 16) d1= 16;
   else if (W_Nchannels < 32) d1 = 32;
   else if (W_Nchannels < 128) d1 = 64;
   else d1 = 128;

   dim3 nthreads = {d1,d2,1};

   int nblocks = Num;
   inflate_from_M_kernel<<<nblocks,nthreads>>>(Num,W,L,ST, Nchannels,M1,L_W_ST_block,myvector,input);
    if (nowait==0) cudaDeviceSynchronize();
}

__global__
void inflate_from_M_kernel(int Num, int W, int L, int ST, int Nchannels, IMTYPE **M1, uint64_t L_W_ST_block, uint64_t *myvector,  float **input){

    #ifdef IMAGE_AS_CHAR
    float scal = 1.0/255.0;
    #else
    float scal = 1.0; 
    #endif

     for (uint64_t k = blockIdx.x; k < Num; k+=gridDim.x){
       const uint64_t index = myvector[k];
       uint64_t r = index/(L_W_ST_block*L_W_ST_block);
       uint64_t shift = index%(L_W_ST_block*L_W_ST_block);
       uint64_t r_mm = (shift / L_W_ST_block)*ST;
       uint64_t c_mm = (shift % L_W_ST_block)*ST;


       for (uint64_t rr_M = threadIdx.y; rr_M < W; rr_M+=blockDim.y){
          IMTYPE *MM = M1[r] + (r_mm+rr_M)*L*Nchannels;

         for (uint64_t cc_M = threadIdx.x; cc_M < W*Nchannels; cc_M+=blockDim.x){
          input[k][cc_M + rr_M*W*Nchannels] = scal*MM[c_mm*Nchannels+cc_M];
         }
       }
    }
}

void inflate_from_M(int Num, int W, uint64_t IM_WIDTH, int ST, int Nchannels, IMTYPE **M1, uint64_t HEIGHT_W_ST_block, uint64_t WIDTH_W_ST_block, uint64_t *myvector,  float **input, int nowait){

   unsigned int W_Nchannels = W*Nchannels;
   unsigned int d1;
   unsigned int d2 = 4;

   if (W_Nchannels < 16) d1= 16;
   else if (W_Nchannels < 32) d1 = 32;
   else if (W_Nchannels < 128) d1 = 64;
   else d1 = 128;

   dim3 nthreads = {d1,d2,1};

   int nblocks = Num;
   inflate_from_M_kernel<<<nblocks,nthreads>>>(Num,W,IM_WIDTH,ST, Nchannels,M1,HEIGHT_W_ST_block,WIDTH_W_ST_block,myvector,input);
    if (nowait==0) cudaDeviceSynchronize();
}

__global__
void inflate_from_M_kernel(int Num, int W, uint64_t IM_WIDTH, int ST, int Nchannels, IMTYPE **M1, uint64_t HEIGHT_W_ST_block, uint64_t WIDTH_W_ST_block, uint64_t *myvector,  float **input){

    #ifdef IMAGE_AS_CHAR
    float scal = 1.0/255.0;
    #else
    float scal = 1.0;
    #endif

     for (uint64_t k = blockIdx.x; k < Num; k+=gridDim.x){
       const uint64_t index = myvector[k];
       uint64_t r = index/(HEIGHT_W_ST_block*WIDTH_W_ST_block);
       uint64_t shift = index%(HEIGHT_W_ST_block*WIDTH_W_ST_block);
       uint64_t r_mm = (shift / WIDTH_W_ST_block )*ST;
       uint64_t c_mm = (shift % WIDTH_W_ST_block )*ST;


       for (uint64_t rr_M = threadIdx.y; rr_M < W; rr_M+=blockDim.y){
          IMTYPE *MM = M1[r] + (r_mm+rr_M)*IM_WIDTH*Nchannels;

         for (uint64_t cc_M = threadIdx.x; cc_M < W*Nchannels; cc_M+=blockDim.x){
          input[k][cc_M + rr_M*W*Nchannels] = scal*MM[c_mm*Nchannels+cc_M];
         }
       }
    }
}


template <class IM_Ptr, unsigned char scalling_factor>
__global__
void inflate_from_M_kernel_2(uint64_t Num, uint64_t start_index,   int W, uint64_t IM_WIDTH, int ST, int Nchannels, IM_Ptr **M1, uint64_t HEIGHT_W_ST_block, uint64_t WIDTH_W_ST_block, uint64_t *rand_image,  float **input){

     float scal = 1.0/( (float) scalling_factor); 

     for (uint64_t k = blockIdx.x; k < Num; k+=gridDim.x){

       uint64_t im_id = rand_image[(k+start_index)/(HEIGHT_W_ST_block*WIDTH_W_ST_block)];
       uint64_t index = im_id*(HEIGHT_W_ST_block*WIDTH_W_ST_block);
       index = index + (k+start_index)%(HEIGHT_W_ST_block*WIDTH_W_ST_block);

       uint64_t r = index/(HEIGHT_W_ST_block*WIDTH_W_ST_block);
       uint64_t shift = index%(HEIGHT_W_ST_block*WIDTH_W_ST_block);
       uint64_t r_mm = (shift / WIDTH_W_ST_block )*ST;
       uint64_t c_mm = (shift % WIDTH_W_ST_block )*ST;


       for (uint64_t rr_M = threadIdx.y; rr_M < W; rr_M+=blockDim.y){
          IM_Ptr *MM = M1[r] + (r_mm+rr_M)*IM_WIDTH*Nchannels;

         for (uint64_t cc_M = threadIdx.x; cc_M < W*Nchannels; cc_M+=blockDim.x){
          input[k][cc_M + rr_M*W*Nchannels] = scal*MM[c_mm*Nchannels+cc_M];
         }
       }
    }
}

template <class IM_Ptr>
__global__
void inflate_from_M_kernel_4(uint64_t Num, int W, uint64_t IM_WIDTH, uint64_t vocabulary_size, IM_Ptr **M1, uint64_t *rand_image, int **input){

     int mid_point= W / 2;

     for (uint64_t k = blockIdx.x; k < Num; k+=gridDim.x){
       uint64_t text_id = rand_image[k];
       for (uint64_t c = threadIdx.x; c < W; c += blockDim.x){
           IM_Ptr val = M1[text_id][c];

           if (val == -1)
              input[k][c] = -1;
           else{
              if (c==mid_point)
                input[k][c] = val + vocabulary_size;
              else
                input[k][c] = val;
          }
       }
     }
}

template <class IM_Ptr>
__global__
void inflate_from_M_kernel_4_and_prune(uint64_t Num, int W, uint64_t IM_WIDTH, uint64_t vocabulary_size, IM_Ptr **M1, uint64_t *rand_image, int **input){

     extern __shared__ char s[];

     IM_Ptr *s_input = (IM_Ptr*) s;

     int mid_point= W / 2;

     for (uint64_t k = blockIdx.x; k < Num; k+=gridDim.x){
       uint64_t text_id = rand_image[k];

       for (int c = threadIdx.x; c < W; c += blockDim.x)
          s_input[c] = M1[text_id][c]; //move data to shared memory            
       
       //prune data
       for (int i = 1; i < W; ++i){
          IM_Ptr val = s_input[i-1];
          for (int c = i + threadIdx.x; c < W; c += blockDim.x){
             if (s_input[c] == val) s_input[c] = -1;
          }
          __syncthreads();
       }
       if (threadIdx.x == 0)
         s_input[mid_point] = M1[text_id][mid_point];
       __syncthreads();

       for (int c = threadIdx.x; c < W; c += blockDim.x){
           IM_Ptr val = s_input[c];

           if (val == -1)
              input[k][c] = -1;
           else{
              if (c==mid_point)
                input[k][c] = val + vocabulary_size;
              else
                input[k][c] = val;
          }
       }
     }
}





template <class IM_Ptr>
__global__
void inflate_from_M_kernel_5(uint64_t Num, int W, uint64_t IM_WIDTH, uint64_t vocabulary_size, IM_Ptr **M1, uint64_t *rand_image, int **input){

     int mid_point= W / 2;
     IM_Ptr * s_in; 

     for (uint64_t k = blockIdx.x; k < Num; k+=gridDim.x){
       uint64_t text_id = rand_image[k];
       for (uint64_t c = threadIdx.x; c < W; c += blockDim.x){
           IM_Ptr val = M1[text_id][c];

           if (val == -1)
              input[k][c] = -1;
           else{
              if (c==mid_point)
                input[k][c] = val + vocabulary_size;
              else
                input[k][c] = val;
          }
       }
     }
}


 
void  max_of_AbsVal(int N, float *data, float *result){

   static float *max_per_block=NULL;
   static int max_per_block_size=0;
   int nthreads = 512;
   int nblocks = (N+nthreads-1)/nthreads;


   if ( max_per_block_size < nblocks){
     if (max_per_block!=NULL) delete[] max_per_block;

     max_per_block = new float[nblocks];
     for (int i=0; i < nblocks; ++i) max_per_block[i] = 0.0;
     max_per_block_size = nblocks;
     cudaMemPrefetchAsync (max_per_block, max_per_block_size*sizeof(float), 0 );
     cudaDeviceSynchronize();  
   }

   #if 0
   int shared_mem_size = nthreads*sizeof(float);
   max_of_AbsVal_kernel<<<nblocks,nthreads,shared_mem_size>>>(N,data,&max_per_block[0]);
   #else
   int shared_mem_size = (nthreads+31)/32*sizeof(float);
   max_of_AbsVal_kernel_2<<<nblocks,nthreads,shared_mem_size>>>(N,data,&max_per_block[0]);
   #endif
   cudaDeviceSynchronize();
   float max_val = max_per_block[0];
   for (int i = 1; i < nblocks; ++i)
      max_val = max_val > max_per_block[i] ? max_val : max_per_block[i] ;

   result[0] = max_val;
}


__global__
void max_of_AbsVal_kernel(int N, float *data, float *max_per_block){

  extern __shared__ char s[];
  float *max_per_thread = (float*) s;
  
  max_per_thread[threadIdx.x] = -1.0; 
  
  int i_start = threadIdx.x + blockDim.x*blockIdx.x;
  int shift = blockDim.x*gridDim.x;
  float my_max = -1.0; 

  for (int i = i_start; i < N; i += shift){
     float val = fabs(data[i]);
     my_max = my_max > val ? my_max : val; 
     //max_per_thread[threadIdx.x] = max_per_thread[threadIdx.x] > val ? max_per_thread[threadIdx.x] : val ;
  }
  max_per_thread[threadIdx.x] = my_max;
  __syncthreads();
  //loop over warps 
  int nwarps = blockDim.x / 32;
  int tlimit = blockDim.x/2;
  for (int n = 0; n < nwarps/2; ++n){
    for (int i = threadIdx.x; i < tlimit; ++i){
      float val = max_per_thread[i+tlimit];
      max_per_thread[threadIdx.x] = max_per_thread[threadIdx.x] > val ? max_per_thread[threadIdx.x] : val ;
    }
    __syncthreads();
    tlimit = tlimit/2;
  }
  //loop over one warp 
  if (threadIdx.x == 0){ 
    for (int i = threadIdx.x+1; i < 32; ++i){
      if (max_per_thread[0] < max_per_thread[i]) max_per_thread[0] = max_per_thread[i];
    }
    max_per_block[blockIdx.x] = max_per_thread[0];  
  }
}

__global__
void max_of_AbsVal_kernel_2(int N, float *data, float *max_per_block){

  extern __shared__ char s[];
  float *max_per_thread = (float*) s;

  int i_start = threadIdx.x + blockDim.x*blockIdx.x;
  int shift = blockDim.x*gridDim.x;

  float my_max = -1.0;
  for (int i = i_start; i < N; i += shift){
     float val = fabs(data[i]);
     my_max = my_max > val ? my_max : val;
  }

  //reduction within each warp
  for (int i=16; i>=1; i/=2){
     float val  = __shfl_down_sync(0xffffffff, my_max, i, 32); 
     my_max = my_max > val ? my_max : val;
  }

  //thread threadIdx.x%32 == 0 has the max over a warp of threads 
  if (threadIdx.x%32 == 0)
    max_per_thread[threadIdx.x/32] = my_max;

  __syncthreads();

  //loop over warps
  if (threadIdx.x == 0){
    for (int n = 1; n < blockDim.x / 32; ++n){
       float val = max_per_thread[n];
       my_max = my_max > val ? my_max : val;
    }
    max_per_block[blockIdx.x] = my_max;
  }
}


__global__
void max_of_AbsVal_kernel_finish(int N, float *data, float *result){

  extern __shared__ char s[];
  float *max_per_thread = (float*) s;

  int i_start = threadIdx.x + blockDim.x*blockIdx.x;
  int shift = blockDim.x*gridDim.x;

  float my_max = -1.0;
  for (int i = i_start; i < N; i += shift){
     float val = data[i];
     my_max = my_max > val ? my_max : val;
  }

  //reduction within each warp
  for (int i=16; i>=1; i/=2){
     float val  = __shfl_down_sync(0xffffffff, my_max, i, 32);
     my_max = my_max > val ? my_max : val;
  }

   __syncthreads();

  if (threadIdx.x%32 == 0)
    max_per_thread[threadIdx.x/32] = my_max;

  __syncthreads();

  if (threadIdx.x == 0){
    for (int n = 1; n < blockDim.x / 32; ++n){
       float val = max_per_thread[n];
       my_max = my_max > val ? my_max : val;
    }
    result[0] = my_max;
  }

}




extern "C"
void launch_epoch_INPUT_AS_IMAGE(void * descr_in, void *MA_in, int deviceID, uint64_t epoch_id, uint64_t Nep){

  model_descriptor * DSCR = (model_descriptor*) descr_in;
  model_arrays * MA = (model_arrays*) MA_in;
   
  unsigned char **M1 = MA[0].INPUT_DATA_uchar;

  cudaSetDevice(deviceID);
  uint64_t N = DSCR[0].W*DSCR[0].W*DSCR[0].Nchannels;
  float eps = DSCR[0].initial_learning_rate*(1.0- ( (float) epoch_id) / ((float) Nep));
  float delta_epoch = DSCR[0].delta*0.5*(1.0+tanh((  (float)  epoch_id-(0.125*Nep))/10.0));
//  MA[0].reshuffle_indices(DSCR[0],deviceID);

  run_epoch<unsigned char,float, 255>(DSCR[0].Ns, DSCR[0].Num, N, DSCR[0].hid, DSCR[0].p, DSCR[0].m, DSCR[0].W, DSCR[0].IM_WIDTH, DSCR[0].ST, DSCR[0].HEIGHT_W_ST_block, DSCR[0].WIDTH_W_ST_block, 
  delta_epoch, DSCR[0].prec, eps,
  DSCR[0].Nchannels, MA[0].indices, MA[0].input, MA[0].input_sparse_indx, MA[0].synapses, MA[0].ds, MA[0].tot_input, MA[0].xx, MA[0].max_IDs, M1, DSCR[0].sparse_input);
}


extern "C"
void launch_epoch_INPUT_AS_FLOAT(void * descr_in, void *MA_in, int deviceID,  uint64_t epoch_id, uint64_t Nep){

  cudaSetDevice(deviceID);
  model_descriptor * DSCR = (model_descriptor*) descr_in;
  model_arrays * MA = (model_arrays*) MA_in;

  float **M1 =  MA[0].INPUT_DATA_float;
  uint64_t N = DSCR[0].W*DSCR[0].W*DSCR[0].Nchannels;
  float eps = DSCR[0].initial_learning_rate*(1.0- ( (float) epoch_id) / ((float) Nep));
  float delta_epoch = DSCR[0].delta*0.5*(1.0+tanh((  (float)  epoch_id-(0.125*Nep))/10.0));

  if (epoch_id%2 != 0){
    #pragma omp parallel for
    for (size_t r = 0; r < DSCR[0].hid; ++r){
      double sum = 0.0;
      for (size_t col = 0; col < N; ++col)
        sum += pow(MA[0].synapses[r][col],DSCR[0].p);
      fprintf(stderr,"Lp_norm(syn[%lu]) = %g\n",r,pow(sum,1.0/DSCR[0].p));
    }
    fprintf(stderr,"#################################\n");
  }

//  if (num_gpus > 1){
//    int list_gpus[num_gpus];
//    for (int g=0; g < num_gpus; ++g ) list_gpus[g] = g;
//    run_epoch<int,int,1>( DSCR,  MA, num_gpus, &list_gpus[0], delta_epoch,  eps);
//  }
//  else{
    run_epoch<float,float, 1>(DSCR[0].Ns, DSCR[0].Num, N, DSCR[0].hid, DSCR[0].p, DSCR[0].m, DSCR[0].W, DSCR[0].IM_WIDTH, DSCR[0].ST, DSCR[0].HEIGHT_W_ST_block, DSCR[0].WIDTH_W_ST_block,
    delta_epoch, DSCR[0].prec, eps,
    DSCR[0].Nchannels, MA[0].indices, MA[0].input, MA[0].input_sparse_indx, MA[0].synapses, MA[0].ds, MA[0].tot_input, MA[0].xx, MA[0].max_IDs, M1, DSCR[0].sparse_input);
//  }
}


extern "C"
void launch_epoch_INPUT_AS_INT(void * descr_in, void *MA_in, int num_gpus,  uint64_t epoch_id, uint64_t Nep){
  
  
  model_descriptor * DSCR = (model_descriptor*) descr_in;
  model_arrays * MA = (model_arrays*) MA_in;
  
  int **M1 =  MA[0].INPUT_DATA_int;
 
  uint64_t N = DSCR[0].N;
  float eps = DSCR[0].initial_learning_rate*(1.0- ( (float) epoch_id) / ((float) Nep));
  float delta_epoch = DSCR[0].delta*0.5*(1.0+tanh((  (float)  epoch_id-(0.125*Nep))/10.0));

  cudaSetDevice(0);
  
  if (epoch_id%1 == 0){
    for (size_t r = 0; r < 10/*DSCR[0].hid*/; ++r){
      double sum = 0.0; 
      for (size_t col = 0; col < N; ++col)
        sum += pow(MA[0].synapses[r][col],DSCR[0].p);
      fprintf(stderr,"Lp_norm(syn[%lu]) = %g\n",r,pow(sum,1.0/DSCR[0].p));
    }
    fprintf(stderr,"#################################\n");
  }
  
//  if (num_gpus > 1){
    int list_gpus[num_gpus]; 
    for (int g=0; g < num_gpus; ++g ) list_gpus[g] = g;
    run_epoch<int,int,1>( DSCR,  MA, num_gpus, &list_gpus[0], delta_epoch,  eps);
/*  }
  else{
    cudaSetDevice(0); //use GPU 0
    run_epoch<int,int,1>(DSCR[0].Ns, DSCR[0].Num, N, DSCR[0].hid, DSCR[0].p, DSCR[0].m, DSCR[0].W, DSCR[0].IM_WIDTH, DSCR[0].ST, DSCR[0].HEIGHT_W_ST_block, DSCR[0].WIDTH_W_ST_block,
    delta_epoch, DSCR[0].prec, eps,
    DSCR[0].Nchannels, MA[0].indices, MA[0].input, MA[0].input_sparse_indx, MA[0].synapses, MA[0].ds, MA[0].tot_input, MA[0].xx, MA[0].max_IDs, M1, DSCR[0].sparse_input);
  }
*/

}





template<class IM_Ptr, class INPUT_Ptr, unsigned char input_scalling_factor>
void run_epoch(uint64_t Ns, int Num, uint64_t N, int hid, int p, int m, int W, uint64_t IM_WIDTH, int ST, uint64_t HEIGHT_W_ST_block,   uint64_t WIDTH_W_ST_block,  float delta, float prec, float eps, int Nchannels, uint64_t *myvector,float  **input, int **input_sparse_indx, float **synapses,float **ds,float **tot_input,float *xx, int **max_IDs, IM_Ptr **M1, int sparse_input){ 

    fprintf(stderr, "in run_epoch \n");
    fprintf(stderr, "N = %lu\n",N);

    gpuErrchk( cudaPeekAtLastError() );

    uint64_t vocabulary_size = 86083;

    cublasHandle_t handle;


    int CUBLAS_INIT_FLAG = 0;
    if (sparse_input != 0){
      if (CUBLAS_INIT_FLAG==0){
         fprintf(stderr,"calling  cublasCreate\n");
         cublasStatus_t stat;
         stat = cublasCreate (& handle );
         CUBLAS_INIT_FLAG=1;
      }
    }

    int cuGraph_FLAG=0;
    cudaStream_t stream1, stream_for_graph;
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    float *max_per_block=NULL;
    int nthreads_max_of_AbsVal;
    int nblocks_max_of_AbsVal;
    int shared_mem_size_max_of_AbsVal;

    int W_Nchannels = W*Nchannels;

    int LDA = (int) N;//ds
    int LDB = (int) N;//input
    int LDC = hid;//tot_input
    float alpha = 1.0;
    float beta = 0.0;
    int NN = (int) N;


    unsigned int d1;
    unsigned int d2 = 4;

    if (W_Nchannels < 16) d1= 16;
    else if (W_Nchannels < 32) d1 = 32;
    else if (W_Nchannels < 128) d1 = 64;
    else d1 = 128;

    int p_1 = p-1;
    int s_mem_size_get_max_and_m_max_kernel_2 = 64*(sizeof(int)+sizeof(float)) + m*(sizeof(int)+sizeof(float));

    int nthreads_func_ds_eq_ds_input_kernel = 128;
    if (N < 33) nthreads_func_ds_eq_ds_input_kernel=32;
    else if (N < 65) nthreads_func_ds_eq_ds_input_kernel = 64;
    else if (N < 97) nthreads_func_ds_eq_ds_input_kernel = 96;

    float host_eps_prec[2];
    host_eps_prec[0] = eps;
    host_eps_prec[1] = prec; 
    cudaMemcpyToSymbol    (eps_prec, host_eps_prec,  sizeof(float)*2);

    //if ( cuGraph_FLAG==0){

      CUCHECK( cudaStreamCreate(&stream1) );
      CUCHECK( cudaStreamCreate(&stream_for_graph) );
      if (sparse_input == 0)
        cublasSetStream(handle,stream1);

      nthreads_max_of_AbsVal = 512;
      nblocks_max_of_AbsVal = (hid*N+nthreads_max_of_AbsVal-1)/nthreads_max_of_AbsVal;
      
      //max_per_block = new float[nblocks_max_of_AbsVal];
      //shared_mem_size_max_of_AbsVal =  nthreads_max_of_AbsVal/32*sizeof(float);//  (nthreads_max_of_AbsVal+31)/32*sizeof(float);
      //for (int i=0; i < nblocks_max_of_AbsVal; ++i) max_per_block[i] = 0.0;
      //cudaMemPrefetchAsync (max_per_block, nblocks_max_of_AbsVal*sizeof(float), 0 );
      //cudaDeviceSynchronize();



      gpuErrchk( cudaPeekAtLastError() );

      cudaMallocManaged (&max_per_block,nblocks_max_of_AbsVal*sizeof(float));
      shared_mem_size_max_of_AbsVal =  nthreads_max_of_AbsVal/32*sizeof(float);//  (nthreads_max_of_AbsVal+31)/32*sizeof(float);
      for (int i=0; i < nblocks_max_of_AbsVal; ++i) max_per_block[i] = 0.0;
      size_t count = (size_t) nblocks_max_of_AbsVal * sizeof(float);
      cudaError_t cuERROR=  cudaMemPrefetchAsync (&max_per_block[0], count, 0 );
      cudaDeviceSynchronize();
      if (cuERROR == cudaErrorInvalidValue){
        fprintf(stderr,"cudaMemPrefetchAsync: cudaErrorInvalidValue pointer=%p, count=%llu  \n", &max_per_block[0], count  );
      }
      else if (cuERROR == cudaErrorInvalidDevice)  fprintf(stderr,"cudaMemPrefetchAsync: cudaErrorInvalidDevice\n");


      gpuErrchk( cudaPeekAtLastError() );


 
//change back to cudaStreamCaptureModeGlobal  in CUDA 10.2
 //     CUCHECK( cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal) );

      size_t cuda_func_nthreads = 256;
      size_t cuda_func_nblocks = (hid*N+cuda_func_nthreads-1)/cuda_func_nthreads;
      if (cuda_func_nblocks > 64*1024) cuda_func_nblocks = 64*1024;

      CUCHECK( cudaStreamBeginCapture(stream1, cudaStreamCaptureModeRelaxed) );

      if (p_1%2 == 0)
        cuda_func1_even<<<cuda_func_nblocks,cuda_func_nthreads,0,stream1>>>(hid*N, synapses[0], ds[0],p_1);
      else
        cuda_func1_odd<<<cuda_func_nblocks,cuda_func_nthreads,0,stream1>>>(hid*N, synapses[0], ds[0],p_1);


      if (sparse_input==0){
        cublasStatus_t stat;
        stat = cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N, hid,Num, NN, &alpha, ds[0], LDA, input[0],LDB, &beta,tot_input[0],LDC);
      }
      else 
        dense_sparse_matmul_special<<<Num,32,0,stream1>>>(Num,hid,N,W,ds,input_sparse_indx,tot_input);


      fill_func_kernel<<<cuda_func_nblocks,cuda_func_nthreads,0,stream1>>>(hid,xx,0.0);
      get_max_and_m_max_kernel_2<<<Num,64,s_mem_size_get_max_and_m_max_kernel_2,stream1>>>(Num,hid,tot_input,m,max_IDs);
      xx_atomic_update_kernel<<<Num,1,0,stream1>>>(Num,max_IDs,xx,tot_input,delta);
      cuda_ds_xx_t_synapses_kernel<<<hid,(N<512?N:512),0,stream1>>>(N, hid, ds, xx, synapses);


     if (sparse_input == 0)
       func_ds_eq_ds_input_kernel<<<Num,nthreads_func_ds_eq_ds_input_kernel,0,stream1>>>(Num,N,ds,input,max_IDs,delta);
     else
       func_ds_eq_ds_input_kernel_sparse<<<Num,nthreads_func_ds_eq_ds_input_kernel,0,stream1>>>(Num,N,ds,W,input_sparse_indx,max_IDs,delta);



      max_of_AbsVal_kernel_2<<<nblocks_max_of_AbsVal,nthreads_max_of_AbsVal,shared_mem_size_max_of_AbsVal,stream1>>>(hid*N,ds[0],&max_per_block[0]);
      max_of_AbsVal_kernel_finish<<<1,nthreads_max_of_AbsVal,shared_mem_size_max_of_AbsVal,stream1>>>(nblocks_max_of_AbsVal,max_per_block,max_per_block);
      update_synapses<<<cuda_func_nblocks,cuda_func_nthreads,0,stream1>>>(hid*N, synapses[0], ds[0], max_per_block);
      CUCHECK( cudaStreamEndCapture(stream1, &graph) );

      cudaGraphNode_t *nodes = NULL;
      size_t num_nodes = 0;
      CUCHECK( cudaGraphGetNodes(graph, nodes, &num_nodes) );
      printf("Num nodes in the created graph = %zu\n", num_nodes);

      CUCHECK( cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0) );
      if (sparse_input == 0)
        cublasSetStream(handle,stream_for_graph);

      //cuGraph_FLAG = 1;
    //}

    uint64_t  work_fraction  = Ns/40;    
    work_fraction = (work_fraction/Num)*Num;


    for (uint64_t k = 0; k < Ns; k += Num){

     uint64_t input_start = k;
     uint64_t input_end   = k + Num;
     if (input_end > Ns) input_end = Ns; 

     dim3 nthreads(d1,d2,1);
     dim3 nblocks(Num, 1, 1);

     gpuErrchk( cudaPeekAtLastError() );


     if (sparse_input == 0)
       inflate_from_M_kernel_2<IM_Ptr, input_scalling_factor><<<nblocks,nthreads,0,stream_for_graph>>>( input_end-input_start,  input_start,  W, IM_WIDTH,
          ST, Nchannels, M1, HEIGHT_W_ST_block, WIDTH_W_ST_block, myvector, input);
     else{
      inflate_from_M_kernel_4<IM_Ptr ><<<nblocks,32,0,stream_for_graph>>> (input_end-input_start, W, IM_WIDTH, vocabulary_size, M1, &myvector[input_start], input_sparse_indx);
      //inflate_from_M_kernel_4_and_prune<IM_Ptr ><<<nblocks,32,sizeof(IM_Ptr)*W,stream_for_graph>>> (input_end-input_start, W, IM_WIDTH, vocabulary_size, M1, &myvector[input_start], input_sparse_indx);
     } 
     CUCHECK( cudaGraphLaunch(graphExec, stream_for_graph) );

     if (k%work_fraction==0)  fprintf(stderr,"  run_epoch: %g %%  is done \n", (double) k/Ns*100.0);


     cudaStreamSynchronize(stream_for_graph);

    // for (int ii=0; ii<10; ++ii)
    //     fprintf(stderr,"max_IDs[%d][0 1] = [%llu %llu]\n",ii,max_IDs[ii][0], max_IDs[ii][1]);


   } //end of  for (int k = 0; k < Ns; k += Num)


  cudaFree(max_per_block);
  CUCHECK(cudaGraphExecDestroy(graphExec));
  CUCHECK(cudaGraphDestroy(graph));
  CUCHECK(cudaStreamDestroy(stream_for_graph));
  CUCHECK(cudaStreamDestroy(stream1));
  if (sparse_input == 0)
    cublasDestroy(handle);

}

template<class IM_Ptr, class INPUT_Ptr, unsigned char input_scalling_factor>
void run_epoch(model_descriptor * DSCR,  model_arrays * MA, int ngpus, int *list_gpus, float delta,  float eps){

  fprintf(stderr, "in run_epoch \n");
  fprintf(stderr, "ngpus = %d\n",ngpus);
  gpuErrchk( cudaPeekAtLastError() );

  uint64_t Ns = DSCR[0].Ns;
  int Num = DSCR[0].Num;
  uint64_t N = DSCR[0].N;
  int hid = DSCR[0].hid;
  int p = DSCR[0].p;
  int m = DSCR[0].m;
  int W = DSCR[0].W;
  uint64_t IM_WIDTH = DSCR[0].IM_WIDTH;
  int ST =  DSCR[0].ST;
  uint64_t HEIGHT_W_ST_block = DSCR[0].HEIGHT_W_ST_block;  
  uint64_t WIDTH_W_ST_block =  DSCR[0].WIDTH_W_ST_block;
  float prec = DSCR[0].prec;
  int Nchannels = DSCR[0].Nchannels;
  uint64_t *myvector =  MA[0].indices;
  uint64_t vocabulary_size = DSCR[0].vocabulary_size;
  int sparse_input = DSCR[0].sparse_input; 
  float Lmid = DSCR[0].Lmid;
  float Lbase = DSCR[0].Lbase; 
  int frequency_scaling = (int) DSCR[0].frequency_scaling;

  float **input[ngpus];
  int **input_sparse_indx[ngpus]; 
  float **synapses[ngpus]; 
  float **ds[ngpus];
  float **tot_input[ngpus];
  float **tot_input_raw[ngpus];

  float *xx[ngpus]; 
  int **max_IDs[ngpus];
  IM_Ptr **M1[ngpus];

  for (int g = 0; g < ngpus; ++g){
     if (MA[g].INPUT_DATA_int != NULL)        M1[g] = (IM_Ptr **) MA[g].INPUT_DATA_int;
     else if (MA[g].INPUT_DATA_float != NULL) M1[g] = (IM_Ptr **) MA[g].INPUT_DATA_float;
     else if (MA[g].INPUT_DATA_uchar != NULL) M1[g] = (IM_Ptr **) MA[g].INPUT_DATA_uchar;
     else { fprintf(stderr,"run_epoch:  problem setting up pointer to the input data\n"); return; }
  }


  for (int g = 0; g < ngpus; ++g){
    input[g] = MA[g].input;
    input_sparse_indx[g] = MA[g].input_sparse_indx;
    synapses[g] = MA[g].synapses;
    ds[g] = MA[g].ds;
    tot_input[g] = MA[g].tot_input;
    xx[g] = MA[g].xx;
    max_IDs[g] = MA[g].max_IDs;
    if (frequency_scaling==1)
      tot_input_raw[g] = MA[g].tot_input_raw;
  }

  float **max_per_block = new float*[ngpus];

  int nthreads_max_of_AbsVal;
  int nblocks_max_of_AbsVal;
  int shared_mem_size_max_of_AbsVal;
  int W_Nchannels = W*Nchannels;

  size_t cuda_func_nthreads = 256;
  size_t cuda_func_nblocks = (hid*N+cuda_func_nthreads-1)/cuda_func_nthreads;
  if (cuda_func_nblocks > 64*1024) cuda_func_nblocks = 64*1024;

  nthreads_max_of_AbsVal = 256;
  nblocks_max_of_AbsVal = (hid*N+nthreads_max_of_AbsVal-1)/nthreads_max_of_AbsVal;
  if (nblocks_max_of_AbsVal > 64*1024) nblocks_max_of_AbsVal = 64*1024;
  shared_mem_size_max_of_AbsVal =  nthreads_max_of_AbsVal/32*sizeof(float);//  (nthreads_max_of_AbsVal+31)/32*sizeof(float);



  int LDA = (int) N;//ds
  int LDB = (int) N;//input
  int LDC = hid;//tot_input
  float alpha = 1.0;
  float beta = 0.0;
  int NN = (int) N;

  unsigned int d1;
  unsigned int d2 = 4;

  if (W_Nchannels < 16) d1= 16;
  else if (W_Nchannels < 32) d1 = 32;
  else if (W_Nchannels < 128) d1 = 64;
  else d1 = 128;

  int p_1 = p-1;
  int s_mem_size_get_max_and_m_max_kernel_2 = 64*(sizeof(int)+sizeof(float)) + m*(sizeof(int)+sizeof(float));

  int nthreads_func_ds_eq_ds_input_kernel = 128;
  if (N < 33) nthreads_func_ds_eq_ds_input_kernel=32;
  else if (N < 65) nthreads_func_ds_eq_ds_input_kernel = 64;
  else if (N < 97) nthreads_func_ds_eq_ds_input_kernel = 96;



  for (int g = 0; g < ngpus; ++g){
    cudaSetDevice(list_gpus[g]);
    cudaMalloc(&max_per_block[g],nblocks_max_of_AbsVal*sizeof(float));   
    //cudaMallocManaged (&max_per_block[g],nblocks_max_of_AbsVal*sizeof(float));
    //for (int i=0; i < nblocks_max_of_AbsVal; ++i) max_per_block[g][i] = 0.0;
   // size_t count = (size_t) nblocks_max_of_AbsVal * sizeof(float);
   // cudaError_t cuERROR=  cudaMemPrefetchAsync (&max_per_block[g][0], count, 0 );
   // cudaDeviceSynchronize();
   // if (cuERROR == cudaErrorInvalidValue){
   //   fprintf(stderr,"cudaMemPrefetchAsync: cudaErrorInvalidValue pointer=%p, count=%llu  \n", &max_per_block[0], count  );
   // }
   // else if (cuERROR == cudaErrorInvalidDevice)  
   //   fprintf(stderr,"cudaMemPrefetchAsync: cudaErrorInvalidDevice\n");
  }


  float *max_abs_val = new float[ngpus]; //will use ATS, result will be stored in a CPU memory 
  cudaHostRegister(max_abs_val,ngpus*sizeof(float),cudaHostAllocPortable);


    
  int cuGraph_FLAG=0;
  cudaStream_t stream1[ngpus], stream_for_graph[ngpus];
  cudaGraph_t graph[ngpus];
  cudaGraphExec_t graphExec[ngpus];

  for (int g = 0; g < ngpus; ++g){
    cudaSetDevice(list_gpus[g]);
    CUCHECK( cudaStreamCreate(&stream1[g]) );
    CUCHECK( cudaStreamCreate(&stream_for_graph[g]) );
  }
  cublasHandle_t handle[ngpus];
  int CUBLAS_INIT_FLAG = 0;
  if (sparse_input == 0){
    if (CUBLAS_INIT_FLAG==0){
      fprintf(stderr,"calling  cublasCreate\n");
      cublasStatus_t stat;
      for (int g = 0; g < ngpus; ++g){
            cudaSetDevice(list_gpus[g]);
            stat = cublasCreate (& handle[g] );
            cublasSetStream(handle[g],stream1[g]);
            cudaDeviceSynchronize();
      }
      CUBLAS_INIT_FLAG=1;
    }
  }
  
    
    uint64_t  work_fraction  = Ns/100;    
    work_fraction = (work_fraction/(Num*ngpus))*(Num*ngpus);
    dim3 nthreads(d1,d2,1);
    dim3 nblocks(Num, 1, 1);

    gpuErrchk( cudaPeekAtLastError() );

    //for multi-gpu gradient averaging and model updates  
    unsigned long long *offsets;
    cudaHostAlloc(&offsets,(ngpus+1)*sizeof(unsigned long long),cudaHostAllocPortable); 
    cudaHostRegister(list_gpus,ngpus*sizeof(int),cudaHostRegisterPortable);
    split_to_chuncks<float>(hid * N, (unsigned long long) ngpus, &offsets[0]);
    
    int *device_list[ngpus];
  
    for (int g = 0; g < ngpus; ++g){
      cudaSetDevice(list_gpus[g]);
      cudaMallocManaged(&device_list[g],ngpus*sizeof(int));
      for (int gg = 0; gg < ngpus; ++gg)
         device_list[g][gg] = list_gpus[gg];
    }

    float **data;
    float **model;
    cudaHostAlloc(&data,ngpus*sizeof(float*),cudaHostAllocPortable);
    cudaHostAlloc(&model,ngpus*sizeof(float*),cudaHostAllocPortable);
    for (int g = 0; g  < ngpus; ++g) {
      data[g] = ds[g][0];
      model[g] = synapses[g][0];
    }

    float ***data2D;
    float ***model2D;
    float **xx2D;
    cudaHostAlloc(&data2D,ngpus*sizeof(float**),cudaHostAllocPortable);
    cudaHostAlloc(&model2D,ngpus*sizeof(float**),cudaHostAllocPortable);
    cudaHostAlloc(&xx2D,ngpus*sizeof(float*),cudaHostAllocPortable);


    for (int g = 0; g  < ngpus; ++g) {
      data2D[g] = ds[g];
      model2D[g] = synapses[g];
      xx2D[g] = xx[g];
    }
    unsigned long long *offsets2D;
    cudaHostAlloc(&offsets2D,(ngpus+1)*sizeof(unsigned long long),cudaHostAllocPortable);

    split_to_chuncks<float>(N, (unsigned long long) ngpus, &offsets2D[0]);

    gpuErrchk( cudaPeekAtLastError() );

    for (uint64_t k = 0; k < Ns; k += (Num*ngpus)){

     #pragma omp parallel for
     for (int g=0; g < ngpus; ++g){

        uint64_t input_start = k + g * Num;
        uint64_t input_end   = input_start + Num;
        if (input_start > Ns) input_start = Ns;
        if (input_end   > Ns) input_end = Ns; 

       cudaSetDevice(list_gpus[g]);

       if (sparse_input == 0)
         inflate_from_M_kernel_2<IM_Ptr, input_scalling_factor><<<nblocks,nthreads,0,stream1[g]>>>(input_end-input_start,  input_start,  W, IM_WIDTH,
                                                                                                  ST, Nchannels, M1[g], HEIGHT_W_ST_block, WIDTH_W_ST_block, 
                                                                                                  myvector, input[g]);
       else{
         inflate_from_M_kernel_4<IM_Ptr ><<<nblocks,32,0,stream1[g]>>> (input_end-input_start, W, IM_WIDTH, vocabulary_size, M1[g], 
                                                                         &myvector[input_start], input_sparse_indx[g]);
         //  inflate_from_M_kernel_4_and_prune<IM_Ptr ><<<nblocks,32,sizeof(IM_Ptr)*W,stream1[g]>>>(input_end-input_start, 
         //                              W, IM_WIDTH, vocabulary_size, M1[g], &myvector[input_start], input_sparse_indx[g]);
       }

       if (p_1%2 == 0)
           cuda_func1_even<<<cuda_func_nblocks,cuda_func_nthreads,0,stream1[g]>>>(hid*N, synapses[g][0], ds[g][0],p_1);
       else
           cuda_func1_odd<<<cuda_func_nblocks,cuda_func_nthreads,0,stream1[g]>>>(hid*N, synapses[g][0], ds[g][0],p_1);


       if (sparse_input==0){
         cublasStatus_t stat;
         stat = cublasSgemm(handle[g],CUBLAS_OP_T,CUBLAS_OP_N, hid,Num, NN, &alpha, ds[g][0], LDA, input[g][0],LDB, &beta,tot_input[g][0],LDC);
       }
       else{
         if ( (Lmid == 1.0) && (Lbase == 1.0) ){ 
           if (frequency_scaling == 0)
              dense_sparse_matmul_special<<<Num,32,0,stream1[g]>>>(Num,hid,N,W,ds[g],input_sparse_indx[g],tot_input[g]);
           else{
              dense_sparse_matmul_special<<<Num,32,0,stream1[g]>>>(Num,hid,N,W,ds[g],input_sparse_indx[g],tot_input[g],MA[g].word_inv_frequency, vocabulary_size);
              dense_sparse_matmul_special<<<Num,32,0,stream1[g]>>>(Num,hid,N,W,ds[g],input_sparse_indx[g],tot_input_raw[g]);
           }
    
         }
         else{
           if (frequency_scaling == 0)
              dense_sparse_matmul_special<<<Num,32,0,stream1[g]>>>(Num,hid,N,W,ds[g],input_sparse_indx[g],tot_input[g],Lmid,Lbase);
           else{
              dense_sparse_matmul_special<<<Num,32,0,stream1[g]>>>(Num,hid,N,W,ds[g],input_sparse_indx[g],tot_input[g],Lmid,Lbase,MA[g].word_inv_frequency, vocabulary_size);
              dense_sparse_matmul_special<<<Num,32,0,stream1[g]>>>(Num,hid,N,W,ds[g],input_sparse_indx[g],tot_input_raw[g],Lmid,Lbase);
           } 
         }
       }

       fill_func_kernel<<<cuda_func_nblocks,cuda_func_nthreads,0,stream1[g]>>>(hid,xx[g],0.0);
       if (frequency_scaling == 0)
         get_max_and_m_max_kernel_2<<<Num,64,s_mem_size_get_max_and_m_max_kernel_2,stream1[g]>>>(Num,hid,tot_input[g],m,max_IDs[g]);
       else
         get_max_and_m_max_kernel_2<<<Num,64,s_mem_size_get_max_and_m_max_kernel_2,stream1[g]>>>(Num,hid,tot_input_raw[g],m,max_IDs[g]);

       xx_atomic_update_kernel<<<Num,1,0,stream1[g]>>>(Num,max_IDs[g],xx[g],tot_input[g],delta);
       cuda_ds_xx_t_synapses_kernel<<<hid,(N<512?N:512),0,stream1[g]>>>(N, hid, ds[g], xx[g], synapses[g]);


       if (sparse_input == 0)
          func_ds_eq_ds_input_kernel<<<Num,nthreads_func_ds_eq_ds_input_kernel,0,stream1[g]>>>(Num,N,ds[g],input[g],max_IDs[g],delta);
       else{
         if ( (Lmid == 1.0) && (Lbase == 1.0) ){
           if (frequency_scaling == 0)
             func_ds_eq_ds_input_kernel_sparse<<<Num,nthreads_func_ds_eq_ds_input_kernel,0,stream1[g]>>>(Num,N,ds[g],W,input_sparse_indx[g],max_IDs[g],delta);
           else
             func_ds_eq_ds_input_kernel_sparse<<<Num,nthreads_func_ds_eq_ds_input_kernel,0,stream1[g]>>>(Num,N,ds[g],W,input_sparse_indx[g],max_IDs[g],delta,MA[g].word_inv_frequency, vocabulary_size);
         }
         else{
           if (frequency_scaling == 0)
             func_ds_eq_ds_input_kernel_sparse<<<Num,nthreads_func_ds_eq_ds_input_kernel,0,stream1[g]>>>(Num,N,ds[g],W,input_sparse_indx[g],max_IDs[g],delta,Lmid,Lbase);
           else
             func_ds_eq_ds_input_kernel_sparse<<<Num,nthreads_func_ds_eq_ds_input_kernel,0,stream1[g]>>>(Num,N,ds[g],W,input_sparse_indx[g],max_IDs[g],delta,Lmid,Lbase,MA[g].word_inv_frequency, vocabulary_size);
         }
       }
       CUCHECK(cudaStreamSynchronize(stream1[g]));  
      }
      gpuErrchk( cudaPeekAtLastError() );



/*      statistics on ds
      uint64_t nbins = 15;
      uint64_t bin[nbins];
      for (int iii = 0; iii < nbins; ++iii) bin[iii] = 0;
 
      for (uint64_t iii = 0; iii < hid*N; ++iii){       
          float val = fabs(ds[0][0][iii]);
          float cut_off = 1.0;
          int b;
          for (b = 0; b < nbins-1; ++b){
            if (val > cut_off) {
               bin[b]+=1;
               break;
            }
            else cut_off = cut_off*0.1;
          }
          if (b == (nbins-1)) bin[nbins-1] += 1;
      }
     
      float hid_t_N = (float) hid*N;
      fprintf(stderr,"bins: ");
      for (int iii = 0; iii < nbins; ++iii)
        fprintf(stderr,"%lu \t",bin[iii]);
      fprintf(stderr,"\n");

      fprintf(stderr,"bins: ");
      for (int iii = 0; iii < nbins; ++iii)
        fprintf(stderr,"%g \t",(float) bin[iii]/ hid_t_N );
      fprintf(stderr,"\n");
*/
/*
      fprintf(stderr,"bins: %lu \t %lu \t %lu \t %lu \t %lu \t %lu \t %lu \t %lu \t %lu\n",bin[0],bin[1],bin[2],bin[3],bin[4],bin[5],bin[6],bin[7],bin[8]);
      fprintf(stderr,"bins: %g\t %g\t %g\t %g\t %g\t %g\t %g\t %g\t %g\n",(float) bin[0] / hid_t_N,
                                                           (float) bin[1] / hid_t_N,
                                                           (float) bin[2] / hid_t_N,
                                                           (float) bin[3] / hid_t_N,
                                                           (float) bin[4] / hid_t_N,
                                                           (float) bin[5] / hid_t_N,
                                                           (float) bin[6] / hid_t_N,
                                                           (float) bin[7] / hid_t_N,
                                                           (float) bin[8] / hid_t_N);
*/
      #if 0    
      float nc = reduce_by_chunks_and_absmax(data, &offsets[0], device_list, max_per_block, nblocks_max_of_AbsVal, max_abs_val, ngpus);
      #else
      float nc = reduce_by_chunks_and_absmax_sparse_2D(data2D, &offsets2D[0], device_list, max_per_block, nblocks_max_of_AbsVal, max_abs_val, ngpus, hid, xx2D);
      #endif

      gpuErrchk( cudaPeekAtLastError() );

      float eps_inv_nc;
      if (prec > nc) eps_inv_nc = eps/prec;
      else           eps_inv_nc = eps/nc;
      #if 0
      update_model_multiGPU_sparse_2D(data, &offsets[0], device_list, ngpus, model2D, eps_inv_nc);
      #else
      update_model_multiGPU_sparse_2D(data2D, &offsets2D[0], device_list, ngpus, model2D, eps_inv_nc, hid, xx2D);
      #endif

      gpuErrchk( cudaPeekAtLastError() );

     if ( (k+1) > work_fraction ){
        fprintf(stderr,"  run_epoch: %.2f %%  is done \n", ( (double) (k+1))/Ns*100.0);
        work_fraction  = work_fraction +  ((Ns/100)/(Num*ngpus))*(Num*ngpus);
     }
   } //end of  for (int k = 0; k < Ns; k += Num)

      gpuErrchk( cudaPeekAtLastError() );


  cudaHostUnregister(list_gpus);
  cudaFreeHost(data);
  cudaFreeHost(model);
  cudaFreeHost(data2D);
  cudaFreeHost(model2D);
  cudaFreeHost(xx2D);
  cudaFreeHost(offsets);
  cudaFreeHost(offsets2D);

  cudaHostUnregister(max_abs_val);
  delete[] max_abs_val;

  for (int g=0; g < ngpus; ++g){
    cudaFree(max_per_block[g]);
//    CUCHECK(cudaGraphExecDestroy(graphExec[g]));
//    CUCHECK(cudaGraphDestroy(graph[g]));
    CUCHECK(cudaStreamDestroy(stream_for_graph[g]));
    CUCHECK(cudaStreamDestroy(stream1[g]));
    if (sparse_input == 0)
      cublasDestroy(handle[g]);
  }

  delete[] max_per_block;

  gpuErrchk( cudaPeekAtLastError() );

}





#define CACHE_LINE_SIZE 128

template <class ELMNT_TYPE>
void split_to_chuncks(unsigned long long length, unsigned long long nchunks, unsigned long long *offsets){
  if (nchunks == 0) return;
  offsets[0]= 0;

  unsigned long long elements_in_one_cache_line = (unsigned long long) CACHE_LINE_SIZE / sizeof(ELMNT_TYPE);
  unsigned long long num_cache_lines = (length + elements_in_one_cache_line - 1) / elements_in_one_cache_line;

  unsigned long long cache_lines_in_chunk = (num_cache_lines + nchunks - 1) / nchunks;

  for (unsigned long long i = 0; i < nchunks; ++i){
     unsigned long long chunk_size = cache_lines_in_chunk*elements_in_one_cache_line;
     if (offsets[i] + chunk_size > length)
       offsets[i+1] = length;
     else
       offsets[i+1] = offsets[i]+chunk_size;
  }


}


