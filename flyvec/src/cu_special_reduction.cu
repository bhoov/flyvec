#include <cuda.h>
#include <stdio.h>
#include <cuda_runtime.h>


template<class Ptr>
__global__ void reduce_by_chunks_kernel(Ptr **data, unsigned long long *offsets, int *device_list,  int num_devices, int deviceID){


    size_t gtid = threadIdx.x + blockIdx.x*blockDim.x;
    size_t thread_offset = blockDim.x*gridDim.x;

    size_t i_start = offsets[deviceID]+gtid;
    size_t i_end = offsets[deviceID+1]; 

    for (size_t i = i_start; i < i_end; i+=thread_offset){

      double sum = 0.0;
      for (int dev = 0; dev < num_devices; ++dev)
        sum += data[device_list[dev]][i];

      data[deviceID][i] = (Ptr) sum;
    }
}

template<class EL_TYPE>
__global__ void reduce_by_chunks_and_absmax_kernel_sparse_2D(EL_TYPE ***data, uint64_t DIM1, EL_TYPE **zero_or_not, unsigned long long *offsets,  int *device_list,  int num_devices, int deviceID, EL_TYPE *max_per_block){

    extern __shared__ char s[];
    EL_TYPE *max_per_warp = (EL_TYPE*) s;
    int *nnz = (int*) (s + sizeof(EL_TYPE)*blockDim.x/32); //vector with length of num_devices+1

    size_t  i_shift = blockDim.x * gridDim.x;
    size_t  i_start = offsets[deviceID] + threadIdx.x + blockDim.x * blockIdx.x;
    size_t  i_end   = offsets[deviceID+1];

    for (size_t h = blockIdx.y; h < DIM1; h += gridDim.y){
         if (gridDim.y < DIM1)  __syncthreads();
         if (threadIdx.x == 0){
           max_per_block[blockIdx.x + blockIdx.y * gridDim.x] = 0.0;
           nnz[num_devices] = 0;
           for (int dev = 0; dev < num_devices; ++dev){
             if (0 == zero_or_not[dev][h]) 
               nnz[dev] = 0;
             else{
               nnz[dev] = 1;
               nnz[num_devices] += 1;//counter
             }
           }
         } 
         __syncthreads();

         double inv_num_devices = 1.0; // /( (double) num_devices  );
         if (nnz[num_devices] > 0) inv_num_devices = inv_num_devices / ( (double) nnz[num_devices] );

         EL_TYPE max_val = 0.0;

         if (nnz[num_devices] > 0 ){
           for (size_t i = i_start; i < i_end; i += i_shift){
             double sum = 0.0;
             for (int dev = 0; dev < num_devices; ++dev){
               if (nnz[dev] != 0) 
                  sum += data[device_list[dev]][h][i];
             }
             EL_TYPE sum_f = (EL_TYPE) sum*inv_num_devices;
             data[deviceID][h][i] = sum_f;

             EL_TYPE val = fabs(sum_f);
             max_val = val > max_val ? val : max_val;
           }
         }
         else{
           EL_TYPE *data_ptr = data[deviceID][h];
           for (size_t i = i_start; i < i_end; i += i_shift)
             data_ptr[i] = 0.0;
         }

         if (nnz[num_devices] > 0 ){
           //reduction within each warp
           for (int ii=16; ii>=1; ii/=2){
             EL_TYPE val  = __shfl_down_sync(0xffffffff, max_val, ii, 32);
             max_val = max_val > val ? max_val : val;
           }
          //thread threadIdx.x%32 == 0 has the max over a warp of threads
          if (threadIdx.x%32 == 0)
            max_per_warp[threadIdx.x/32] = max_val;
          __syncthreads();
       
          //reduction over data from warps
          if (threadIdx.x == 0){
            for (int ii = 1; ii < blockDim.x/32; ++ii){
              float val = max_per_warp[ii];
              max_val = max_val > val ? max_val : val;
            }
            max_per_block[blockIdx.x + blockIdx.y*gridDim.x] = max_per_block[blockIdx.x + blockIdx.y*gridDim.x] > max_val ? max_per_block[blockIdx.x + blockIdx.y*gridDim.x] : max_val;
          }
        }
    }
}


//number of threads in a block is multiple of 32
template<class EL_TYPE>
__global__ void reduce_by_chunks_and_absmax_kernel(EL_TYPE **data, unsigned long long *offsets, int *device_list,  int num_devices, int deviceID, EL_TYPE *max_per_block){

    extern __shared__ char s[];
    EL_TYPE *max_per_thread = (EL_TYPE*) s;


    size_t gtid = threadIdx.x + blockIdx.x*blockDim.x;
    size_t thread_offset = blockDim.x*gridDim.x;

    size_t i_start = offsets[deviceID]+gtid;
    size_t i_end = offsets[deviceID+1];
    EL_TYPE max_val = 0.0;
    EL_TYPE inv_num_devices = 1.0/((EL_TYPE)num_devices);


    for (size_t i = i_start; i < i_end; i+=thread_offset){

      double sum = 0.0;
      for (int dev = 0; dev < num_devices; ++dev)
        sum += data[device_list[dev]][i];

      EL_TYPE sum_f = (EL_TYPE) sum*inv_num_devices;
      data[deviceID][i] = sum_f;

      EL_TYPE val = fabs(sum_f);  
      max_val = val > max_val ? val : max_val;
    }
    //reduction within each warp
    for (int i=16; i>=1; i/=2){
       EL_TYPE val  = __shfl_down_sync(0xffffffff, max_val, i, 32);
       max_val = max_val > val ? max_val : val;
    }

    //thread threadIdx.x%32 == 0 has the max over a warp of threads
    if (threadIdx.x%32 == 0)
      max_per_thread[threadIdx.x/32] = max_val;

    __syncthreads();

    if (threadIdx.x == 0){
     for (int n = 1; n < blockDim.x / 32; ++n){
        float val = max_per_thread[n];
        max_val = max_val > val ? max_val : val;
    
     max_per_block[blockIdx.x] = max_val;
   }
  }
}

template<class EL_TYPE>
__global__ void max_of_AbsVal_kernel_finish(size_t N, EL_TYPE *data, EL_TYPE *result){

  extern __shared__ char s[];
  EL_TYPE *max_per_thread = (EL_TYPE*) s;

  int i_start = threadIdx.x + blockDim.x*blockIdx.x;
  int shift = blockDim.x*gridDim.x;

  EL_TYPE my_max = -1.0;
  for (int i = i_start; i < N; i += shift){
     EL_TYPE val = data[i];
     my_max = my_max > val ? my_max : val;
  }

  //reduction within each warp
  for (int i=16; i>=1; i/=2){
     EL_TYPE val  = __shfl_down_sync(0xffffffff, my_max, i, 32);
     my_max = my_max > val ? my_max : val;
  }

  // __syncthreads();

  if (threadIdx.x%32 == 0)
    max_per_thread[threadIdx.x/32] = my_max;

  __syncthreads();

  if (threadIdx.x == 0){
    for (int n = 1; n < blockDim.x / 32; ++n){
       EL_TYPE val = max_per_thread[n];
       my_max = my_max > val ? my_max : val;
    }
    result[0] = my_max;
  }

}



//template<class Ptr>
void reduce_by_chunks(float **data,  unsigned long long *offsets, int **device_list, int num_devices){

   size_t nthreads=256;

   for (int dev = 0; dev < num_devices; ++dev){
     cudaSetDevice(dev);
     size_t nblocks = (offsets[dev+1]- offsets[dev] + nthreads - 1)/nthreads;
     nblocks = nblocks > (64*1024) ? (64*1024) : nblocks;

     reduce_by_chunks_kernel<float><<<nblocks,nthreads>>>(data, offsets, device_list[dev], num_devices , dev);
   }
   for (int dev = 0; dev < num_devices; ++dev){
     cudaSetDevice(dev);
     cudaDeviceSynchronize();
   }
}

//template<class Ptr>
float reduce_by_chunks_and_absmax(float **data,  unsigned long long *offsets, int **device_list, float **max_per_block_in, unsigned long long nblocks_in, float *max_abs_val,  int num_devices){

   size_t nthreads=256;
   size_t shared_mem_size_max_of_AbsVal = nthreads/32*sizeof(float);
   float **max_per_block;
   max_per_block = new float*[num_devices];
   for (int dev = 0; dev < num_devices; ++dev){
     cudaSetDevice(dev);
     size_t nblocks = (offsets[dev+1]- offsets[dev] + nthreads - 1)/nthreads;
     nblocks = nblocks > (64*1024) ? (64*1024) : nblocks;
     if (nblocks_in < nblocks) nblocks = nblocks_in;
     max_per_block[dev] = max_per_block_in[dev];
     reduce_by_chunks_and_absmax_kernel<float><<<nblocks,nthreads,shared_mem_size_max_of_AbsVal>>>(data, offsets, device_list[dev], num_devices , dev, max_per_block[dev]);
     max_of_AbsVal_kernel_finish<float><<<1,nthreads,shared_mem_size_max_of_AbsVal>>>(nblocks,max_per_block[dev],&max_abs_val[dev]);
    
   }
   float result = 0.0;

   for (int dev = 0; dev < num_devices; ++dev){
     cudaSetDevice(dev);
     cudaDeviceSynchronize();
     result = result > max_abs_val[dev] ? result : max_abs_val[dev];
   }
   delete[] max_per_block;
   return result;

}


float reduce_by_chunks_and_absmax_sparse_2D(float ***data2D,  unsigned long long *offsets, int **device_list, float **max_per_block_in, unsigned long long nblocks_in, float *max_abs_val,  int num_devices, uint64_t DIM1, float **zero_or_not){

   size_t nthreads=128;
   size_t shared_mem_size_max_of_AbsVal = nthreads/32*sizeof(float) + sizeof(int)*(num_devices+1);
   float **max_per_block;
   max_per_block = new float*[num_devices];
   for (int dev = 0; dev < num_devices; ++dev){
     cudaSetDevice(dev);
     dim3 nblocks = {128,(uint) DIM1,1};
     max_per_block[dev] = max_per_block_in[dev];
     reduce_by_chunks_and_absmax_kernel_sparse_2D<float><<<nblocks,nthreads,shared_mem_size_max_of_AbsVal>>>(data2D, DIM1, zero_or_not, offsets, device_list[dev], num_devices , dev, max_per_block[dev]);
     max_of_AbsVal_kernel_finish<float><<<1,512,shared_mem_size_max_of_AbsVal>>>(nblocks.x*nblocks.y, max_per_block[dev], &max_abs_val[dev]);

   }
   float result = 0.0;

   for (int dev = 0; dev < num_devices; ++dev){
     cudaSetDevice(dev);
     cudaDeviceSynchronize();
     result = result > max_abs_val[dev] ? result : max_abs_val[dev];
   }
   delete[] max_per_block;
   return result;

}


template<class EL_TYPE>
__global__ void update_model_multiGPU_kernel(EL_TYPE *data,  unsigned long long *offsets, int *device_list, int num_devices, EL_TYPE** model, EL_TYPE scale_factor, int deviceID){


    size_t gtid = threadIdx.x + blockIdx.x*blockDim.x;
    size_t thread_offset = blockDim.x*gridDim.x;

    size_t i_start = offsets[deviceID]+gtid;
    size_t i_end = offsets[deviceID+1];

    for (size_t i = i_start; i < i_end; i+=thread_offset){
       EL_TYPE val = data[i];
       EL_TYPE input = model[deviceID][i];
       EL_TYPE update = input +  scale_factor*val;

       for (int dev = 0; dev < num_devices; ++dev)
          model[device_list[dev]][i] = update;
       
    }
}




template<class EL_TYPE>
__global__ void update_model_multiGPU_kernel_sparse(EL_TYPE *data,  unsigned long long *offsets, int *device_list, int num_devices, EL_TYPE** model, EL_TYPE scale_factor, int deviceID){


    size_t gtid = threadIdx.x + blockIdx.x*blockDim.x;
    size_t thread_offset = blockDim.x*gridDim.x;

    size_t i_start = offsets[deviceID]+gtid;
    size_t i_end = offsets[deviceID+1];

    for (size_t i = i_start; i < i_end; i+=thread_offset){
       EL_TYPE val = data[i];
       if (fabs(val) > 1.0e-10){
         EL_TYPE input = model[deviceID][i];
         EL_TYPE update = input +  scale_factor*val;

         for (int dev = 0; dev < num_devices; ++dev)
            model[device_list[dev]][i] = update;
       }
    }

}


template<class EL_TYPE>
__global__ void update_model_multiGPU_kernel_sparse_2D(EL_TYPE **data,  unsigned long long *offsets, uint64_t DIM1, EL_TYPE **zero_or_not, int *device_list, int num_devices, EL_TYPE*** model, EL_TYPE scale_factor, int deviceID){

    __shared__ int nnz[1];

    size_t  i_start = offsets[deviceID] + threadIdx.x + blockDim.x*blockIdx.x;
    size_t  i_end   = offsets[deviceID+1];

    for (size_t h = blockIdx.y; h < DIM1; h += gridDim.y){
        
         if (threadIdx.x == 0){
           nnz[0] = 0;
           for (int dev = 0; dev < num_devices; ++dev){
             if (0 != zero_or_not[dev][h]){
               nnz[0] += 1;//counter
             }
           }
         }
         __syncthreads();
        
        if (nnz[0] > 0){//update model
           EL_TYPE *model_ptr = model[deviceID][h];

           for (size_t i = i_start; i < i_end; i+=blockDim.x*gridDim.x){
               EL_TYPE val = scale_factor*data[h][i];

               if ( fabs(val) > 1.e-10){
                 EL_TYPE update = model_ptr[i] + val;
               
                 for (int dev = 0; dev < num_devices; ++dev)
                    model[device_list[dev]][h][i] = update;

              }
           }
        }
        if (gridDim.y < DIM1)  __syncthreads();

    } 
}


void update_model_multiGPU(float **data,  unsigned long long *offsets, int **device_list, int num_devices, float** model, float scale_factor){

   size_t nthreads=256;

   for (int dev = 0; dev < num_devices; ++dev){
     cudaSetDevice(dev);
     size_t nblocks = (offsets[dev+1]- offsets[dev] + nthreads - 1)/nthreads;
     nblocks = nblocks > (64*1024) ? (64*1024) : nblocks;
     update_model_multiGPU_kernel_sparse<float><<<nblocks,nthreads>>>(data[dev],  offsets, device_list[dev], num_devices, model, scale_factor, dev);
    }
   for (int dev = 0; dev < num_devices; ++dev){
     cudaSetDevice(dev);
     cudaDeviceSynchronize();
   }
}

void update_model_multiGPU_sparse_2D(float ***data,  unsigned long long *offsets, int **device_list, int num_devices, float*** model, float scale_factor, uint64_t DIM1, float **zero_or_not){

   size_t nthreads=256;

   for (int dev = 0; dev < num_devices; ++dev){
     cudaSetDevice(dev);
     dim3 nblocks = {64,(uint) DIM1,1};
     update_model_multiGPU_kernel_sparse_2D<float><<<nblocks,nthreads>>>(data[dev],  offsets, DIM1, zero_or_not, device_list[dev], num_devices, model, scale_factor, dev);
    }
   for (int dev = 0; dev < num_devices; ++dev){
     cudaSetDevice(dev);
     cudaDeviceSynchronize();
   }
}

