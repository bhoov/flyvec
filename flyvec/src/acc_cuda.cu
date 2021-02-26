#include <cuda.h>
#include <cuda_runtime.h>


#ifdef IMAGE_AS_CHAR
#define IMTYPE unsigned char
#else
#define IMTYPE float
#endif


__global__
void conv_TOP_K_kernel(uint64_t Npatches, uint64_t N, float** image_data_patches, uint64_t hid, float **synapses, uint64_t TOP_K,
               uint64_t** maxPATCH_INDEX, float **maxPATCH_VAL, double *sum_all);

__global__
void  normalize_kernel(uint64_t hid, uint64_t N, float **input, float **output);

__global__
void inflate_from_M_kernel_2(uint64_t W, uint64_t IM_WIDTH, uint64_t IM_HEIGHT, uint64_t WIDTH_W_ST_block, uint64_t ST, uint64_t Nchannels, IMTYPE **M1, float **image_data);

__global__
void inflate_and_max_kernel(uint64_t Nchannels, uint64_t W, uint64_t IM_WIDTH, uint64_t IM_HEIGHT, uint64_t WIDTH_W_ST_block, uint64_t ST,  float ***DATA_MAX_VAL_PER_CHANNEL, float **OUTPUT);


__global__
void update_DATA_MAX_VAL_PER_CHANNEL_kernel(uint64_t HEIGHT_W_ST_block, uint64_t WIDTH_W_ST_block, uint64_t TOP_K, float **maxPATCH_VAL, uint64_t **maxPATCH_INDEX, float ***DATA_MAX_VAL_PER_CHANNEL);

__global__
void zero_DATA_MAX_VAL_PER_CHANNEL_kernel(uint64_t HEIGHT_W_ST_block, uint64_t WIDTH_W_ST_block, uint64_t TOP_K, float **maxPATCH_VAL, uint64_t **maxPATCH_INDEX, float ***DATA_MAX_VAL_PER_CHANNEL);

//template<class Tptr, class Tdim, class Tscal>
//void fill( Tdim N, Tptr* data, Tscal value);


void conv_TOP_K_cuda(uint64_t Npatches, uint64_t N, float** image_data_patches, uint64_t hid, float **synapses, uint64_t TOP_K,
               uint64_t** maxPATCH_INDEX, float **maxPATCH_VAL, double *sum_all){


     //double *sum_all;
     dim3 nthreads(32,8,1);
     int nblocks = Npatches;
     //cudaMalloc(&sum_all,nblocks*hid*sizeof(double)); 
     int shared_memory_size = (sizeof(float) + sizeof(uint64_t))*8;
//     conv_TOP_K_kernel<<<nblocks,nthreads,shared_memory_size>>>(Npatches,N,image_data_patches,hid,synapses,TOP_K,maxPATCH_INDEX,maxPATCH_VAL,sum_all);
     conv_TOP_K_kernel<<<nblocks,nthreads,shared_memory_size>>>(Npatches,N,image_data_patches,hid,synapses,TOP_K,maxPATCH_INDEX,maxPATCH_VAL,sum_all);
     cudaDeviceSynchronize();
     //cudaFree(sum_all);


}

__global__
void conv_TOP_K_kernel(uint64_t Npatches, uint64_t N, float** image_data_patches, uint64_t hid, float **synapses, uint64_t TOP_K,
               uint64_t** maxPATCH_INDEX, float **maxPATCH_VAL, double *sum_all){

  extern __shared__ char s[];
  uint64_t *s_ind = (uint64_t*) s;
  float *s_val = (float*) (s + sizeof(uint64_t)*blockDim.y);

//  float *s_val = (float*) s; 
//  uint64_t *s_ind = (uint64_t*)   (s + sizeof(float)*blockDim.y); 

//  __shared__ float s_val[4];
//  __shared__ uint64_t s_ind[4];

  int btid = threadIdx.x + threadIdx.y*blockDim.x;

  for (uint64_t patch = blockIdx.x ; patch < Npatches; patch += gridDim.x){

     float *p_ptr = image_data_patches[patch];

     double *sums = sum_all + blockIdx.x*hid;

     for (uint64_t h = threadIdx.y; h < hid; h += blockDim.y){
       double sum = 0.0;
       for (uint64_t nn = threadIdx.x; nn < N; nn += blockDim.x)
          sum += p_ptr[nn]*synapses[h][nn];

       __syncwarp();

       //reduction within warp 
       for (int i=16; i>=1; i/=2)
          sum += __shfl_down_sync(0xffffffff, sum, i, 32);
      
       if (threadIdx.x == 0)  
         sums[h] = sum;
     }
      __syncthreads();

     for (uint64_t k = 0; k < TOP_K; ++k){
        float max_val = sums[0];
        uint64_t max_ind = 0;
        for (uint64_t h = btid; h < hid; h += blockDim.y*blockDim.x ){
          if (sums[h] > max_val){ max_val = sums[h]; max_ind = h;}
        }
       //reduction within warp
        __syncwarp();
        for (int i=16; i>=1; i/=2){
          float     val = __shfl_down_sync(0xffffffff, max_val, i, 32);
          uint64_t  ind = __shfl_down_sync(0xffffffff, max_ind, i, 32);
          if (val > max_val){
            max_val = val;
            max_ind = ind;
          }
        }

        if (threadIdx.x == 0){
          s_val[threadIdx.y] = max_val;
          s_ind[threadIdx.y] = max_ind;
        }

        __syncthreads();
  
       if (btid == 0){ //replaced s_val[0] with max_val, and s_ind[0] with max_ind
          max_val = s_val[0];
          max_ind = s_ind[0];
          for (int i=1; i<blockDim.y; ++i){
            if (s_val[i] > max_val){
              max_val = s_val[i]; 
              max_ind = s_ind[i];
            }
          }
          sums[max_ind] = 0.0;
          maxPATCH_INDEX[patch][k] = max_ind;
          maxPATCH_VAL[patch][k] = max_val;
        }
    
        __syncthreads();
     }
   
  }

}

void  normalize_cuda(uint64_t hid, uint64_t N, float **input, float **output){

   int nthreads=128;
   int nblocks = hid;
   int shared_memory_size = sizeof(float)*nthreads/32;
   normalize_kernel<<<nblocks,nthreads,shared_memory_size>>>(hid,N,input,output);
   cudaDeviceSynchronize();

}
__global__
void  normalize_kernel(uint64_t hid, uint64_t N, float **input, float **output){


  extern __shared__ char s[];
  float * s_sum = (float*) s;

  float **out = output;
  if (output == NULL) out = input; //inplace

 
  for (uint64_t h=blockIdx.x; h < hid; h+=gridDim.x){
     double sum = 0.0;
     for (uint64_t c = threadIdx.x; c < N; c+=blockDim.x)
       sum += input[h][c]*input[h][c];

     __syncwarp();
     //reductoin within a warp
     for (int i=16; i>=1; i/=2)
       sum += __shfl_down_sync(0xffffffff, sum, i, 32);
     
     //reduction over warps
     if (threadIdx.x%32 == 0)
       s_sum[threadIdx.x/32] = sum;

     __syncthreads();
     if (threadIdx.x == 0){
       for (int i = 1; i < blockDim.x/32; ++i)
         sum += s_sum[i];
       s_sum[0] = sqrt(sum);
     }
     __syncthreads();

     for (uint64_t c = threadIdx.x; c < N; c+=blockDim.x)
       out[h][c] = input[h][c]/s_sum[0];
  }

}


void inflate_from_M_cuda(uint64_t W, uint64_t IM_WIDTH, uint64_t IM_HEIGHT, uint64_t WIDTH_W_ST_block, uint64_t ST, uint64_t Nchannels, IMTYPE **M1, float **image_data){

    dim3 nblocks(IM_HEIGHT-W+1,IM_WIDTH-W+1,1);
    dim3 nthreads(64,4,1);
    inflate_from_M_kernel_2<<<nblocks,nthreads>>>(W,IM_WIDTH,IM_HEIGHT,WIDTH_W_ST_block,ST,Nchannels,M1,image_data);
    cudaDeviceSynchronize();
}

__global__
void inflate_from_M_kernel_2(uint64_t W, uint64_t IM_WIDTH, uint64_t IM_HEIGHT, uint64_t WIDTH_W_ST_block, uint64_t ST, uint64_t Nchannels, IMTYPE **M1, float **image_data){

    #ifdef IMAGE_AS_CHAR
    float scal = 1.0/255.0;
    #else
    float scal = 1.0;
    #endif

    
//    for (uint64_t r_mm = 0; r_mm < IM_HEIGHT-W+1; r_mm+=ST){
    for (uint64_t r_mm = blockIdx.x*ST; r_mm < IM_HEIGHT-W+1; r_mm+= gridDim.x*ST){
//      for (uint64_t c_mm = 0; c_mm < IM_WIDTH-W+1; c_mm+=ST){
      for (uint64_t c_mm = blockIdx.y*ST; c_mm < IM_WIDTH-W+1; c_mm+=gridDim.y*ST){

        for (uint64_t rr_M = threadIdx.y; rr_M < W; rr_M+=blockDim.y){
          IMTYPE *MM = M1[0] + (r_mm+rr_M)*IM_WIDTH*Nchannels;

         for (uint64_t cc_M = threadIdx.x; cc_M < W*Nchannels; cc_M += blockDim.x){
          image_data[r_mm*WIDTH_W_ST_block + c_mm][cc_M + rr_M*W*Nchannels] = scal*MM[c_mm*Nchannels+cc_M];
         }
       }

      }
    }
}

void inflate_and_max_cuda(uint64_t Nchannels, uint64_t W, uint64_t IM_WIDTH, uint64_t IM_HEIGHT, uint64_t WIDTH_W_ST_block, uint64_t ST,  float ***DATA_MAX_VAL_PER_CHANNEL, float **OUTPUT){


//    dim3 nblocks(Nchannels,10,1);
    dim3 nblocks(WIDTH_W_ST_block,Nchannels,1);

    unsigned int blockDim_x=32, blockDim_y=8;
    if (W < 8) {blockDim_x = 8; blockDim_y=4;}
    dim3 nthreads(blockDim_x,blockDim_y,1);
    int shared_memory_size = sizeof(float)*blockDim_y;
    inflate_and_max_kernel<<<nblocks,nthreads,shared_memory_size>>>(Nchannels, W, IM_WIDTH, IM_HEIGHT, WIDTH_W_ST_block, ST,  DATA_MAX_VAL_PER_CHANNEL, OUTPUT);
    cudaDeviceSynchronize();


}

//assumption blockDim.x%32 == 0
__global__
void inflate_and_max_kernel(uint64_t Nchannels, uint64_t W, uint64_t IM_WIDTH, uint64_t IM_HEIGHT, uint64_t WIDTH_W_ST_block, uint64_t ST,  float ***DATA_MAX_VAL_PER_CHANNEL, float **OUTPUT){

 extern __shared__ char s[];
 float * s_max_val = (float*) s;

 int btid = threadIdx.y*blockDim.x + threadIdx.x;

 
 for (uint64_t channel = blockIdx.y; channel < Nchannels; channel+=gridDim.y){

    //for (uint64_t r_mm = 0; r_mm < IM_HEIGHT-W+1; r_mm+=ST){
    for (uint64_t r_mm = blockIdx.x*ST; r_mm < IM_HEIGHT-W+1; r_mm += gridDim.x*ST){
      for (uint64_t c_mm = 0; c_mm < IM_WIDTH-W+1; c_mm+=ST){
        float max_val = 0;
        for (uint64_t rr_M = threadIdx.y; rr_M < W; rr_M+=blockDim.y){
          float *MM = DATA_MAX_VAL_PER_CHANNEL[channel][0] + (r_mm+rr_M)*IM_WIDTH;
          for (uint64_t cc_M = threadIdx.x; cc_M < W; cc_M+=blockDim.x){
            max_val = max_val > MM[c_mm+cc_M] ?  max_val : MM[c_mm+cc_M];
          }
        }
        __syncwarp(); 

        //reduce within each warp
#if 0
        for (int i=16; i>=1; i/=2){
          float max_val_from_another_thread = __shfl_down_sync(0xffffffff, max_val, i, 32);
          max_val = max_val > max_val_from_another_thread ? max_val : max_val_from_another_thread; 
        }
#else
        for (int i=blockDim.x/2; i>=1; i/=2){
          float max_val_from_another_thread = __shfl_down_sync(0xffffffff, max_val, i, blockDim.x);
          max_val = max_val > max_val_from_another_thread ? max_val : max_val_from_another_thread;
        }
#endif


        //reduction over warps
        if (threadIdx.x == 0)
          s_max_val[threadIdx.y] = max_val;

        __syncthreads();
        if (btid == 0){
          max_val = s_max_val[0];
          for (int i = 1; i < blockDim.y; ++i)
            max_val = max_val > s_max_val[i] ? max_val : s_max_val[i];

          OUTPUT[r_mm*WIDTH_W_ST_block/ST + c_mm/ST][channel] = max_val;

        } 
        __syncthreads();

      }
    }
 }
}

void update_DATA_MAX_VAL_PER_CHANNEL(uint64_t HEIGHT_W_ST_block, uint64_t WIDTH_W_ST_block, uint64_t TOP_K, float **maxPATCH_VAL, uint64_t **maxPATCH_INDEX, float ***DATA_MAX_VAL_PER_CHANNEL){

  dim3 nblocks(HEIGHT_W_ST_block*WIDTH_W_ST_block,1,1);
  dim3 nthreads(32,1,1);
  if (maxPATCH_VAL == NULL)
    zero_DATA_MAX_VAL_PER_CHANNEL_kernel<<<nblocks,nthreads>>>(HEIGHT_W_ST_block,WIDTH_W_ST_block, TOP_K, maxPATCH_VAL,maxPATCH_INDEX,DATA_MAX_VAL_PER_CHANNEL);
  else 
    update_DATA_MAX_VAL_PER_CHANNEL_kernel<<<nblocks,nthreads>>>(HEIGHT_W_ST_block,WIDTH_W_ST_block, TOP_K, maxPATCH_VAL,maxPATCH_INDEX,DATA_MAX_VAL_PER_CHANNEL);
  cudaDeviceSynchronize();

}

__global__
void update_DATA_MAX_VAL_PER_CHANNEL_kernel(uint64_t HEIGHT_W_ST_block, uint64_t WIDTH_W_ST_block, uint64_t TOP_K, float **maxPATCH_VAL, uint64_t **maxPATCH_INDEX, float ***DATA_MAX_VAL_PER_CHANNEL){

   for (uint64_t patch_index = blockIdx.x; patch_index < HEIGHT_W_ST_block*WIDTH_W_ST_block; patch_index+=gridDim.x){
     uint64_t patch_row_index = patch_index/WIDTH_W_ST_block;
     uint64_t patch_c_index =   patch_index%WIDTH_W_ST_block;
     for (uint64_t m = threadIdx.x; m < TOP_K; m+=blockDim.x){
       float val = maxPATCH_VAL[patch_index][m];
       DATA_MAX_VAL_PER_CHANNEL[ maxPATCH_INDEX[patch_index][m] ][patch_row_index][patch_c_index] = val;
     }
   }
}


__global__
void zero_DATA_MAX_VAL_PER_CHANNEL_kernel(uint64_t HEIGHT_W_ST_block, uint64_t WIDTH_W_ST_block, uint64_t TOP_K, float **maxPATCH_VAL, uint64_t **maxPATCH_INDEX, float ***DATA_MAX_VAL_PER_CHANNEL){

   for (uint64_t patch_index = blockIdx.x; patch_index < HEIGHT_W_ST_block*WIDTH_W_ST_block; patch_index+=gridDim.x){
     uint64_t patch_row_index = patch_index/WIDTH_W_ST_block;
     uint64_t patch_c_index =   patch_index%WIDTH_W_ST_block;
     for (uint64_t m = threadIdx.x; m < TOP_K; m+=blockDim.x){
       DATA_MAX_VAL_PER_CHANNEL[ maxPATCH_INDEX[patch_index][m] ][patch_row_index][patch_c_index] = 0;
     }
   }
}



template<class  Tptr, class  Tdim, class  Tscal>
__global__ void fill_kernel( Tdim N, Tptr* data, Tscal value){

   for (Tdim i = threadIdx.x + blockIdx.x*blockDim.x; i < N; i += blockDim.x*gridDim.x)
      data[i] = value;

}


template<class Tptr, class Tdim, class Tscal>
void fill( Tdim N, Tptr* data, Tscal value){
   unsigned int nthreads = 256;
   unsigned int nblocks = (N+nthreads-1)/nthreads;
   if (nblocks > 65536) nblocks = 65536;
   fill_kernel<Tptr,Tdim,Tscal><<<nblocks,nthreads>>>(N,data,value);
   cudaDeviceSynchronize();
}


void fill( uint64_t N, float* data, float value){
   unsigned int nthreads = 256;
   unsigned int nblocks = (N+nthreads-1)/nthreads;
   if (nblocks > 65536) nblocks = 65536;
   fill_kernel<float,uint64_t,float><<<nblocks,nthreads>>>(N,data,value);
   cudaDeviceSynchronize();
}

void fill( uint64_t N, uint64_t* data, float value){
   unsigned int nthreads = 256;
   unsigned int nblocks = (N+nthreads-1)/nthreads;
   if (nblocks > 65536) nblocks = 65536;
   fill_kernel<uint64_t,uint64_t,float><<<nblocks,nthreads>>>(N,data,value);
   cudaDeviceSynchronize();
}


void fill( uint64_t N, int* data, int value){
   unsigned int nthreads = 256;
   unsigned int nblocks = (N+nthreads-1)/nthreads;
   if (nblocks > 65536) nblocks = 65536;
   fill_kernel<int,uint64_t,int><<<nblocks,nthreads>>>(N,data,value);
   cudaDeviceSynchronize();
}













