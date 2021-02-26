

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>

extern "C" void * do_gpu_cudaMallocManaged(unsigned long int size){
   char *ptr;
   cudaMallocManaged(&ptr,size);
   printf("do_gpu_cudaMallocManaged ptr = %lu\n",(unsigned long int) ptr);

   return (void*) ptr;
}


extern "C" void * do_cpu_malloc(unsigned long int size){
   return  malloc(size);

}

extern "C" void * do_gpu_cudaHostAlloc(unsigned long int size){
   char *ptr;
   cudaHostAlloc(&ptr,size,cudaHostAllocPortable);
   printf("do_gpu_cudaHostAlloc ptr = %lu %g [GB]\n",(unsigned long int) ptr, (double) size/(1024.0*1024.0*1024.0)  );

   return (void*) ptr;
}


extern "C" int get_cuda_num_devices(){
   int num_gpus;
   cudaGetDeviceCount(&num_gpus); 
   return num_gpus;
}
