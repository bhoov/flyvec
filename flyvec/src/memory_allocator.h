#ifndef MEMORY_ALLOCATOR_H
#define MEMORY_ALLOCATOR_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>


//int posix_memalign(void **memptr, size_t alignment, size_t size);

template <class Tptr, class Tdim>
Tptr*  mem_alloc1D_cuda_managed (Tdim dim1) {
    
    if (dim1 == 0) return NULL;

    Tptr* data = NULL; 
    if ( cudaSuccess != cudaMallocManaged(&data,dim1*sizeof(Tptr*), cudaMemAttachGlobal) )
       fprintf(stderr,"ERROR: mem_alloc1D_cuda_managed, memory allocation failed\n");
    return data;
} 

template <class Tptr>
void mem_free1D_cuda_managed(Tptr* ptr){
   cudaFree( (void*) ptr);
}


 
template <class Tptr, class Tdim>
Tptr**  mem_alloc2D_cuda_managed (Tdim dim1, Tdim dim2) {
    
    if (dim1*dim2 == 0) return NULL;

    Tptr** data = NULL; 
    data = mem_alloc1D_cuda_managed<Tptr*,Tdim>(dim1);
    data[0] = mem_alloc1D_cuda_managed<Tptr,Tdim>(dim1*dim2); 
    for (Tdim i = 1; i < dim1; ++i)
        data[i] = data[0] + i*dim2;

    return data;
} 

template <class Tptr>
void mem_free2D_cuda_managed(Tptr** ptr){
   cudaFree( (void*) ptr[0]);
   cudaFree( (void*) ptr);
}


template <class Tptr, class Tdim>
Tptr*  mem_alloc1D_cuda (Tdim dim1) {
    
    if (dim1 == 0) return NULL;

    Tptr* data = NULL; 
    if ( cudaSuccess != cudaMalloc(&data,dim1*sizeof(Tptr*)) )
       fprintf(stderr,"ERROR: mem_alloc1D_cuda, memory allocation failed\n");
    
    return data;
} 

template <class Tptr>
void mem_free1D_cuda(Tptr* ptr){
   cudaFree( (void*) ptr);
}

template <class Tptr, class Tdim>
Tptr*  mem_alloc1D (Tdim dim1) {

    if (dim1 == 0) return NULL;

    Tptr* data = NULL;
    if (0 != posix_memalign( (void**) &data, 128, dim1*sizeof(Tptr)))
       fprintf(stderr,"ERROR: mem_alloc1D, memory allocation failed\n");
    return data;
}


template <class Tptr, class Tdim>
Tptr**  mem_alloc2D (Tdim dim1, Tdim dim2) {

    if (dim1*dim2 == 0) return NULL;

    Tptr** data = NULL;
    data = mem_alloc1D<Tptr*,Tdim>(dim1);
    data[0] = mem_alloc1D<Tptr,Tdim>(dim1*dim2);
    for (Tdim i = 1; i < dim1; ++i)
        data[i] = data[0] + i*dim2;

    return data;
}

template <class Tptr, class Tdim>
Tptr*  mem_alloc1D_cuda_host_registered (Tdim dim1) {

    if (dim1 == 0) return NULL;
    fprintf(stderr,"mem_alloc1D_cuda_host_registered:  allocating %g [GB]\n",(double) dim1*sizeof(Tptr)/(1024.0*1024.0*1024.0));


    Tptr* data = NULL;
    if ( cudaSuccess != cudaHostAlloc(&data, dim1*sizeof(Tptr*), cudaHostAllocPortable) )
       fprintf(stderr,"ERROR: mem_alloc1D_cuda_host_registered, memory allocation failed\n");
    return data;
}

template <class Tptr>
void mem_free1D_host_registered(Tptr* ptr){
   cudaFree( (void*) ptr);
}





#endif
