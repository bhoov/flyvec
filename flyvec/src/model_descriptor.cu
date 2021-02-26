#include "model_descriptor.h"
#include <stdio.h>

//extern "C"
//void print_model_params(void* MODEL);


uint64_t model_descriptor::compute_num_patches(uint64_t IM_DIM, uint64_t Window_size, uint64_t Stride){
   uint64_t  result = 0;
   if (IM_DIM <= Window_size) return 1;

   for (uint64_t i = 0; i < IM_DIM-Window_size+1; i+=Stride)
     result++;
   return result;
}

extern "C"
void * model_create_descriptor(int num_gpus){
  fprintf(stderr,"model_create_descriptor:  num_gpus = %d\n",num_gpus);

  model_descriptor * DSCR = new model_descriptor[num_gpus];
  //set default parameters
  for (int gpu = 0; gpu < num_gpus; ++gpu){
    DSCR[gpu].IM_HEIGHT=32;  DSCR[gpu].IM_WIDTH=32;
    DSCR[gpu].Nchannels=3;
    DSCR[gpu].Ns_1 = 0;
    DSCR[gpu].W = 4;         DSCR[gpu].ST = 1;
    DSCR[gpu].hid = 100;     DSCR[gpu].Num = 100;
    DSCR[gpu].N = DSCR[gpu].W*DSCR[gpu].W*DSCR[gpu].Nchannels;
    DSCR[gpu].m = 2;         DSCR[gpu].p=2;
    DSCR[gpu].initial_learning_rate = 0.001;
    DSCR[gpu].delta = 0.0;
    DSCR[gpu].prec = 1.0e-30;
    DSCR[gpu].mu = 0.0;
    DSCR[gpu].sigma = 1.0;  
    DSCR[gpu].sparse_input = 0;
    DSCR[gpu].vocabulary_size = 0; 
    DSCR[gpu].Lmid = 1.0;
    DSCR[gpu].Lbase = 1.0;
    DSCR[gpu].frequency_scaling = 0;

    DSCR[gpu].HEIGHT_W_ST_block = DSCR[gpu].compute_num_patches(DSCR[gpu].IM_HEIGHT, DSCR[gpu].W, DSCR[gpu].ST);
    DSCR[gpu].WIDTH_W_ST_block = DSCR[gpu].compute_num_patches(DSCR[gpu].IM_WIDTH, DSCR[gpu].W, DSCR[gpu].ST);
    DSCR[gpu].Ns = DSCR[gpu].HEIGHT_W_ST_block * DSCR[gpu].WIDTH_W_ST_block * DSCR[gpu].Ns_1; 
  }

  return (void*) DSCR;
}

extern "C"
void set_model_param_int(uint64_t value, const char *s, void* MODEL){

   model_descriptor *M = (model_descriptor*)  MODEL;
   printf("set_param_int: setting parameter %s\n",s);

   if (0==strcmp(s,"IM_HEIGHT")) M[0].IM_HEIGHT = value;
   else if (0==strcmp(s,"IM_WIDTH")) M[0].IM_WIDTH = value;
   else if (0==strcmp(s,"Nchannels")) M[0].Nchannels = value;
   else if (0==strcmp(s,"Ns_1")) M[0].Ns_1 = value;
   else if (0==strcmp(s,"W")) M[0].W = value;
   else if (0==strcmp(s,"ST")) M[0].ST = value;
   else if (0==strcmp(s,"hid")) M[0].hid = value;
   else if (0==strcmp(s,"Num")) M[0].Num = value;
   else if (0==strcmp(s,"p")) M[0].p = value;
   else if (0==strcmp(s,"m")) M[0].m = value;
   else if (0==strcmp(s,"vocabulary_size")) M[0].vocabulary_size = value;
   else if (0==strcmp(s,"sparse_input")) M[0].sparse_input = value; 
   else if (0==strcmp(s,"frequency_scaling")) M[0].frequency_scaling = value;
   else printf("set_param_int: unknown parameter %s\n",s);
}


extern "C"
void set_model_param_float(float value, const char *s, void* MODEL){

   model_descriptor *M = (model_descriptor*)  MODEL;
   printf("set_param_float: setting parameter %s\n",s);

   if (0==strcmp(s,"initial_learning_rate")) M[0].initial_learning_rate = value;
   else if (0==strcmp(s,"delta")) M[0].delta = value;
   else if (0==strcmp(s,"prec")) M[0].prec = value;
   else if (0==strcmp(s,"mu")) M[0].mu = value;
   else if (0==strcmp(s,"sigma")) M[0].sigma = value;
   else if (0==strcmp(s,"Lmid")) M[0].Lmid = value;
   else if (0==strcmp(s,"Lbase")) M[0].Lbase = value;
   else printf("set_param_float: unknown parameter %s\n",s);
}


extern "C"
void compute_model_derived_parameters(void* MODEL){
   model_descriptor *M = (model_descriptor*)  MODEL;

   M[0].HEIGHT_W_ST_block = M[0].compute_num_patches(M[0].IM_HEIGHT, M[0].W, M[0].ST);
   M[0].WIDTH_W_ST_block = M[0].compute_num_patches(M[0].IM_WIDTH, M[0].W, M[0].ST);
   M[0].Ns = M[0].HEIGHT_W_ST_block * M[0].WIDTH_W_ST_block * M[0].Ns_1;

   if (M[0].sparse_input != 0)
     M[0].N = 2*M[0].vocabulary_size;
   else
     M[0].N = M[0].W * M[0].W * M[0].Nchannels;
    

   curandCreateGenerator(&M[0].cu_rand_gen_handle, curandRngType_t::CURAND_RNG_QUASI_SCRAMBLED_SOBOL64);
   fprintf(stderr,"file=%s line=%d\n",__FILE__, __LINE__);
   curandSetGeneratorOffset (M[0].cu_rand_gen_handle, rand());
   fprintf(stderr,"file=%s line=%d\n",__FILE__, __LINE__);

   curandSetQuasiRandomGeneratorDimensions(M[0].cu_rand_gen_handle, M[0].Ns_1);

   fprintf(stderr,"curandSetQuasiRandomGeneratorDimensions - done, M[0].Ns_1=%lu\n",M[0].Ns_1);
}

extern "C"
void copy_model_params(void* MODEL, int source, int destination){

   model_descriptor *M = (model_descriptor*)  MODEL;
   M[destination].IM_HEIGHT = M[source].IM_HEIGHT;
   M[destination].IM_WIDTH = M[source].IM_WIDTH;
   M[destination].Nchannels = M[source].Nchannels;
   M[destination].Ns_1 = M[source].Ns_1;
   M[destination].W = M[source].W;
   M[destination].ST = M[source].ST;
   M[destination].hid = M[source].hid;
   M[destination].Num = M[source].Num;
   M[destination].m = M[source].m;
   M[destination].p = M[source].p;
   M[destination].sparse_input = M[source].sparse_input;
   M[destination].vocabulary_size = M[source].vocabulary_size;
   M[destination].initial_learning_rate = M[source].initial_learning_rate;
   M[destination].delta = M[source].delta;
   M[destination].prec = M[source].prec;
   M[destination].mu = M[source].mu;
   M[destination].sigma = M[source].sigma;
   M[destination].HEIGHT_W_ST_block = M[source].HEIGHT_W_ST_block;
   M[destination].WIDTH_W_ST_block = M[source].WIDTH_W_ST_block; 
   M[destination].Ns = M[source].Ns;
   M[destination].N = M[source].N;   
   M[destination].Lmid = M[source].Lmid;
   M[destination].Lbase = M[source].Lbase;
   M[destination].frequency_scaling = M[source].frequency_scaling;
}

extern "C"
void print_model_params(void* MODEL){

   model_descriptor *M = (model_descriptor*)  MODEL;
   printf("print_model_params: MODEL = %p\n",MODEL);

   printf("print_model_params: M = %p\n",M);

   printf(" IM_HEIGHT = %lu \n",M[0].IM_HEIGHT);
   printf(" IM_WIDTH = %lu \n",M[0].IM_WIDTH);
   printf(" Nchannels = %lu \n",M[0].Nchannels);
   printf(" Ns_1 = %lu \n",M[0].Ns_1);
   printf(" W = %lu \n",M[0].W);
   printf(" ST = %lu \n",M[0].ST);
   printf(" hid = %lu \n",M[0].hid );
   printf(" Num = %lu \n",M[0].Num );
   printf(" m = %lu \n",M[0].m);
   printf(" p = %lu \n",M[0].p);
   printf(" sparse_input = %d \n",M[0].sparse_input);
   printf(" vocabulary_size = %d \n",M[0].vocabulary_size);
   printf(" frequency_scaling = %lu \n",M[0].frequency_scaling);

   printf(" initial_learning_rate = %f \n",M[0].initial_learning_rate);
   printf(" delta = %g \n",M[0].delta);
   printf(" prec  = %g \n",M[0].prec);
   printf(" mu    = %g \n",M[0].mu);
   printf(" sigma = %g \n",M[0].sigma);
   printf(" Lmid  = %g \n",M[0].Lmid);
   printf(" Lbase = %g \n",M[0].Lbase);

   //derived variables
   printf(" HEIGHT_W_ST_block = %lu \n",M[0].HEIGHT_W_ST_block);
   printf(" WIDTH_W_ST_block = %lu \n",M[0].WIDTH_W_ST_block);
   printf(" Ns = %lu \n",M[0].Ns);
   printf(" N  = %lu \n",M[0].N);

}




