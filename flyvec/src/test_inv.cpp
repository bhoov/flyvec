#include <stdio.h>
#include <omp.h>

int main(){


  //FILE *pFile=fopen("/gpfs/wscgpfs02/hebml/OpenWeb/OWT_20000voc_STOP_REMOVED_terrier.npy","r");
  FILE *pFile=fopen("/gpfs/wscgpfs02/hebml/OpenWeb/gensim_ascii_encodings/encodings.npy","r");

  int vocabulary_size = 21854;
  size_t Nwords = 5785895233;
  int *input = new int[Nwords];
   
  float *word_inv_frequency = new float[vocabulary_size];


  char buf[512];
  fgets (buf , 512 , pFile);
  fread(input,sizeof(int),Nwords,pFile);
  fclose(pFile);


  for (size_t i = 0; i < vocabulary_size; ++i)
   word_inv_frequency[i] = 0.0;


  for (size_t i = 0; i < Nwords; ++i){
    if (input[i] >= 0)
      word_inv_frequency[ input[i] ] += 1.0;
  }

  pFile=fopen("counts.dat","w");
  for (size_t i = 0; i < vocabulary_size; ++i)
      fprintf(pFile,"%f\n",word_inv_frequency[i]);
  fclose(pFile);


  for (size_t i = 0; i < vocabulary_size; ++i){
    if (word_inv_frequency[i] != 0)
      word_inv_frequency[i] = 1.0/word_inv_frequency[i];
    else
      word_inv_frequency[i] = 0.0;
  }

  pFile=fopen("word_inv_frequency.dat","w");
  for (size_t i = 0; i < vocabulary_size; ++i)
      fprintf(pFile,"%f\n",word_inv_frequency[i]);
  fclose(pFile);

  delete[] input;
  delete[] word_inv_frequency;

  return 0;
}








