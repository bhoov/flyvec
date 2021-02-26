#include <stdio.h>
#include <omp.h>


int main(){

  int *list_to_remove;
  int *data;
  char buf[512];
  size_t list_to_remove_size = 419;
  size_t d_size = 5731746159;

  list_to_remove = new int[list_to_remove_size];
  data = new int[d_size];

  FILE *pFile = fopen("/gpfs/wscgpfs02/hebml/OpenWeb/phrases3/encodings.npy","r");
  fgets (buf , 512 , pFile);
  fread(data,sizeof(int),d_size,pFile); 
  fclose(pFile);
  
  pFile = fopen("/gpfs/wscgpfs02/hebml/OpenWeb/phrases3/exceptional_ids_terrier_stop_phrases_VOC_25000.npy","r");
  fgets (buf , 512 , pFile);
  fread(list_to_remove,sizeof(int),list_to_remove_size,pFile);
  fclose(pFile);

  

  for (int i=0; i < list_to_remove_size; ++i){

    int indx = list_to_remove[i];
    printf("i = %d: removing %d\n",i,indx);

    #pragma omp parallel for
    for (size_t w = 0; w < d_size; ++w)
      if (data[w] == indx) data[w] = -1; 


  }

  pFile = fopen("new2_encodings.bin","w");
  fwrite(data,sizeof(int),d_size,pFile);
  fclose(pFile);


  delete[] data;
  delete[] list_to_remove; 

  return 0;
}


