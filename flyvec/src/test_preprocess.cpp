#include <stdio.h>
#include <omp.h>

int main(){

  FILE *pFile;
  char buf[1024];
  unsigned long int * offsets;
  int Nsentences = 318910741;
  offsets = new unsigned long int[Nsentences+1];

  long int W = 5;

  pFile = fopen("/gpfs/wscgpfs02/lgrinbe/DIMA/LEV_TOLSTOY/DATA/OpenWebText_offsets.npy","r");
  //ignore the first line
  fgets ( buf, 1023, pFile);
  fread (offsets, sizeof(unsigned long int),Nsentences+1, pFile);
  fclose(pFile);


  fprintf(stderr,"offsets[0] = %ld \n",offsets[0]);
  fprintf(stderr,"offsets[Nsentences] = %ld \n",offsets[Nsentences]);


  //count number of sentences with at least W  words:
  unsigned long int count = 0;
  unsigned long int count_phrases = 0;
  //#pragma omp parallel for reduction(+:count,count_phrases)
  for (unsigned long int s = 0; s < Nsentences; ++s){
     unsigned  long int length = offsets[s+1]-offsets[s];
     if (length > 5600) fprintf(stderr,"s = %lu, length = %lu\n",s,length);
     if (length >= W) {
       count++;
       count_phrases += (length-W+1);
     }
  }
  fprintf(stderr,"%lu sentences with at lest %ld  words\n",count,W);
  fprintf(stderr,"%lu phrases with W %ld  words\n",count_phrases,W);

  //FILE *file_offsets = fopen("offset_phrases.dat","w");
  unsigned long int *offset_phrases;
  offset_phrases = new unsigned long int[/*count_phrases*/200+1];
  fprintf(stderr,"offset_phrases = %p\n",offset_phrases);


  size_t global_count = 0;
  size_t cnt = 0;
  for (size_t s = 0; s < Nsentences; ++s){
     unsigned long int length = offsets[s+1]-offsets[s];
//     if (length > 300) fprintf(stderr,"s = %lu, length = %lu\n",s, length);
     if (length >= W){ //go to the next sentence 
       for (size_t i = 0; i < (length - W + 1); ++i){
          offset_phrases[cnt%200] = /*global_count +*/ i; 
          //fprintf(file_offsets,"%lu\n",global_count + i);
          cnt++;
         if (cnt == 0) fprintf(stderr,"cnt == 0");
       }
     }
     global_count += length;
     if ( (s+1)%1000000 == 0 ) fprintf(stderr,"compute_offset_phrases: s = %lu [of %lu] cnt=%lu global_count = %lu \n",s+1, Nsentences,cnt,global_count);

  }
  //fclose(file_offsets);


  FILE *input_data_on_disk_encoding = fopen("/gpfs/wscgpfs02/lgrinbe/DIMA/LEV_TOLSTOY/DATA/OpenWebText_encodings.npy","r");
  unsigned long int Nwords = 8068819596;
  int *input_data_encoding = new int[Nwords];

  fgets (buf, 1023, input_data_on_disk_encoding);
  fread (input_data_encoding, sizeof(int),Nwords, input_data_on_disk_encoding);
  fclose(input_data_on_disk_encoding);


  int max_val = -1;
  //find max val in 
  for (size_t s = 0; s < Nwords; ++s){
     if (max_val < input_data_encoding[s]) max_val = input_data_encoding[s];
  }
  fprintf(stderr,"max_val = %d\n",max_val);



  delete[] input_data_encoding;
  delete[] offsets;

  return 0;
}

