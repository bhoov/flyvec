#include <omp.h>
//for definition of uint64_t
#include <stdlib.h>
#include <cuda.h>
#include <stdio.h>

int MAXLEN = 214;

extern "C"
void prune_input_data(void *DATA_in, uint64_t Nrows, uint64_t Ncolumns);

/* remove repeated indices   */
extern "C"
void prune_input_data(void *DATA_in, uint64_t Nrows, uint64_t Ncolumns){


  int **DATA;
  DATA = new int*[Nrows];
  DATA[0] = (int*) DATA_in;

  for (uint64_t row = 1; row < Nrows; ++row)
      DATA[row] = DATA[0] + row * Ncolumns;


  uint64_t mid_point = Ncolumns/2;
  #pragma omp parallel for
  for (uint64_t row = 0; row < Nrows; ++row){

     int TMP[Ncolumns];
     TMP[0] = DATA[row][0];
     uint64_t counter=1;
     int mid_point_value = DATA[row][mid_point];

     for (uint64_t c = 1; c < Ncolumns; ++c){

        if (c == mid_point) {
           TMP[c] = -1;
           counter++;
           continue;
        }

        int val = DATA[row][c];
        uint64_t cc;
        for (cc = 0; cc < counter; ++cc)
          if (TMP[cc] == val) break;

        if (cc == counter) //true if a duplicate was not found
           TMP[c] = val;
        else
           TMP[c] = -1; //replace duplicates with -1;

        counter++;
     }

     TMP[mid_point] = mid_point_value;
     for (uint64_t c = 0; c < Ncolumns; ++c)
       DATA[row][c] = TMP[c];
  }
  delete[] DATA;
}


extern "C"
uint64_t getnum_samples( void* offsets_in, uint64_t Nsentences, uint64_t W){

  uint64_t* offsets = (uint64_t*) offsets_in;

  // fprintf(stderr,"getnum_samples: Nsentences = %llu, W = %llu, offsets[0] = %llu offsets[1] = %llu\n",Nsentences,W,offsets[0],offsets[1]);


  uint64_t count = 0;
  uint64_t count_phrases = 0;
  #pragma omp parallel for reduction(+:count,count_phrases)
  for (uint64_t s = 0; s < Nsentences; ++s){
     uint64_t length = offsets[s+1]-offsets[s];
     if ( (length >= W) && (length < MAXLEN) ) {
       count++;
       count_phrases += (length-W+1);
     }
  }
  fprintf(stderr,"%lu sentences with at lest %ld  words\n",count,W);
  fprintf(stderr,"%lu phrases with W %ld  words\n",count_phrases,W);
  return count_phrases;
}


extern "C"
void compute_offset_phrases( void *offset_phrases_in, void* offsets_sentences_in,  uint64_t Nsentences, uint64_t W){


  uint64_t nsamples = getnum_samples(offsets_sentences_in,Nsentences,W);
  uint64_t *offset_phrases = (uint64_t*) offset_phrases_in; //new uint64_t[nsamples];

  double t1 = omp_get_wtime();

  //uint64_t* offset_phrases = (uint64_t*) offset_phrases_in;
  uint64_t* offsets_sentences = (uint64_t*) offsets_sentences_in;

  fprintf(stderr,"offset_phrases = %p\n",offset_phrases);

  //offset_phrases = new unsigned long int[count_phrases+1];

  uint64_t global_count = 0;
  uint64_t cnt = 0;
  uint64_t o_p = 0;
  for (uint64_t s = 0; s < Nsentences; ++s){
     uint64_t length = offsets_sentences[s+1]-offsets_sentences[s];
     if ( (length >= W) && (length < MAXLEN) ){
       for (uint64_t i = 0; i < (length - W + 1); ++i){
          offset_phrases[cnt] = global_count + i;
          o_p = global_count + i;
          cnt++;
       }
     }
     global_count += length;
     //if ( (s+1)%1000000 == 0 ) fprintf(stderr,"compute_offset_phrases: s = %llu [of %llu] cnt=%llu \n",s+1, Nsentences,cnt);
  }
  double t2 = omp_get_wtime();
  fprintf(stderr,"compute_offset_phrases:  completed in %g [s]\n",t2-t1);
  //delete[] offset_phrases;

}
