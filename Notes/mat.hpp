#include <iostream>
#include <stdlib.h>
#include <math.h>

class mat{
  public:
    int * arr;
    int size;
    int * width;
    int dim;
    int layerDim;
    mat * lower;

  // Create matrix
  mat(int in_dim, int * in_width){
    layerDim = in_dim;
    *width = *in_width;

    if(layerDim > 1){
      lower = (mat*)malloc(sizeof(mat)*width[layerDim]); 
      for(int i = 0; i < width[layerDim]; i++){
        lower[i] = mat(layerDim-1, width+1);
      }
    } else {
      
    }




    // // create an array big enough for the whole matrix.
    // for(int i = 0; i < dim; i++){
    //   size += width[i];
    // }
    
  }

};