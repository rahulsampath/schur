
#ifndef __SCHUR__
#define __SCHUR__

#include "mpi.h"
#include "petscmat.h"

struct RSDnode {
  int childNumber;
  MPI_Comm comm;
  RSDnode* child1;
  RSDnode* child2;
  Mat k11, k22, k33;
  Mat k13, k31;
  Mat k23, k32; 
};

#endif



