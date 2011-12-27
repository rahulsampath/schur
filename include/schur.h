
#ifndef __SCHUR__
#define __SCHUR__

#include "mpi.h"
#include "petscmat.h"

struct RSDnode {
  MPI_Comm comm;
  RSDnode* child1;
  RSDnode* child2;
  Mat k11, k13;
  Mat k22, k23;
  Mat k31, k32, k33;
};


#endif



