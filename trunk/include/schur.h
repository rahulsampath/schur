
#ifndef __SCHUR__
#define __SCHUR__

#include "mpi.h"
#include "petscmat.h"

struct RSDnode {
  MPI_Comm comm;
  RSDnode* child1;
  RSDnode* child2;
};

#endif



