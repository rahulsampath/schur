
#ifndef __SCHUR__
#define __SCHUR__

#include "mpi.h"

struct RSDnode {
  MPI_Comm comm;
  RSDnode* child;
};

#endif



