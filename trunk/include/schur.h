
#ifndef __SCHUR__
#define __SCHUR__

#include "mpi.h"

struct RSDnode {
  RSDnode* child;
  int rankForCurrLevel;
  int npesForCurrLevel;
};

void createRSDtree(RSDnode *& root, int rank, int npes);

#endif



