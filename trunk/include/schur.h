
#ifndef __SCHUR__
#define __SCHUR__

#include "mpi.h"

struct RSDnode {
  RSDnode* child;
  int rankForCurrLevel;
  int npesForCurrLevel;
};

#endif



