
#ifndef __SCHUR__
#define __SCHUR__

#include "mpi.h"
#include "petscmat.h"
#include "petscdmmg.h"

struct RSDnode {
  RSDnode* child;
  int rankForCurrLevel;
  int npesForCurrLevel;
};

struct LocalData {
  MPI_Comm commAll;
  MPI_Comm commLow, commHigh;
  Mat Kssl, Kssh;
  Mat Ksl, Ksh;
  Mat Kls, Khs;
  DMMG* mgObj;
};

void createRSDtree(RSDnode *& root, int rank, int npes);

void createLowAndHighComms(LocalData* data);

#endif



