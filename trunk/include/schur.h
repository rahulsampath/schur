
#ifndef __SCHUR__
#define __SCHUR__

#include "mpi.h"
#include "petscmat.h"
#include "petscdmmg.h"
#include <vector>

struct RSDnode {
  RSDnode* child;
  int rankForCurrLevel;
  int npesForCurrLevel;
};

struct LocalData {
  MPI_Comm commAll, commLow, commHigh;
  Mat Kssl, Kssh;
  Mat Ksl, Ksh;
  Mat Kls, Khs;
  Mat Kll, Khh;
  Vec diagS;
  DMMG* mgObj;
};

void createRSDtree(RSDnode *& root, int rank, int npes);

void destroyRSDtree(RSDnode *root);

void createLowAndHighComms(LocalData* data);

void computeSchurDiag(LocalData* data);

void mgSolve(LocalData* data, std::vector<double> & rhs, std::vector<double> & sol);

void schurMatVec(LocalData* data, bool isLow, Vec uSin, Vec uSout);

#endif



