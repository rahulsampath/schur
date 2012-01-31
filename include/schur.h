
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

//S = Schur
//L = Low
//H = High
//O = Owned = S + V
//V = Volume or Interior (includes domain boundaries)
//MG = Multigrid (includes 0 dirichlet on both ends)
enum ListType {
  S, L, H, O, V, MG
};

void createRSDtree(RSDnode *& root, int rank, int npes);

void destroyRSDtree(RSDnode *root);

void createLowAndHighComms(LocalData* data);

void computeSchurDiag(LocalData* data);

//Uses MG ordering 
void mgSolve(LocalData* data, Vec rhs, Vec sol);

//Uses MG ordering 
void mgMatMult(LocalData* data, Vec in, Vec out);

//Uses S ordering
void schurMatVec(LocalData* data, bool isLow, Vec uSin, Vec uSout);

//Uses O ordering
void KmatVec(LocalData* data, RSDnode* root, Vec uIn, Vec uOut);

//Uses O ordering
void RSDapplyInverse(LocalData* data, RSDnode* root, Vec f, Vec u);

//This only allocates memory. The values may be junk. 
void createVector(LocalData* data, ListType type, Vec & v);

//This only sets the relevant values. It leaves the other values untouched.
void map(LocalData* data, ListType fromType, Vec fromVec, ListType toType, Vec toVec);

#endif



