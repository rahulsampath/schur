
#ifndef __SCHUR__
#define __SCHUR__

#include "mpi.h"
#include "petscmat.h"
#include "petscdmmg.h"
#include <vector>
#include <cassert>

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
  int N;
  Vec diagS; 
  DMMG* mgObj;
};

//MG = Multigrid (includes 0 dirichlet on both ends)
//O = Owned = S + V
//V = Volume or Interior (includes domain boundaries)
//L = Low
//H = High
//S = Schur
enum ListType {
  MG, O, L, H, S 
};

void createRSDtree(RSDnode *& root, int rank, int npes);

void destroyRSDtree(RSDnode *root);

void createLowAndHighComms(LocalData* data);

void computeSchurDiag(LocalData* data);

//Uses S ordering
void schurMatVec(LocalData* data, bool isLow, Vec uSin, Vec uSout);

//Uses S ordering
void schurSolve(LocalData* data, bool isLow, Vec rhs, Vec sol);

//Uses O ordering
void KmatVec(LocalData* data, RSDnode* root, Vec uIn, Vec uOut);

//Uses O ordering
void RSDapplyInverse(LocalData* data, RSDnode* root, Vec f, Vec u);

//This only sets the relevant values. It leaves the other values untouched.
template<ListType fromType, ListType toType>
inline void map(LocalData* data, Vec fromVec, Vec toVec);

template<>
inline void map<O, S>(LocalData* data, Vec fromVec, Vec toVec) {
  assert(false);
}

template<>
inline void map<S, O>(LocalData* data, Vec fromVec, Vec toVec) {
  assert(false);
}

template<>
inline void map<L, O>(LocalData* data, Vec fromVec, Vec toVec) {
  assert(false);
}

template<>
inline void map<O, L>(LocalData* data, Vec fromVec, Vec toVec) {
  assert(false);
}

template<>
inline void map<H, O>(LocalData* data, Vec fromVec, Vec toVec) {
  assert(false);
}

template<>
inline void map<O, H>(LocalData* data, Vec fromVec, Vec toVec) {
  assert(false);
}

template<>
inline void map<MG, L>(LocalData* data, Vec fromVec, Vec toVec) {
  assert(false);
}

template<>
inline void map<L, MG>(LocalData* data, Vec fromVec, Vec toVec) {
  assert(false);
}

template<>
inline void map<MG, H>(LocalData* data, Vec fromVec, Vec toVec) {
  assert(false);
}

template<>
inline void map<H, MG>(LocalData* data, Vec fromVec, Vec toVec) {
  assert(false);
}

template<>
inline void map<MG, O>(LocalData* data, Vec fromVec, Vec toVec) {
  int rank, npes;
  MPI_Comm_rank(data->commAll, &rank);
  MPI_Comm_size(data->commAll, &npes);

  DA da = DMMGGetDA(data->mgObj);

  PetscScalar** mgArr;
  PetscScalar* oArr;

  VecGetArray(toVec, &oArr);
  DAVecGetArray(da, fromVec, &mgArr);

  int oxs, onx;
  if(rank == 0) {
    oxs = 0;
    onx = data->N;
  } else {
    oxs = 1;
    onx = (data->N) - 1;
  }

  int vnx;
  if(rank == (npes - 1)) {
    vnx = onx;
  } else {
    vnx = onx - 1;
  }

  for(int yi = 0; yi < (data->N); ++yi) {
    for(int xi = oxs; xi < (oxs + vnx); ++xi) {
      oArr[(yi*onx) + (xi - oxs)] = mgArr[yi][xi];
    }//end for xi
  }//end for yi

  DAVecRestoreArray(da, fromVec, &mgArr);
  VecRestoreArray(toVec, &oArr);
}

template<>
inline void map<O, MG>(LocalData* data, Vec fromVec, Vec toVec) {
  int rank, npes;
  MPI_Comm_rank(data->commAll, &rank);
  MPI_Comm_size(data->commAll, &npes);

  DA da = DMMGGetDA(data->mgObj);

  PetscScalar** mgArr;
  PetscScalar* oArr;

  VecGetArray(fromVec, &oArr);
  DAVecGetArray(da, toVec, &mgArr);

  int oxs, onx;
  if(rank == 0) {
    oxs = 0;
    onx = data->N;
  } else {
    oxs = 1;
    onx = (data->N) - 1;
  }

  int vnx;
  if(rank == (npes - 1)) {
    vnx = onx;
  } else {
    vnx = onx - 1;
  }

  for(int yi = 0; yi < (data->N); ++yi) {
    for(int xi = oxs; xi < (oxs + vnx); ++xi) {
      mgArr[yi][xi] = oArr[(yi*onx) + (xi - oxs)];
    }//end for xi
  }//end for yi

  DAVecRestoreArray(da, toVec, &mgArr);
  VecRestoreArray(fromVec, &oArr);
}


#endif




