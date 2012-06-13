
#ifndef __SCHUR__
#define __SCHUR__

#include "mpi.h"
#include "petscksp.h"
#include "petscmat.h"
#include "petscdmmg.h"
#include <vector>
#include <cassert>

struct RSDnode {
  RSDnode* child;
  int rankForCurrLevel;
  int npesForCurrLevel;
};

struct VecBufType1 {
  Vec inSeq;
  Vec outSeq;
};

struct VecBufType2 {
  Vec rhsKspL;
  Vec rhsKspH;
  Vec solKspL;
  Vec solKspH;
};

struct VecBufType3 {
  Vec uL;
  Vec uH;
  Vec vL;
  Vec vH;
  Vec wL;
  Vec wH;
  Vec wSl;
  Vec wSh;
  Vec uStarL;
  Vec uStarH;
  Vec uSinCopy;
};

struct VecBufType4 {
  Vec uSout;
  Vec uSl;
  Vec uSh;
  Vec wSl;
  Vec wSh;
  Vec uL;
  Vec uH;
  Vec bSl;
  Vec bSh;
  Vec ySl;
  Vec ySh;
  Vec cL;
  Vec cH;
  Vec cOl;
  Vec cOh;
};

struct VecBufType5 {
  Vec fStarHcopy;
  Vec gS;
  Vec fTmpL;
  Vec fTmpH;
  Vec fL;
  Vec fH;
  Vec fStarL;
  Vec fStarH;
  Vec uSl;
  Vec uSh;
  Vec gL;
  Vec gH;
};

struct LocalData {
  int N;
  int dofsPerNode;
  MPI_Comm commAll, commLow, commHigh;
  Mat Kssl, Kssh;
  Mat Ksl, Ksh;
  Mat Kls, Khs;
  Mat Kll, Khh;
  Mat lowSchurMat, highSchurMat;
  KSP lowSchurKsp, highSchurKsp;
  DMMG* mgObj;
  VecBufType1* buf1;
  VecBufType1* buf2; 
  VecBufType1* buf3; 
  VecBufType2* buf4;
  VecBufType3* buf5;
  VecBufType4* buf6;
  VecBufType5* buf7;
};

struct OuterContext {
  LocalData* data;
  RSDnode* root;
  Mat outerMat;
  KSP outerKsp;
  PC outerPC;
  Vec outerSol;
  Vec outerRhs;
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

void createLinearElasticMechanicsStencil();

void createConvectionDiffusionStencil2();

void createConvectionDiffusionStencil();

void createPoissonStencil();

void createHardStencil(const double epsilon, const double kappa);

void createHardStencilType1();

void createHardStencilType2();

void destroyStencil();

double Phi(int i, double psi, double eta);

double dPhidPsi(int i, double eta);

double dPhidEta(int i, double psi);

void zeroBoundary(LocalData* data, Vec vec);

void createOuterContext(OuterContext* & ctx);

void destroyOuterContext(OuterContext* ctx);

void createLocalData(LocalData* & data);

void destroyLocalData(LocalData* data);

void createLocalMatrices(LocalData* data);

void createOuterMat(OuterContext* ctx);

PetscErrorCode outerMatMult(Mat mat, Vec in, Vec out);

void createSchurMat(LocalData* data);

PetscErrorCode lowSchurMatMult(Mat mat, Vec in, Vec out);

PetscErrorCode highSchurMatMult(Mat mat, Vec in, Vec out);

void createMG(LocalData* data);

PetscErrorCode computeMGmatrix(DMMG dmmg, Mat J, Mat B);

void createOuterPC(OuterContext* ctx); 

PetscErrorCode outerPCapply(void* ctx, Vec in, Vec out);

void createOuterKsp(OuterContext* ctx);

void createInnerKsp(LocalData* data);

void createRSDtree(RSDnode *& root, int rank, int npes);

void destroyRSDtree(RSDnode *root);

void createLowAndHighComms(LocalData* data);

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

#endif




