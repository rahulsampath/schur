
#include <iostream>
#include <cstdlib>
#include "mpi.h"
#include "petsc.h"
#include "schur.h"

double stencil[4][4];

int main(int argc, char** argv) {
  int debugWait = 0;

  if(argc > 1) {
    debugWait = atoi(argv[1]);
  }

  while (debugWait);

  PetscInitialize(&argc, &argv, "options", PETSC_NULL);

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  computeStencil();

  OuterContext* ctx;
  createOuterContext(ctx);

  VecSetRandom(ctx->outerSol, PETSC_NULL);
  zeroBoundary(ctx->data, ctx->outerSol);
  MatMult(ctx->outerMat, ctx->outerSol, ctx->outerRhs);

  VecZeroEntries(ctx->outerSol);
  KSPSolve(ctx->outerKsp, ctx->outerRhs, ctx->outerSol);

  MPI_Barrier(PETSC_COMM_WORLD);
  if(!rank) {
    std::cout<<"Done Solve"<<std::endl;
  }

  destroyOuterContext(ctx);

  MPI_Barrier(PETSC_COMM_WORLD);
  if(!rank) {
    std::cout<<"End Program"<<std::endl;
  }

  PetscFinalize();
  return 0;
}


