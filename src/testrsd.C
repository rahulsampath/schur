
#include <iostream>
#include <cstdlib>
#include "mpi.h"
#include "petsc.h"
#include "petscsys.h"
#include "schur.h"

double stencil[4][4];
PetscViewer viewer;

int main(int argc, char** argv) {
  int debugWait = 0;

  if(argc > 1) {
    debugWait = atoi(argv[1]);
  }

  while (debugWait);

  PetscInitialize(&argc, &argv, "options", PETSC_NULL);

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  char outFile[200];
  sprintf(outFile, "rsdOut_%d.txt", rank);
  PetscViewerASCIIOpen(PETSC_COMM_SELF, outFile, &viewer);

  computeStencil();

  OuterContext* ctx;
  createOuterContext(ctx);

  VecSetRandom(ctx->outerSol, PETSC_NULL);
  zeroBoundary(ctx->data, ctx->outerSol);
  MatMult(ctx->outerMat, ctx->outerSol, ctx->outerRhs);

  VecZeroEntries(ctx->outerSol);
  KSPSolve(ctx->outerKsp, ctx->outerRhs, ctx->outerSol);

  destroyOuterContext(ctx);

  PetscViewerDestroy(viewer);

  PetscFinalize();
  return 0;
}


