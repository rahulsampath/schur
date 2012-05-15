
#include "mpi.h"
#include <iostream>
#include <cstdlib>
#include "petsc.h"
#include "schur.h"

double stencil[4][4];

int main(int argc, char** argv) {
  int debugWait = 0;

  if(argc > 1) {
    debugWait = atoi(argv[1]);
  }

  while (debugWait);

  MPI_Init(&argc, &argv);

  int rank, npes;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &npes);

  MPI_Barrier(MPI_COMM_WORLD);
  if(!rank) {
    std::cout<<"Npes = "<<npes<<std::endl;
  } 
  MPI_Barrier(MPI_COMM_WORLD);

  PETSC_COMM_WORLD = MPI_COMM_WORLD;
  PetscInitialize(&argc, &argv, "options", PETSC_NULL);

  int G = 1;
  PetscOptionsGetInt(PETSC_NULL, "-inner_ksp_max_it", &G, PETSC_NULL);
  if(!rank) {
    std::cout<<"G = "<<G<<std::endl;
  }

  computeStencil();

  OuterContext* ctx;
  createOuterContext(ctx);

  VecSetRandom(ctx->outerSol, PETSC_NULL);
  zeroBoundary(ctx->data, ctx->outerSol);
  MatMult(ctx->outerMat, ctx->outerSol, ctx->outerRhs);

  VecZeroEntries(ctx->outerSol);
  KSPSolve(ctx->outerKsp, ctx->outerRhs, ctx->outerSol);

  MPI_Barrier(MPI_COMM_WORLD);
  if(!rank) {
    std::cout<<"Done Solve"<<std::endl<<std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  destroyOuterContext(ctx);

  MPI_Barrier(MPI_COMM_WORLD);
  if(!rank) {
    std::cout<<"Destroyed Outer Context"<<std::endl<<std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  PetscFinalize();

  MPI_Barrier(MPI_COMM_WORLD);
  if(!rank) {
    std::cout<<"Finalized Petsc"<<std::endl<<std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Finalize();

  return 0;
}


