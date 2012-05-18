
#include "mpi.h"
#include <iostream>
#include <cstdlib>
#include "petsc.h"
#include "schur.h"

double stencil[4][4];

PetscLogEvent outerKspEvent;
PetscCookie rsdCookie;

int main(int argc, char** argv) {
  PetscInitialize(&argc, &argv, "options", PETSC_NULL);

  PetscCookieRegister("RSD", &rsdCookie);
  PetscLogEventRegister("OuterKsp", rsdCookie, &outerKspEvent);

  int rank, npes;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &npes);

  MPI_Barrier(PETSC_COMM_WORLD);
  if(!rank) {
    std::cout<<"Npes = "<<npes<<std::endl;
  } 
  MPI_Barrier(PETSC_COMM_WORLD);

  int G = 1;
  PetscOptionsGetInt(PETSC_NULL, "-inner_ksp_max_it", &G, PETSC_NULL);
  if(!rank) {
    std::cout<<"G = "<<G<<std::endl;
  }

  computeStencil();

  OuterContext* ctx;
  createOuterContext(ctx);

  const unsigned int seed = (0x3456782  + (54763*rank));

  PetscRandom rndCtx;
  PetscRandomCreate(PETSC_COMM_WORLD, &rndCtx);
  PetscRandomSetType(rndCtx, PETSCRAND48);
  PetscRandomSetSeed(rndCtx, seed);
  PetscRandomSeed(rndCtx);

  VecSetRandom(ctx->outerSol, rndCtx);

  PetscRandomDestroy(rndCtx);

  zeroBoundary(ctx->data, ctx->outerSol);
  MatMult(ctx->outerMat, ctx->outerSol, ctx->outerRhs);

  VecZeroEntries(ctx->outerSol);

  PetscLogEventBegin(outerKspEvent, 0, 0, 0, 0);

  KSPSolve(ctx->outerKsp, ctx->outerRhs, ctx->outerSol);

  PetscLogEventEnd(outerKspEvent, 0, 0, 0, 0);

  MPI_Barrier(PETSC_COMM_WORLD);
  if(!rank) {
    std::cout<<"Done Solve"<<std::endl<<std::endl;
  }
  MPI_Barrier(PETSC_COMM_WORLD);

  destroyOuterContext(ctx);

  MPI_Barrier(PETSC_COMM_WORLD);
  if(!rank) {
    std::cout<<"Destroyed Outer Context"<<std::endl<<std::endl;
  }
  MPI_Barrier(PETSC_COMM_WORLD);

  MPI_Barrier(PETSC_COMM_WORLD);
  if(!rank) {
    std::cout<<"Done!"<<std::endl<<std::endl;
  }
  MPI_Barrier(PETSC_COMM_WORLD);

  PetscFinalize();

  return 0;
}


