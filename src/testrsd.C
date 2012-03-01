
#include <iostream>
#include "mpi.h"
#include "petsc.h"
#include "schur.h"

double stencil[4][4];

int main(int argc, char** argv) {
  PetscInitialize(&argc, &argv, 0, 0);

  computeStencil();

  OuterContext* ctx;
  createOuterContext(ctx);

  destroyOuterContext(ctx);

  std::cout<<"Done!"<<std::endl;
  PetscFinalize();
  return 0;
}


