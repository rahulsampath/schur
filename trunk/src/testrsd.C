
#include <iostream>
#include "mpi.h"
#include "petsc.h"
#include "schur.h"

int main(int argc, char** argv) {
  PetscInitialize(&argc, &argv, 0, 0);

  std::cout<<"Done!"<<std::endl;
  PetscFinalize();
  return 0;
}


