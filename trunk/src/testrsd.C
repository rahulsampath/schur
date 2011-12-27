
#include <stdio.h>
#include "mpi.h"
#include "petsc.h"

int main(int argc, char** argv) {
  PetscInitialize(&argc, &argv, 0, 0);

  printf("Done!\n");
  PetscFinalize();
  return 0;
}


