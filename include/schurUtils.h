
#ifndef __SCHUR_UTILS__
#define __SCHUR_UTILS__

#include "schur.h"

void createMatrices(RSDnode* root, int localNx, int localNy);

void createRSDtree(RSDnode* & root, MPI_Comm rootComm, int childNumber = 1);

void deleteRSDtree(RSDnode* & root);

#endif

