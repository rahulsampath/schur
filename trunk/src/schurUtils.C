
#include "schurUtils.h"

void createRSDtree(RSDnode* & root, MPI_Comm rootComm) {
  root = new RSDnode;
  MPI_Comm_dup(rootComm, &(root->comm));
  root->child1 = root->child2 = NULL;
  root->k11 = root->k22 = root->k33 = NULL;
  root->k13 = root->k31 = NULL;
  root->k23 = root->k32 = NULL;
  int npes;
  MPI_Comm_size(rootComm, &npes);
  if(npes > 1) {
    int rank;
    MPI_Comm_rank(rootComm, &rank);
    MPI_Comm newComm;
    if(rank < (npes/2)) {
      MPI_Comm_split(rootComm, 1, rank, &newComm);
      createRSDtree(root->child1, newComm);
    } else {
      MPI_Comm_split(rootComm, 2, rank, &newComm);
      createRSDtree(root->child2, newComm);
    }
    MPI_Comm_free(&newComm);
  }
}

