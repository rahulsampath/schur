
#include "schurUtils.h"


void deleteRSDtree(RSDnode* & root) {
  if(root->child1) {
    deleteRSDtree(root->child1);
  }
  if(root->child2) {
    deleteRSDtree(root->child2);
  }
  MPI_Comm_free(&(root->comm));
  MatDestroy(root->k11);
  MatDestroy(root->k22);
  MatDestroy(root->k33);
  MatDestroy(root->k13);
  MatDestroy(root->k31);
  MatDestroy(root->k23);
  MatDestroy(root->k32);
  delete root;
  root = NULL;
}

void createRSDtree(RSDnode* & root, MPI_Comm rootComm, int childNumber) {
  root = new RSDnode;
  root->childNumber = childNumber;
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
      createRSDtree(root->child1, newComm, 1);
    } else {
      MPI_Comm_split(rootComm, 2, rank, &newComm);
      createRSDtree(root->child2, newComm, 2);
    }
    MPI_Comm_free(&newComm);
  }
}

