
#include "schurUtils.h"
#include <cassert>


void deleteRSDtree(RSDnode* & root) {
  if(root->child1) {
    deleteRSDtree(root->child1);
  }
  if(root->child2) {
    deleteRSDtree(root->child2);
  }
  MPI_Comm_free(&(root->comm));
  delete root;
  root = NULL;
}

void createRSDtree(RSDnode* & root, MPI_Comm rootComm) {
  root = new RSDnode;
  MPI_Comm_dup(rootComm, &(root->comm));
  root->child1 = root->child2 = NULL;
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


