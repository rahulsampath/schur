
#include "mpi.h"
#include "schur.h"
#include <cassert>

void createRSDtree(RSDnode *& root, int rank, int npes) {
  root = new RSDnode;
  root->rankForCurrLevel = rank;
  root->npesForCurrLevel = npes;
  if(npes > 1) {
    if(rank < (npes/2)) {
      createRSDtree(root->child, rank, (npes/2));
    } else {
      createRSDtree(root->child, (rank - (npes/2)), (npes/2));
    }
  } else {
    root->child = NULL;
  }
}

void createLowAndHighComms(LocalData* data) {
  int rank, npes;
  MPI_Comm_rank(data->commAll, &rank);
  MPI_Comm_size(data->commAll, &npes);

  MPI_Group groupAll;
  MPI_Comm_group(data->commAll, &groupAll);

  if(rank > 0) {
    int highRanks[2];
    highRanks[0] = rank - 1;
    highRanks[1] = rank;
    MPI_Group highGroup;
    MPI_Group_incl(groupAll, 2, highRanks, &highGroup);
    MPI_Comm_create(data->commAll, highGroup, &(data->commHigh));
    MPI_Group_free(&highGroup);
  } else {
    data->commHigh = MPI_COMM_NULL;
  }

  if(rank < (npes - 1)) {
    int lowRanks[2];
    lowRanks[0] = rank;
    lowRanks[1] = rank + 1;
    MPI_Group lowGroup;
    MPI_Group_incl(groupAll, 2, lowRanks, &lowGroup);
    MPI_Comm_create(data->commAll, lowGroup, &(data->commLow));
    MPI_Group_free(&lowGroup);
  } else {
    data->commLow = MPI_COMM_NULL;
  }
}




