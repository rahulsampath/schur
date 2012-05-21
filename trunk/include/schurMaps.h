
#ifndef __SCHUR_MAPS__
#define __SCHUR_MAPS__

template<>
inline void map<L, O>(LocalData* data, Vec fromVec, Vec toVec) {
  int rank, npes;
  MPI_Comm_rank(data->commAll, &rank);
  MPI_Comm_size(data->commAll, &npes);

  assert(rank < (npes - 1));

  PetscScalar* oArr;
  PetscScalar* lArr;

  int oxs, onx;
  if(rank == 0) {
    oxs = 0;
    onx = data->N;
  } else {
    oxs = 1;
    onx = (data->N) - 1;
  }

  VecGetArray(fromVec, &lArr);
  VecGetArray(toVec, &oArr);

  for(int yi = 0; yi < (data->N); ++yi) {
    for(int d = 0; d < (data->dofsPerNode); ++d) {
      oArr[(((yi*onx) + ((data->N - 2) - oxs))*(data->dofsPerNode)) + d] = lArr[(yi*(data->dofsPerNode)) + d];
    }//end for d
  }//end for yi

  VecRestoreArray(fromVec, &lArr);
  VecRestoreArray(toVec, &oArr);
}

template<>
inline void map<O, L>(LocalData* data, Vec fromVec, Vec toVec) {
  int rank, npes;
  MPI_Comm_rank(data->commAll, &rank);
  MPI_Comm_size(data->commAll, &npes);

  assert(rank < (npes - 1));

  PetscScalar* oArr;
  PetscScalar* lArr;

  int oxs, onx;
  if(rank == 0) {
    oxs = 0;
    onx = data->N;
  } else {
    oxs = 1;
    onx = (data->N) - 1;
  }

  VecGetArray(fromVec, &oArr);
  VecGetArray(toVec, &lArr);

  for(int yi = 0; yi < (data->N); ++yi) {
    for(int d = 0; d < (data->dofsPerNode); ++d) {
      lArr[(yi*(data->dofsPerNode)) + d] = oArr[(((yi*onx) + ((data->N - 2) - oxs))*(data->dofsPerNode)) + d];
    }//end for d
  }//end for yi

  VecRestoreArray(fromVec, &oArr);
  VecRestoreArray(toVec, &lArr);
}

template<>
inline void map<H, O>(LocalData* data, Vec fromVec, Vec toVec) {
  int rank;
  MPI_Comm_rank(data->commAll, &rank);

  assert(rank > 0);

  PetscScalar* oArr;
  PetscScalar* hArr;

  int oxs = 1;
  int onx = (data->N) - 1;

  VecGetArray(fromVec, &hArr);
  VecGetArray(toVec, &oArr);

  for(int yi = 0; yi < (data->N); ++yi) {
    for(int d = 0; d < (data->dofsPerNode); ++d) {
      oArr[(((yi*onx) + (1 - oxs))*(data->dofsPerNode)) + d] = hArr[(yi*(data->dofsPerNode)) + d];
    }//end for d
  }//end for yi

  VecRestoreArray(fromVec, &hArr);
  VecRestoreArray(toVec, &oArr);
}

template<>
inline void map<O, H>(LocalData* data, Vec fromVec, Vec toVec) {
  int rank;
  MPI_Comm_rank(data->commAll, &rank);

  assert(rank > 0);

  PetscScalar* oArr;
  PetscScalar* hArr;

  int oxs = 1;
  int onx = (data->N) - 1;

  VecGetArray(fromVec, &oArr);
  VecGetArray(toVec, &hArr);

  for(int yi = 0; yi < (data->N); ++yi) {
    for(int d = 0; d < (data->dofsPerNode); ++d) {
      hArr[(yi*(data->dofsPerNode)) + d] = oArr[(((yi*onx) + (1 - oxs))*(data->dofsPerNode)) + d];
    }//end for d
  }//end for yi

  VecRestoreArray(fromVec, &oArr);
  VecRestoreArray(toVec, &hArr);
}

template<>
inline void map<O, S>(LocalData* data, Vec fromVec, Vec toVec) {
  int rank, npes;
  MPI_Comm_rank(data->commAll, &rank);
  MPI_Comm_size(data->commAll, &npes);

  assert(rank < (npes - 1));

  PetscScalar* oArr;
  PetscScalar* sArr;

  int oxs, onx;
  if(rank == 0) {
    oxs = 0;
    onx = data->N;
  } else {
    oxs = 1;
    onx = (data->N) - 1;
  }

  VecGetArray(fromVec, &oArr);
  VecGetArray(toVec, &sArr);

  for(int yi = 0; yi < (data->N); ++yi) {
    for(int d = 0; d < (data->dofsPerNode); ++d) {
      sArr[(yi*(data->dofsPerNode)) + d] = oArr[(((yi*onx) + ((data->N - 1) - oxs))*(data->dofsPerNode)) + d];
    }//end for d
  }//end for yi

  VecRestoreArray(fromVec, &oArr);
  VecRestoreArray(toVec, &sArr);
}

template<>
inline void map<S, O>(LocalData* data, Vec fromVec, Vec toVec) {
  int rank, npes;
  MPI_Comm_rank(data->commAll, &rank);
  MPI_Comm_size(data->commAll, &npes);

  assert(rank < (npes - 1));

  PetscScalar* oArr;
  PetscScalar* sArr;

  int oxs, onx;
  if(rank == 0) {
    oxs = 0;
    onx = data->N;
  } else {
    oxs = 1;
    onx = (data->N) - 1;
  }

  VecGetArray(fromVec, &sArr);
  VecGetArray(toVec, &oArr);

  for(int yi = 0; yi < (data->N); ++yi) {
    for(int d = 0; d < (data->dofsPerNode); ++d) {
      oArr[(((yi*onx) + ((data->N - 1) - oxs))*(data->dofsPerNode)) + d] = sArr[(yi*(data->dofsPerNode)) + d];
    }//end for d
  }//end for yi

  VecRestoreArray(fromVec, &sArr);
  VecRestoreArray(toVec, &oArr);
}

template<>
inline void map<MG, L>(LocalData* data, Vec fromVec, Vec toVec) {
  int rank, npes;
  MPI_Comm_rank(data->commAll, &rank);
  MPI_Comm_size(data->commAll, &npes);

  PetscScalar* mgArr;
  PetscScalar* lArr;

  assert(rank < (npes - 1));

  VecGetArray(toVec, &lArr);
  VecGetArray(fromVec, &mgArr);

  for(int yi = 0; yi < (data->N); ++yi) {
    for(int d = 0; d < (data->dofsPerNode); ++d) {
      lArr[(yi*(data->dofsPerNode)) + d] = mgArr[(((yi*(data->N)) + (data->N - 2))*(data->dofsPerNode)) + d];
    }//end for d
  }//end for yi

  VecRestoreArray(fromVec, &mgArr);
  VecRestoreArray(toVec, &lArr);
}

template<>
inline void map<L, MG>(LocalData* data, Vec fromVec, Vec toVec) {
  int rank, npes;
  MPI_Comm_rank(data->commAll, &rank);
  MPI_Comm_size(data->commAll, &npes);

  PetscScalar* mgArr;
  PetscScalar* lArr;

  assert(rank < (npes - 1));

  VecGetArray(fromVec, &lArr);
  VecGetArray(toVec, &mgArr);

  for(int yi = 0; yi < (data->N); ++yi) {
    for(int d = 0; d < (data->dofsPerNode); ++d) {
      mgArr[(((yi*(data->N)) + (data->N - 2))*(data->dofsPerNode)) + d] = lArr[(yi*(data->dofsPerNode)) + d];
    }//end for d
  }//end for yi

  VecRestoreArray(toVec, &mgArr);
  VecRestoreArray(fromVec, &lArr);
}

template<>
inline void map<MG, H>(LocalData* data, Vec fromVec, Vec toVec) {
  int rank;
  MPI_Comm_rank(data->commAll, &rank);

  PetscScalar* mgArr;
  PetscScalar* hArr;

  VecGetArray(toVec, &hArr);
  VecGetArray(fromVec, &mgArr);

  assert(rank > 0);

  for(int yi = 0; yi < (data->N); ++yi) {
    for(int d = 0; d < (data->dofsPerNode); ++d) {
      hArr[(yi*(data->dofsPerNode)) + d] = mgArr[(((yi*(data->N)) + 1)*(data->dofsPerNode)) + d];
    }//end for d
  }//end for yi

  VecRestoreArray(fromVec, &mgArr);
  VecRestoreArray(toVec, &hArr);
}

template<>
inline void map<H, MG>(LocalData* data, Vec fromVec, Vec toVec) {
  int rank;
  MPI_Comm_rank(data->commAll, &rank);

  PetscScalar* mgArr;
  PetscScalar* hArr;

  VecGetArray(fromVec, &hArr);
  VecGetArray(toVec, &mgArr);

  assert(rank > 0);

  for(int yi = 0; yi < (data->N); ++yi) {
    for(int d = 0; d < (data->dofsPerNode); ++d) {
      mgArr[(((yi*(data->N)) + 1)*(data->dofsPerNode)) + d] = hArr[(yi*(data->dofsPerNode)) + d];
    }//end for d
  }//end for yi

  VecRestoreArray(toVec, &mgArr);
  VecRestoreArray(fromVec, &hArr);
}

template<>
inline void map<MG, O>(LocalData* data, Vec fromVec, Vec toVec) {
  int rank, npes;
  MPI_Comm_rank(data->commAll, &rank);
  MPI_Comm_size(data->commAll, &npes);

  PetscScalar* mgArr;
  PetscScalar* oArr;

  VecGetArray(toVec, &oArr);
  VecGetArray(fromVec, &mgArr);

  int oxs, onx;
  if(rank == 0) {
    oxs = 0;
    onx = data->N;
  } else {
    oxs = 1;
    onx = (data->N) - 1;
  }

  int vnx;
  if(rank == (npes - 1)) {
    vnx = onx;
  } else {
    vnx = onx - 1;
  }

  for(int yi = 0; yi < (data->N); ++yi) {
    for(int xi = oxs; xi < (oxs + vnx); ++xi) {
      for(int d = 0; d < (data->dofsPerNode); ++d) {
        oArr[(((yi*onx) + (xi - oxs))*(data->dofsPerNode)) + d] = mgArr[(((yi*(data->N)) + xi)*(data->dofsPerNode)) + d];
      }//end for d
    }//end for xi
  }//end for yi

  VecRestoreArray(fromVec, &mgArr);
  VecRestoreArray(toVec, &oArr);
}

template<>
inline void map<O, MG>(LocalData* data, Vec fromVec, Vec toVec) {
  int rank, npes;
  MPI_Comm_rank(data->commAll, &rank);
  MPI_Comm_size(data->commAll, &npes);

  PetscScalar* mgArr;
  PetscScalar* oArr;

  VecGetArray(fromVec, &oArr);
  VecGetArray(toVec, &mgArr);

  int oxs, onx;
  if(rank == 0) {
    oxs = 0;
    onx = data->N;
  } else {
    oxs = 1;
    onx = (data->N) - 1;
  }

  int vnx;
  if(rank == (npes - 1)) {
    vnx = onx;
  } else {
    vnx = onx - 1;
  }

  for(int yi = 0; yi < (data->N); ++yi) {
    for(int xi = oxs; xi < (oxs + vnx); ++xi) {
      for(int d = 0; d < (data->dofsPerNode); ++d) {
        mgArr[(((yi*(data->N)) + xi)*(data->dofsPerNode)) + d] = oArr[(((yi*onx) + (xi - oxs))*(data->dofsPerNode)) + d];
      }//end for d
    }//end for xi
  }//end for yi

  VecRestoreArray(toVec, &mgArr);
  VecRestoreArray(fromVec, &oArr);
}

#endif

