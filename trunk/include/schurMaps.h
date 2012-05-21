
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
    oArr[(yi*onx) + ((data->N - 2) - oxs)] = lArr[yi];
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
    lArr[yi] = oArr[(yi*onx) + ((data->N - 2) - oxs)];
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
    oArr[(yi*onx) + (1 - oxs)] = hArr[yi];
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
    hArr[yi] = oArr[(yi*onx) + (1 - oxs)];
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
    sArr[yi] = oArr[(yi*onx) + ((data->N - 1) - oxs)];
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
    oArr[(yi*onx) + ((data->N - 1) - oxs)] = sArr[yi];
  }//end for yi

  VecRestoreArray(fromVec, &sArr);
  VecRestoreArray(toVec, &oArr);
}

template<>
inline void map<MG, L>(LocalData* data, Vec fromVec, Vec toVec) {
  int rank, npes;
  MPI_Comm_rank(data->commAll, &rank);
  MPI_Comm_size(data->commAll, &npes);

  DA da = DMMGGetDA(data->mgObj);

  PetscScalar** mgArr;
  PetscScalar* lArr;

  assert(rank < (npes - 1));

  VecGetArray(toVec, &lArr);
  DAVecGetArray(da, fromVec, &mgArr);

  for(int yi = 0; yi < (data->N); ++yi) {
    lArr[yi] = mgArr[yi][(data->N)- 2];
  }//end for yi

  DAVecRestoreArray(da, fromVec, &mgArr);
  VecRestoreArray(toVec, &lArr);
}

template<>
inline void map<L, MG>(LocalData* data, Vec fromVec, Vec toVec) {
  int rank, npes;
  MPI_Comm_rank(data->commAll, &rank);
  MPI_Comm_size(data->commAll, &npes);

  DA da = DMMGGetDA(data->mgObj);

  PetscScalar** mgArr;
  PetscScalar* lArr;

  assert(rank < (npes - 1));

  VecGetArray(fromVec, &lArr);
  DAVecGetArray(da, toVec, &mgArr);

  for(int yi = 0; yi < (data->N); ++yi) {
    mgArr[yi][(data->N) - 2] = lArr[yi];
  }//end for yi

  DAVecRestoreArray(da, toVec, &mgArr);
  VecRestoreArray(fromVec, &lArr);
}

template<>
inline void map<MG, H>(LocalData* data, Vec fromVec, Vec toVec) {
  int rank;
  MPI_Comm_rank(data->commAll, &rank);

  DA da = DMMGGetDA(data->mgObj);

  PetscScalar** mgArr;
  PetscScalar* hArr;

  VecGetArray(toVec, &hArr);
  DAVecGetArray(da, fromVec, &mgArr);

  assert(rank > 0);

  for(int yi = 0; yi < (data->N); ++yi) {
    hArr[yi] = mgArr[yi][1];
  }//end for yi

  DAVecRestoreArray(da, fromVec, &mgArr);
  VecRestoreArray(toVec, &hArr);
}

template<>
inline void map<H, MG>(LocalData* data, Vec fromVec, Vec toVec) {
  int rank;
  MPI_Comm_rank(data->commAll, &rank);

  DA da = DMMGGetDA(data->mgObj);

  PetscScalar** mgArr;
  PetscScalar* hArr;

  VecGetArray(fromVec, &hArr);
  DAVecGetArray(da, toVec, &mgArr);

  assert(rank > 0);

  for(int yi = 0; yi < (data->N); ++yi) {
    mgArr[yi][1] = hArr[yi];
  }//end for yi

  DAVecRestoreArray(da, toVec, &mgArr);
  VecRestoreArray(fromVec, &hArr);
}

template<>
inline void map<MG, O>(LocalData* data, Vec fromVec, Vec toVec) {
  int rank, npes;
  MPI_Comm_rank(data->commAll, &rank);
  MPI_Comm_size(data->commAll, &npes);

  DA da = DMMGGetDA(data->mgObj);

  PetscScalar** mgArr;
  PetscScalar* oArr;

  VecGetArray(toVec, &oArr);
  DAVecGetArray(da, fromVec, &mgArr);

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
      oArr[(yi*onx) + (xi - oxs)] = mgArr[yi][xi];
    }//end for xi
  }//end for yi

  DAVecRestoreArray(da, fromVec, &mgArr);
  VecRestoreArray(toVec, &oArr);
}

template<>
inline void map<O, MG>(LocalData* data, Vec fromVec, Vec toVec) {
  int rank, npes;
  MPI_Comm_rank(data->commAll, &rank);
  MPI_Comm_size(data->commAll, &npes);

  DA da = DMMGGetDA(data->mgObj);

  PetscScalar** mgArr;
  PetscScalar* oArr;

  VecGetArray(fromVec, &oArr);
  DAVecGetArray(da, toVec, &mgArr);

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
      mgArr[yi][xi] = oArr[(yi*onx) + (xi - oxs)];
    }//end for xi
  }//end for yi

  DAVecRestoreArray(da, toVec, &mgArr);
  VecRestoreArray(fromVec, &oArr);
}

#endif

