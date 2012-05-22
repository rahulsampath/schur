
#include "schur.h"
#include <vector>

extern double** stencil;

PetscErrorCode computeMGmatrix(DMMG dmmg, Mat J, Mat B) {
  assert(J == B);

  DA da = (DA)(dmmg->dm);

  int N;
  int dofsPerNode;
  DAGetInfo(da, PETSC_NULL, &N, PETSC_NULL, PETSC_NULL, 
      PETSC_NULL, PETSC_NULL, PETSC_NULL, 
      &dofsPerNode, PETSC_NULL, PETSC_NULL, PETSC_NULL);

  MatZeroEntries(J);

  int Ne = N - 1;
  for(int yi = 0; yi < Ne; ++yi) {
    for(int xi = 0; xi < Ne; ++xi) {
      std::vector<int> dofs(4*dofsPerNode);
      for(int d = 0; d < dofsPerNode; ++d) {
        dofs[(0*dofsPerNode) + d] = (((yi*N) + xi)*dofsPerNode) + d;
        dofs[(1*dofsPerNode) + d] = (((yi*N) + xi + 1)*dofsPerNode) + d;
        dofs[(2*dofsPerNode) + d] = ((((yi + 1)*N) + xi)*dofsPerNode) + d;
        dofs[(3*dofsPerNode) + d] = ((((yi + 1)*N) + xi + 1)*dofsPerNode) + d;
      }//end d
      for(int j = 0; j < (4*dofsPerNode); ++j) {
        for(int i = 0; i < (4*dofsPerNode); i++) {
          MatSetValue(J, dofs[j], dofs[i], stencil[j][i], ADD_VALUES);
        }//end i
      }//end j
    }//end xi
  }//end yi

  MatAssemblyBegin(J, MAT_FLUSH_ASSEMBLY);
  MatAssemblyEnd(J, MAT_FLUSH_ASSEMBLY);

  //Left
  for(int yi = 0; yi < N; ++yi) {
    int xi = 0;
    int bndNd = (yi*N) + xi;
    int nhNd[] = {-1, -1, -1, -1, -1, -1, -1, -1};
    if(yi > 0) {
      if(xi > 0) {
        nhNd[0] = ((yi - 1)*N) + xi - 1;
      }
      nhNd[1] = ((yi - 1)*N) + xi;
      if(xi < (N - 1)) {
        nhNd[2] = ((yi - 1)*N) + xi + 1;
      }
    }
    if(xi > 0) {
      nhNd[3] = (yi*N) + xi - 1;
    }
    if(xi < (N - 1)) {
      nhNd[4] = (yi*N) + xi + 1;
    }
    if(yi < (N - 1)) {
      if(xi > 0) {
        nhNd[5] = ((yi + 1)*N) + xi - 1;
      }
      nhNd[6] = ((yi + 1)*N) + xi;   
      if(xi < (N - 1)) {
        nhNd[7] = ((yi + 1)*N) + xi + 1;
      }
    }
    for(int i = 0; i < 8; ++i) {
      if(nhNd[i] != -1) {
        MatSetValue(J, bndNd, nhNd[i], 0.0, INSERT_VALUES);
        MatSetValue(J, nhNd[i], bndNd, 0.0, INSERT_VALUES);
      }
    }//end i
    MatSetValue(J, bndNd, bndNd, 1.0, INSERT_VALUES);
  }//end yi

  //Right
  for(int yi = 0; yi < N; ++yi) {
    int xi = (N - 1);
    int bndNd = (yi*N) + xi;
    int nhNd[] = {-1, -1, -1, -1, -1, -1, -1, -1};
    if(yi > 0) {
      if(xi > 0) {
        nhNd[0] = ((yi - 1)*N) + xi - 1;
      }
      nhNd[1] = ((yi - 1)*N) + xi;
      if(xi < (N - 1)) {
        nhNd[2] = ((yi - 1)*N) + xi + 1;
      }
    }
    if(xi > 0) {
      nhNd[3] = (yi*N) + xi - 1;
    }
    if(xi < (N - 1)) {
      nhNd[4] = (yi*N) + xi + 1;
    }
    if(yi < (N - 1)) {
      if(xi > 0) {
        nhNd[5] = ((yi + 1)*N) + xi - 1;
      }
      nhNd[6] = ((yi + 1)*N) + xi;   
      if(xi < (N - 1)) {
        nhNd[7] = ((yi + 1)*N) + xi + 1;
      }
    }
    for(int i = 0; i < 8; ++i) {
      if(nhNd[i] != -1) {
        MatSetValue(J, bndNd, nhNd[i], 0.0, INSERT_VALUES);
        MatSetValue(J, nhNd[i], bndNd, 0.0, INSERT_VALUES);
      }
    }//end i
    MatSetValue(J, bndNd, bndNd, 1.0, INSERT_VALUES);
  }//end yi

  //Top
  for(int xi = 0; xi < N; ++xi) {
    int yi = (N - 1);
    int bndNd = (yi*N) + xi;
    int nhNd[] = {-1, -1, -1, -1, -1, -1, -1, -1};
    if(yi > 0) {
      if(xi > 0) {
        nhNd[0] = ((yi - 1)*N) + xi - 1;
      }
      nhNd[1] = ((yi - 1)*N) + xi;
      if(xi < (N - 1)) {
        nhNd[2] = ((yi - 1)*N) + xi + 1;
      }
    }
    if(xi > 0) {
      nhNd[3] = (yi*N) + xi - 1;
    }
    if(xi < (N - 1)) {
      nhNd[4] = (yi*N) + xi + 1;
    }
    if(yi < (N - 1)) {
      if(xi > 0) {
        nhNd[5] = ((yi + 1)*N) + xi - 1;
      }
      nhNd[6] = ((yi + 1)*N) + xi;   
      if(xi < (N - 1)) {
        nhNd[7] = ((yi + 1)*N) + xi + 1;
      }
    }
    for(int i = 0; i < 8; ++i) {
      if(nhNd[i] != -1) {
        MatSetValue(J, bndNd, nhNd[i], 0.0, INSERT_VALUES);
        MatSetValue(J, nhNd[i], bndNd, 0.0, INSERT_VALUES);
      }
    }//end i
    MatSetValue(J, bndNd, bndNd, 1.0, INSERT_VALUES);
  }//end xi

  //Bottom
  for(int xi = 0; xi < N; ++xi) {
    int yi = 0;
    int bndNd = (yi*N) + xi;
    int nhNd[] = {-1, -1, -1, -1, -1, -1, -1, -1};
    if(yi > 0) {
      if(xi > 0) {
        nhNd[0] = ((yi - 1)*N) + xi - 1;
      }
      nhNd[1] = ((yi - 1)*N) + xi;
      if(xi < (N - 1)) {
        nhNd[2] = ((yi - 1)*N) + xi + 1;
      }
    }
    if(xi > 0) {
      nhNd[3] = (yi*N) + xi - 1;
    }
    if(xi < (N - 1)) {
      nhNd[4] = (yi*N) + xi + 1;
    }
    if(yi < (N - 1)) {
      if(xi > 0) {
        nhNd[5] = ((yi + 1)*N) + xi - 1;
      }
      nhNd[6] = ((yi + 1)*N) + xi;   
      if(xi < (N - 1)) {
        nhNd[7] = ((yi + 1)*N) + xi + 1;
      }
    }
    for(int i = 0; i < 8; ++i) {
      if(nhNd[i] != -1) {
        MatSetValue(J, bndNd, nhNd[i], 0.0, INSERT_VALUES);
        MatSetValue(J, nhNd[i], bndNd, 0.0, INSERT_VALUES);
      }
    }//end i
    MatSetValue(J, bndNd, bndNd, 1.0, INSERT_VALUES);
  }//end xi

  MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);

  return 0;
}

void createLocalMatrices(LocalData* data) {
  int rank, npes;
  MPI_Comm_rank(data->commAll, &rank);
  MPI_Comm_size(data->commAll, &npes);

  int Ne = (data->N) - 1;

  if(rank > 0) {
    MatCreateSeqAIJ(PETSC_COMM_SELF, ((data->N)*(data->dofsPerNode)), ((data->N)*(data->dofsPerNode)),
        (9*(data->dofsPerNode)), PETSC_NULL, &(data->Kssh));
    MatCreateSeqAIJ(PETSC_COMM_SELF, ((data->N)*(data->dofsPerNode)), ((data->N)*(data->dofsPerNode)),
        (9*(data->dofsPerNode)), PETSC_NULL, &(data->Ksh));
    MatCreateSeqAIJ(PETSC_COMM_SELF, ((data->N)*(data->dofsPerNode)), ((data->N)*(data->dofsPerNode)),
        (9*(data->dofsPerNode)), PETSC_NULL, &(data->Khs));
    MatCreateSeqAIJ(PETSC_COMM_SELF, ((data->N)*(data->dofsPerNode)), ((data->N)*(data->dofsPerNode)),
        (9*(data->dofsPerNode)), PETSC_NULL, &(data->Khh));
    MatZeroEntries(data->Kssh);
    MatZeroEntries(data->Ksh);
    MatZeroEntries(data->Khs);
    MatZeroEntries(data->Khh);

    for(int yi = 0; yi < Ne; ++yi) {
      int dofId[] = {0, 2};
      int dofs[2];
      dofs[0] = yi;
      dofs[1] = yi + 1;
      for(int j = 0; j < 2; j++) {
        for(int i = 0; i < 2; i++) {
          MatSetValue(data->Kssh, dofs[j], dofs[i], stencil[dofId[j]][dofId[i]], ADD_VALUES);
        }//end i
      }//end j
    }//end yi

    MatAssemblyBegin(data->Kssh, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(data->Kssh, MAT_FLUSH_ASSEMBLY);

    MatSetValue(data->Kssh, 0, 1, 0.0, INSERT_VALUES);
    MatSetValue(data->Kssh, 1, 0, 0.0, INSERT_VALUES);
    MatSetValue(data->Kssh, 0, 0, 0.0, INSERT_VALUES);
    MatSetValue(data->Kssh, ((data->N) - 1), ((data->N) - 2), 0.0, INSERT_VALUES);
    MatSetValue(data->Kssh, ((data->N) - 2), ((data->N) - 1), 0.0, INSERT_VALUES);
    MatSetValue(data->Kssh, ((data->N) - 1), ((data->N) - 1), 0.0, INSERT_VALUES);

    MatAssemblyBegin(data->Kssh, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(data->Kssh, MAT_FINAL_ASSEMBLY);

    for(int yi = 0; yi < Ne; ++yi) {
      int sDofId[] = {0, 2};
      int hDofId[] = {1, 3};
      int dofs[2];
      dofs[0] = yi;
      dofs[1] = yi + 1;
      for(int si = 0; si < 2; si++) {
        for(int hi = 0; hi < 2; hi++) {
          MatSetValue(data->Ksh, dofs[si], dofs[hi], stencil[sDofId[si]][hDofId[hi]], ADD_VALUES);
          MatSetValue(data->Khs, dofs[hi], dofs[si], stencil[hDofId[hi]][sDofId[si]], ADD_VALUES);
        }//end hi
      }//end si
    }//end yi

    MatAssemblyBegin(data->Ksh, MAT_FLUSH_ASSEMBLY);
    MatAssemblyBegin(data->Khs, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(data->Ksh, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(data->Khs, MAT_FLUSH_ASSEMBLY);

    MatSetValue(data->Ksh, 0, 1, 0.0, INSERT_VALUES);
    MatSetValue(data->Ksh, 1, 0, 0.0, INSERT_VALUES);
    MatSetValue(data->Ksh, 0, 0, 0.0, INSERT_VALUES);
    MatSetValue(data->Ksh, ((data->N) - 1), ((data->N) - 2), 0.0, INSERT_VALUES);
    MatSetValue(data->Ksh, ((data->N) - 2), ((data->N) - 1), 0.0, INSERT_VALUES);
    MatSetValue(data->Ksh, ((data->N) - 1), ((data->N) - 1), 0.0, INSERT_VALUES);

    MatSetValue(data->Khs, 0, 1, 0.0, INSERT_VALUES);
    MatSetValue(data->Khs, 1, 0, 0.0, INSERT_VALUES);
    MatSetValue(data->Khs, 0, 0, 0.0, INSERT_VALUES);
    MatSetValue(data->Khs, ((data->N) - 1), ((data->N) - 2), 0.0, INSERT_VALUES);
    MatSetValue(data->Khs, ((data->N) - 2), ((data->N) - 1), 0.0, INSERT_VALUES);
    MatSetValue(data->Khs, ((data->N) - 1), ((data->N) - 1), 0.0, INSERT_VALUES);

    MatAssemblyBegin(data->Ksh, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(data->Khs, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(data->Ksh, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(data->Khs, MAT_FINAL_ASSEMBLY);

    for(int yi = 0; yi < Ne; ++yi) {
      int e1DofId[] = {0, 2};
      int e2DofId[] = {1, 3};
      int dofs[2];
      dofs[0] = yi;
      dofs[1] = yi + 1;
      for(int j = 0; j < 2; j++) {
        for(int i = 0; i < 2; i++) {
          MatSetValue(data->Khh, dofs[j], dofs[i], stencil[e1DofId[j]][e1DofId[i]], ADD_VALUES);
          MatSetValue(data->Khh, dofs[j], dofs[i], stencil[e2DofId[j]][e2DofId[i]], ADD_VALUES);
        }//end i
      }//end j
    }//end yi

    MatAssemblyBegin(data->Khh, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(data->Khh, MAT_FLUSH_ASSEMBLY);

    MatSetValue(data->Khh, 0, 1, 0.0, INSERT_VALUES);
    MatSetValue(data->Khh, 1, 0, 0.0, INSERT_VALUES);
    MatSetValue(data->Khh, 0, 0, 1.0, INSERT_VALUES);
    MatSetValue(data->Khh, ((data->N) - 1), ((data->N) - 2), 0.0, INSERT_VALUES);
    MatSetValue(data->Khh, ((data->N) - 2), ((data->N) - 1), 0.0, INSERT_VALUES);
    MatSetValue(data->Khh, ((data->N) - 1), ((data->N) - 1), 1.0, INSERT_VALUES);

    MatAssemblyBegin(data->Khh, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(data->Khh, MAT_FINAL_ASSEMBLY);
  } else {
    data->Kssh = PETSC_NULL;
    data->Ksh  = PETSC_NULL;
    data->Khs  = PETSC_NULL;
    data->Khh  = PETSC_NULL;
  }

  if(rank < (npes - 1)) {
    MatCreateSeqAIJ(PETSC_COMM_SELF, ((data->N)*(data->dofsPerNode)), ((data->N)*(data->dofsPerNode)),
        (9*(data->dofsPerNode)), PETSC_NULL, &(data->Kssl));
    MatCreateSeqAIJ(PETSC_COMM_SELF, ((data->N)*(data->dofsPerNode)), ((data->N)*(data->dofsPerNode)),
        (9*(data->dofsPerNode)), PETSC_NULL, &(data->Ksl));
    MatCreateSeqAIJ(PETSC_COMM_SELF, ((data->N)*(data->dofsPerNode)), ((data->N)*(data->dofsPerNode)),
        (9*(data->dofsPerNode)), PETSC_NULL, &(data->Kls));
    MatCreateSeqAIJ(PETSC_COMM_SELF, ((data->N)*(data->dofsPerNode)), ((data->N)*(data->dofsPerNode)),
        (9*(data->dofsPerNode)), PETSC_NULL, &(data->Kll));
    MatZeroEntries(data->Kssl);
    MatZeroEntries(data->Ksl);
    MatZeroEntries(data->Kls);
    MatZeroEntries(data->Kll);

    for(int yi = 0; yi < Ne; ++yi) {
      int dofId[] = {1, 3};
      int dofs[2];
      dofs[0] = yi;
      dofs[1] = yi + 1;
      for(int j = 0; j < 2; j++) {
        for(int i = 0; i < 2; i++) {
          MatSetValue(data->Kssl, dofs[j], dofs[i], stencil[dofId[j]][dofId[i]], ADD_VALUES);
        }//end i
      }//end j
    }//end yi

    MatAssemblyBegin(data->Kssl, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(data->Kssl, MAT_FLUSH_ASSEMBLY);

    MatSetValue(data->Kssl, 0, 1, 0.0, INSERT_VALUES);
    MatSetValue(data->Kssl, 1, 0, 0.0, INSERT_VALUES);
    MatSetValue(data->Kssl, 0, 0, 1.0, INSERT_VALUES);
    MatSetValue(data->Kssl, ((data->N) - 1), ((data->N) - 2), 0.0, INSERT_VALUES);
    MatSetValue(data->Kssl, ((data->N) - 2), ((data->N) - 1), 0.0, INSERT_VALUES);
    MatSetValue(data->Kssl, ((data->N) - 1), ((data->N) - 1), 1.0, INSERT_VALUES);

    MatAssemblyBegin(data->Kssl, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(data->Kssl, MAT_FINAL_ASSEMBLY);

    for(int yi = 0; yi < Ne; ++yi) {
      int sDofId[] = {1, 3};
      int lDofId[] = {0, 2};
      int dofs[2];
      dofs[0] = yi;
      dofs[1] = yi + 1;
      for(int si = 0; si < 2; si++) {
        for(int li = 0; li < 2; li++) {
          MatSetValue(data->Ksl, dofs[si], dofs[li], stencil[sDofId[si]][lDofId[li]], ADD_VALUES);
          MatSetValue(data->Kls, dofs[li], dofs[si], stencil[lDofId[li]][sDofId[si]], ADD_VALUES);
        }//end li
      }//end si
    }//end yi

    MatAssemblyBegin(data->Ksl, MAT_FLUSH_ASSEMBLY);
    MatAssemblyBegin(data->Kls, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(data->Ksl, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(data->Kls, MAT_FLUSH_ASSEMBLY);

    MatSetValue(data->Ksl, 0, 1, 0.0, INSERT_VALUES);
    MatSetValue(data->Ksl, 1, 0, 0.0, INSERT_VALUES);
    MatSetValue(data->Ksl, 0, 0, 0.0, INSERT_VALUES);
    MatSetValue(data->Ksl, ((data->N) - 1), ((data->N) - 2), 0.0, INSERT_VALUES);
    MatSetValue(data->Ksl, ((data->N) - 2), ((data->N) - 1), 0.0, INSERT_VALUES);
    MatSetValue(data->Ksl, ((data->N) - 1), ((data->N) - 1), 0.0, INSERT_VALUES);

    MatSetValue(data->Kls, 0, 1, 0.0, INSERT_VALUES);
    MatSetValue(data->Kls, 1, 0, 0.0, INSERT_VALUES);
    MatSetValue(data->Kls, 0, 0, 0.0, INSERT_VALUES);
    MatSetValue(data->Kls, ((data->N) - 1), ((data->N) - 2), 0.0, INSERT_VALUES);
    MatSetValue(data->Kls, ((data->N) - 2), ((data->N) - 1), 0.0, INSERT_VALUES);
    MatSetValue(data->Kls, ((data->N) - 1), ((data->N) - 1), 0.0, INSERT_VALUES);

    MatAssemblyBegin(data->Ksl, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(data->Kls, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(data->Ksl, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(data->Kls, MAT_FINAL_ASSEMBLY);

    for(int yi = 0; yi < Ne; ++yi) {
      int e1DofId[] = {0, 2};
      int e2DofId[] = {1, 3};
      int dofs[2];
      dofs[0] = yi;
      dofs[1] = yi + 1;
      for(int j = 0; j < 2; j++) {
        for(int i = 0; i < 2; i++) {
          MatSetValue(data->Kll, dofs[j], dofs[i], stencil[e1DofId[j]][e1DofId[i]], ADD_VALUES);
          MatSetValue(data->Kll, dofs[j], dofs[i], stencil[e2DofId[j]][e2DofId[i]], ADD_VALUES);
        }//end i
      }//end j
    }//end yi

    MatAssemblyBegin(data->Kll, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(data->Kll, MAT_FLUSH_ASSEMBLY);

    MatSetValue(data->Kll, 0, 1, 0.0, INSERT_VALUES);
    MatSetValue(data->Kll, 1, 0, 0.0, INSERT_VALUES);
    MatSetValue(data->Kll, 0, 0, 1.0, INSERT_VALUES);
    MatSetValue(data->Kll, ((data->N) - 1), ((data->N) - 2), 0.0, INSERT_VALUES);
    MatSetValue(data->Kll, ((data->N) - 2), ((data->N) - 1), 0.0, INSERT_VALUES);
    MatSetValue(data->Kll, ((data->N) - 1), ((data->N) - 1), 1.0, INSERT_VALUES);

    MatAssemblyBegin(data->Kll, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(data->Kll, MAT_FINAL_ASSEMBLY);
  } else {
    data->Kssl = PETSC_NULL;
    data->Ksl = PETSC_NULL;
    data->Kls = PETSC_NULL;
    data->Kll = PETSC_NULL;
  }
}

void zeroBoundary(LocalData* data, Vec vec) {
  int rank, npes;
  MPI_Comm_rank(data->commAll, &rank);
  MPI_Comm_size(data->commAll, &npes);

  PetscScalar* arr;
  VecGetArray(vec, &arr);

  int oxs, onx;
  if(rank == 0) {
    oxs = 0;
    onx = data->N;
  } else {
    oxs = 1;
    onx = (data->N) - 1;
  }

  //Left
  if(rank == 0) {
    for(int yi = 0; yi < (data->N); ++yi) {
      int xi = oxs;
      arr[(yi*onx) + (xi - oxs)] = 0;
    }//end for yi
  }

  //Right
  if(rank == (npes - 1)) {
    for(int yi = 0; yi < (data->N); ++yi) {
      int xi = (oxs + onx) - 1;
      arr[(yi*onx) + (xi - oxs)] = 0;
    }//end for yi
  }

  //Top
  for(int xi = oxs; xi < (oxs + onx); ++xi) {
    int yi = (data->N) - 1;
    arr[(yi*onx) + (xi - oxs)] = 0;
  }//end for xi

  //Bottom
  for(int xi = oxs; xi < (oxs + onx); ++xi) {
    int yi = 0;
    arr[(yi*onx) + (xi - oxs)] = 0;
  }//end for xi

  VecRestoreArray(vec, &arr);
}



