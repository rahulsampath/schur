
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

  int* dofs = new int[4*dofsPerNode];
  int Ne = N - 1;
  for(int yi = 0; yi < Ne; ++yi) {
    for(int xi = 0; xi < Ne; ++xi) {
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
  delete [] dofs;
  dofs = NULL;

  MatAssemblyBegin(J, MAT_FLUSH_ASSEMBLY);
  MatAssemblyEnd(J, MAT_FLUSH_ASSEMBLY);

  //Left
  for(int yi = 0; yi < N; ++yi) {
    int xi = 0;
    int bnd = (yi*N) + xi;
    int nh[] = {-1, -1, -1, -1, -1, -1, -1, -1};
    if(yi > 0) {
      if(xi > 0) {
        nh[0] = ((yi - 1)*N) + xi - 1;
      }
      nh[1] = ((yi - 1)*N) + xi;
      if(xi < (N - 1)) {
        nh[2] = ((yi - 1)*N) + xi + 1;
      }
    }
    if(xi > 0) {
      nh[3] = (yi*N) + xi - 1;
    }
    if(xi < (N - 1)) {
      nh[4] = (yi*N) + xi + 1;
    }
    if(yi < (N - 1)) {
      if(xi > 0) {
        nh[5] = ((yi + 1)*N) + xi - 1;
      }
      nh[6] = ((yi + 1)*N) + xi;   
      if(xi < (N - 1)) {
        nh[7] = ((yi + 1)*N) + xi + 1;
      }
    }
    for(int i = 0; i < 8; ++i) {
      if(nh[i] != -1) {
        for(int db = 0; db < dofsPerNode; ++db) {
          for(int dn = 0; dn < dofsPerNode; ++dn) {
            MatSetValue(J, ((bnd*dofsPerNode) + db), ((nh[i]*dofsPerNode) + dn), 0.0, INSERT_VALUES);
            MatSetValue(J, ((nh[i]*dofsPerNode) + dn), ((bnd*dofsPerNode) + db), 0.0, INSERT_VALUES);
          }//end dn
        }//end db
      }
    }//end i
    for(int dr = 0; dr < dofsPerNode; ++dr) {
      for(int dc = 0; dc < dofsPerNode; ++dc) {
        if(dr == dc) {
          MatSetValue(J, ((bnd*dofsPerNode) + dr), ((bnd*dofsPerNode) + dc), 1.0, INSERT_VALUES);
        } else {
          MatSetValue(J, ((bnd*dofsPerNode) + dr), ((bnd*dofsPerNode) + dc), 0.0, INSERT_VALUES);
        }
      }//end dc
    }//end dr
  }//end yi

  //Right
  for(int yi = 0; yi < N; ++yi) {
    int xi = (N - 1);
    int bnd = (yi*N) + xi;
    int nh[] = {-1, -1, -1, -1, -1, -1, -1, -1};
    if(yi > 0) {
      if(xi > 0) {
        nh[0] = ((yi - 1)*N) + xi - 1;
      }
      nh[1] = ((yi - 1)*N) + xi;
      if(xi < (N - 1)) {
        nh[2] = ((yi - 1)*N) + xi + 1;
      }
    }
    if(xi > 0) {
      nh[3] = (yi*N) + xi - 1;
    }
    if(xi < (N - 1)) {
      nh[4] = (yi*N) + xi + 1;
    }
    if(yi < (N - 1)) {
      if(xi > 0) {
        nh[5] = ((yi + 1)*N) + xi - 1;
      }
      nh[6] = ((yi + 1)*N) + xi;   
      if(xi < (N - 1)) {
        nh[7] = ((yi + 1)*N) + xi + 1;
      }
    }
    for(int i = 0; i < 8; ++i) {
      if(nh[i] != -1) {
        for(int db = 0; db < dofsPerNode; ++db) {
          for(int dn = 0; dn < dofsPerNode; ++dn) {
            MatSetValue(J, ((bnd*dofsPerNode) + db), ((nh[i]*dofsPerNode) + dn), 0.0, INSERT_VALUES);
            MatSetValue(J, ((nh[i]*dofsPerNode) + dn), ((bnd*dofsPerNode) + db), 0.0, INSERT_VALUES);
          }//end dn
        }//end db
      }
    }//end i
    for(int dr = 0; dr < dofsPerNode; ++dr) {
      for(int dc = 0; dc < dofsPerNode; ++dc) {
        if(dr == dc) {
          MatSetValue(J, ((bnd*dofsPerNode) + dr), ((bnd*dofsPerNode) + dc), 1.0, INSERT_VALUES);
        } else {
          MatSetValue(J, ((bnd*dofsPerNode) + dr), ((bnd*dofsPerNode) + dc), 0.0, INSERT_VALUES);
        }
      }//end dc
    }//end dr
  }//end yi

  //Top
  for(int xi = 0; xi < N; ++xi) {
    int yi = (N - 1);
    int bnd = (yi*N) + xi;
    int nh[] = {-1, -1, -1, -1, -1, -1, -1, -1};
    if(yi > 0) {
      if(xi > 0) {
        nh[0] = ((yi - 1)*N) + xi - 1;
      }
      nh[1] = ((yi - 1)*N) + xi;
      if(xi < (N - 1)) {
        nh[2] = ((yi - 1)*N) + xi + 1;
      }
    }
    if(xi > 0) {
      nh[3] = (yi*N) + xi - 1;
    }
    if(xi < (N - 1)) {
      nh[4] = (yi*N) + xi + 1;
    }
    if(yi < (N - 1)) {
      if(xi > 0) {
        nh[5] = ((yi + 1)*N) + xi - 1;
      }
      nh[6] = ((yi + 1)*N) + xi;   
      if(xi < (N - 1)) {
        nh[7] = ((yi + 1)*N) + xi + 1;
      }
    }
    for(int i = 0; i < 8; ++i) {
      if(nh[i] != -1) {
        for(int db = 0; db < dofsPerNode; ++db) {
          for(int dn = 0; dn < dofsPerNode; ++dn) {
            MatSetValue(J, ((bnd*dofsPerNode) + db), ((nh[i]*dofsPerNode) + dn), 0.0, INSERT_VALUES);
            MatSetValue(J, ((nh[i]*dofsPerNode) + dn), ((bnd*dofsPerNode) + db), 0.0, INSERT_VALUES);
          }//end dn
        }//end db
      }
    }//end i
    for(int dr = 0; dr < dofsPerNode; ++dr) {
      for(int dc = 0; dc < dofsPerNode; ++dc) {
        if(dr == dc) {
          MatSetValue(J, ((bnd*dofsPerNode) + dr), ((bnd*dofsPerNode) + dc), 1.0, INSERT_VALUES);
        } else {
          MatSetValue(J, ((bnd*dofsPerNode) + dr), ((bnd*dofsPerNode) + dc), 0.0, INSERT_VALUES);
        }
      }//end dc
    }//end dr
  }//end xi

  //Bottom
  for(int xi = 0; xi < N; ++xi) {
    int yi = 0;
    int bnd = (yi*N) + xi;
    int nh[] = {-1, -1, -1, -1, -1, -1, -1, -1};
    if(yi > 0) {
      if(xi > 0) {
        nh[0] = ((yi - 1)*N) + xi - 1;
      }
      nh[1] = ((yi - 1)*N) + xi;
      if(xi < (N - 1)) {
        nh[2] = ((yi - 1)*N) + xi + 1;
      }
    }
    if(xi > 0) {
      nh[3] = (yi*N) + xi - 1;
    }
    if(xi < (N - 1)) {
      nh[4] = (yi*N) + xi + 1;
    }
    if(yi < (N - 1)) {
      if(xi > 0) {
        nh[5] = ((yi + 1)*N) + xi - 1;
      }
      nh[6] = ((yi + 1)*N) + xi;   
      if(xi < (N - 1)) {
        nh[7] = ((yi + 1)*N) + xi + 1;
      }
    }
    for(int i = 0; i < 8; ++i) {
      if(nh[i] != -1) {
        for(int db = 0; db < dofsPerNode; ++db) {
          for(int dn = 0; dn < dofsPerNode; ++dn) {
            MatSetValue(J, ((bnd*dofsPerNode) + db), ((nh[i]*dofsPerNode) + dn), 0.0, INSERT_VALUES);
            MatSetValue(J, ((nh[i]*dofsPerNode) + dn), ((bnd*dofsPerNode) + db), 0.0, INSERT_VALUES);
          }//end dn
        }//end db
      }
    }//end i
    for(int dr = 0; dr < dofsPerNode; ++dr) {
      for(int dc = 0; dc < dofsPerNode; ++dc) {
        if(dr == dc) {
          MatSetValue(J, ((bnd*dofsPerNode) + dr), ((bnd*dofsPerNode) + dc), 1.0, INSERT_VALUES);
        } else {
          MatSetValue(J, ((bnd*dofsPerNode) + dr), ((bnd*dofsPerNode) + dc), 0.0, INSERT_VALUES);
        }
      }//end dc
    }//end dr
  }//end xi

  MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);

  return 0;
}

void createLocalMatrices(LocalData* data) {
  int rank, npes;
  MPI_Comm_rank(data->commAll, &rank);
  MPI_Comm_size(data->commAll, &npes);

  const int N = data->N;
  const int dofsPerNode = data->dofsPerNode;
  const int Ne = N - 1;
  const int locSize = N*dofsPerNode;
  const int numNonZeros = 9*dofsPerNode;

  if(rank > 0) {
    MatCreateSeqAIJ(PETSC_COMM_SELF, locSize, locSize, numNonZeros, PETSC_NULL, &(data->Kssh));
    MatCreateSeqAIJ(PETSC_COMM_SELF, locSize, locSize, numNonZeros, PETSC_NULL, &(data->Ksh));
    MatCreateSeqAIJ(PETSC_COMM_SELF, locSize, locSize, numNonZeros, PETSC_NULL, &(data->Khs));
    MatCreateSeqAIJ(PETSC_COMM_SELF, locSize, locSize, numNonZeros, PETSC_NULL, &(data->Khh));
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
        for(int dj = 0; dj < dofsPerNode; ++dj) {
          for(int i = 0; i < 2; i++) {
            for(int di = 0; di < dofsPerNode; ++di) {
              MatSetValue(data->Kssh, ((dofs[j]*dofsPerNode) + dj), ((dofs[i]*dofsPerNode) + di), 
                  stencil[(dofId[j]*dofsPerNode) + dj][(dofId[i]*dofsPerNode) + di], ADD_VALUES);
            }//end di
          }//end i
        }//end dj
      }//end j
    }//end yi

    MatAssemblyBegin(data->Kssh, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(data->Kssh, MAT_FLUSH_ASSEMBLY);

    MatSetValue(data->Kssh, 0, 1, 0.0, INSERT_VALUES);
    MatSetValue(data->Kssh, 1, 0, 0.0, INSERT_VALUES);
    MatSetValue(data->Kssh, 0, 0, 0.0, INSERT_VALUES);
    MatSetValue(data->Kssh, (N - 1), (N - 2), 0.0, INSERT_VALUES);
    MatSetValue(data->Kssh, (N - 2), (N - 1), 0.0, INSERT_VALUES);
    MatSetValue(data->Kssh, (N - 1), (N - 1), 0.0, INSERT_VALUES);

    MatAssemblyBegin(data->Kssh, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(data->Kssh, MAT_FINAL_ASSEMBLY);

    for(int yi = 0; yi < Ne; ++yi) {
      int sDofId[] = {0, 2};
      int hDofId[] = {1, 3};
      int dofs[2];
      dofs[0] = yi;
      dofs[1] = yi + 1;
      for(int si = 0; si < 2; si++) {
        for(int ds = 0; ds < dofsPerNode; ++ds) {
          for(int hi = 0; hi < 2; hi++) {
            for(int dh = 0; dh < dofsPerNode; ++dh) {
              MatSetValue(data->Ksh, ((dofs[si]*dofsPerNode) + ds), ((dofs[hi]*dofsPerNode) + dh),
                  stencil[(sDofId[si]*dofsPerNode) + ds][(hDofId[hi]*dofsPerNode) + dh], ADD_VALUES);
              MatSetValue(data->Khs, ((dofs[hi]*dofsPerNode) + dh), ((dofs[si]*dofsPerNode) + ds),
                  stencil[(hDofId[hi]*dofsPerNode) + dh][(sDofId[si]*dofsPerNode) + ds], ADD_VALUES);
            }//end dh
          }//end hi
        }//end ds
      }//end si
    }//end yi

    MatAssemblyBegin(data->Ksh, MAT_FLUSH_ASSEMBLY);
    MatAssemblyBegin(data->Khs, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(data->Ksh, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(data->Khs, MAT_FLUSH_ASSEMBLY);

    MatSetValue(data->Ksh, 0, 1, 0.0, INSERT_VALUES);
    MatSetValue(data->Ksh, 1, 0, 0.0, INSERT_VALUES);
    MatSetValue(data->Ksh, 0, 0, 0.0, INSERT_VALUES);
    MatSetValue(data->Ksh, (N - 1), (N - 2), 0.0, INSERT_VALUES);
    MatSetValue(data->Ksh, (N - 2), (N - 1), 0.0, INSERT_VALUES);
    MatSetValue(data->Ksh, (N - 1), (N - 1), 0.0, INSERT_VALUES);

    MatSetValue(data->Khs, 0, 1, 0.0, INSERT_VALUES);
    MatSetValue(data->Khs, 1, 0, 0.0, INSERT_VALUES);
    MatSetValue(data->Khs, 0, 0, 0.0, INSERT_VALUES);
    MatSetValue(data->Khs, (N - 1), (N - 2), 0.0, INSERT_VALUES);
    MatSetValue(data->Khs, (N - 2), (N - 1), 0.0, INSERT_VALUES);
    MatSetValue(data->Khs, (N - 1), (N - 1), 0.0, INSERT_VALUES);

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
        for(int dj = 0; dj < dofsPerNode; ++dj) {
          for(int i = 0; i < 2; i++) {
            for(int di = 0; di < dofsPerNode; ++di) {
              MatSetValue(data->Khh, ((dofs[j]*dofsPerNode) + dj), ((dofs[i]*dofsPerNode) + di), 
                  stencil[(e1DofId[j]*dofsPerNode) + dj][(e1DofId[i]*dofsPerNode) + di], ADD_VALUES);
              MatSetValue(data->Khh, ((dofs[j]*dofsPerNode) + dj), ((dofs[i]*dofsPerNode) + di),
                  stencil[(e2DofId[j]*dofsPerNode) + dj][(e2DofId[i]*dofsPerNode) + di], ADD_VALUES);
            }//end di
          }//end i
        }//end dj
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
    MatCreateSeqAIJ(PETSC_COMM_SELF, locSize, locSize, numNonZeros, PETSC_NULL, &(data->Kssl));
    MatCreateSeqAIJ(PETSC_COMM_SELF, locSize, locSize, numNonZeros, PETSC_NULL, &(data->Ksl));
    MatCreateSeqAIJ(PETSC_COMM_SELF, locSize, locSize, numNonZeros, PETSC_NULL, &(data->Kls));
    MatCreateSeqAIJ(PETSC_COMM_SELF, locSize, locSize, numNonZeros, PETSC_NULL, &(data->Kll));
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
    MatSetValue(data->Ksl, (N - 1), (N - 2), 0.0, INSERT_VALUES);
    MatSetValue(data->Ksl, (N - 2), (N - 1), 0.0, INSERT_VALUES);
    MatSetValue(data->Ksl, (N - 1), (N - 1), 0.0, INSERT_VALUES);

    MatSetValue(data->Kls, 0, 1, 0.0, INSERT_VALUES);
    MatSetValue(data->Kls, 1, 0, 0.0, INSERT_VALUES);
    MatSetValue(data->Kls, 0, 0, 0.0, INSERT_VALUES);
    MatSetValue(data->Kls, (N - 1), (N - 2), 0.0, INSERT_VALUES);
    MatSetValue(data->Kls, (N - 2), (N - 1), 0.0, INSERT_VALUES);
    MatSetValue(data->Kls, (N - 1), (N - 1), 0.0, INSERT_VALUES);

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
    MatSetValue(data->Kll, (N - 1), (N - 2), 0.0, INSERT_VALUES);
    MatSetValue(data->Kll, (N - 2), (N - 1), 0.0, INSERT_VALUES);
    MatSetValue(data->Kll, (N - 1), (N - 1), 1.0, INSERT_VALUES);

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

  const int dofsPerNode = data->dofsPerNode;
  const int N = data->N;

  int oxs, onx;
  if(rank == 0) {
    oxs = 0;
    onx = N;
  } else {
    oxs = 1;
    onx = N - 1;
  }

  PetscScalar* arr;
  VecGetArray(vec, &arr);

  //Left
  if(rank == 0) {
    for(int yi = 0; yi < N; ++yi) {
      int xi = 0;
      for(int d = 0; d < dofsPerNode; ++d) {
        arr[(((yi*onx) + xi)*dofsPerNode) + d] = 0;
      }//end d
    }//end for yi
  }

  //Right
  if(rank == (npes - 1)) {
    for(int yi = 0; yi < N; ++yi) {
      int xi = onx - 1;
      for(int d = 0; d < dofsPerNode; ++d) {
        arr[(((yi*onx) + xi)*dofsPerNode) + d] = 0;
      }//end d
    }//end for yi
  }

  //Top
  for(int xi = 0; xi < onx; ++xi) {
    int yi = N - 1;
    for(int d = 0; d < dofsPerNode; ++d) {
      arr[(((yi*onx) + xi)*dofsPerNode) + d] = 0;
    }//end d
  }//end for xi

  //Bottom
  for(int xi = 0; xi < onx; ++xi) {
    int yi = 0;
    for(int d = 0; d < dofsPerNode; ++d) {
      arr[(((yi*onx) + xi)*dofsPerNode) + d] = 0;
    }//end d
  }//end for xi

  VecRestoreArray(vec, &arr);
}



