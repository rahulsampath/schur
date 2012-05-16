
#include "mpi.h"
#include <iostream>
#include "schur.h"
#include <cassert>
#include <cmath>
#include <vector>
#include "petscdmmg.h"

extern double stencil[4][4];

void createOuterContext(OuterContext* & ctx) {
  int rank;
  int npes;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &npes);

  ctx = new OuterContext;

  ctx->data = NULL;
  ctx->root = NULL;
  ctx->outerMat = PETSC_NULL;
  ctx->outerKsp = PETSC_NULL;
  ctx->outerPC = PETSC_NULL;
  ctx->outerSol = PETSC_NULL;
  ctx->outerRhs = PETSC_NULL;

  createLocalData(ctx->data);

  createRSDtree(ctx->root, rank, npes);

  createOuterMat(ctx);

  createOuterPC(ctx);

  createOuterKsp(ctx);

  MatGetVecs(ctx->outerMat, &(ctx->outerSol), &(ctx->outerRhs));
}

void destroyOuterContext(OuterContext* ctx) {
  if(ctx->outerSol) {
    VecDestroy(ctx->outerSol);
  }
  if(ctx->outerRhs) {
    VecDestroy(ctx->outerRhs);
  }
  if(ctx->outerKsp) {
    KSPDestroy(ctx->outerKsp);
  }
  if(ctx->outerPC) {
    PCDestroy(ctx->outerPC);
  }
  if(ctx->outerMat) {
    MatDestroy(ctx->outerMat);
  }
  if(ctx->root) {
    destroyRSDtree(ctx->root);
  }
  if(ctx->data) {
    destroyLocalData(ctx->data);
  }
  delete ctx;
}

void createLocalData(LocalData* & data) {
  data = new LocalData;

  data->commAll = MPI_COMM_NULL;
  data->commLow = MPI_COMM_NULL;
  data->commHigh = MPI_COMM_NULL;
  data->Kssl = PETSC_NULL;
  data->Kssh = PETSC_NULL;
  data->Ksl = PETSC_NULL;
  data->Ksh = PETSC_NULL;
  data->Kls = PETSC_NULL;
  data->Khs = PETSC_NULL;
  data->Kll = PETSC_NULL;
  data->Khh = PETSC_NULL;
  data->lowSchurMat = PETSC_NULL;
  data->highSchurMat = PETSC_NULL;
  data->lowSchurKsp = PETSC_NULL;
  data->highSchurKsp = PETSC_NULL;
  data->mgObj = PETSC_NULL;

  data->dofsPerNode = 1;
  data->N = 17;
  PetscOptionsGetInt(PETSC_NULL, "-N", &(data->N), PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-D", &(data->dofsPerNode), PETSC_NULL);

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if(!rank) {
    std::cout<<"N = "<<(data->N)<<std::endl;
    std::cout<<"D = "<<(data->dofsPerNode)<<std::endl;
  }

  data->commAll = PETSC_COMM_WORLD;

  createLowAndHighComms(data);

  createMG(data);

  createLocalMatrices(data);

  createSchurMat(data);

  createInnerKsp(data);
}

void destroyLocalData(LocalData* data) {
  if(data->lowSchurKsp) {
    KSPDestroy(data->lowSchurKsp);
  }
  if(data->highSchurKsp) {
    KSPDestroy(data->highSchurKsp);
  }
  if(data->lowSchurMat) {
    MatDestroy(data->lowSchurMat);
  }
  if(data->highSchurMat) {
    MatDestroy(data->highSchurMat);
  }
  if(data->Kssl) {
    MatDestroy(data->Kssl);
  }
  if(data->Kssh) {
    MatDestroy(data->Kssh);
  }
  if(data->Ksl) {
    MatDestroy(data->Ksl);
  }
  if(data->Ksh) {
    MatDestroy(data->Ksh);
  }
  if(data->Kls) {
    MatDestroy(data->Kls);
  }
  if(data->Khs) {
    MatDestroy(data->Khs);
  }
  if(data->Kll) {
    MatDestroy(data->Kll);
  }
  if(data->Khh) {
    MatDestroy(data->Khh);
  }
  if(data->mgObj) {
    DMMGDestroy(data->mgObj);
  }
  if(data->commLow != MPI_COMM_NULL) {
    MPI_Comm_free(&(data->commLow));
  }
  if(data->commHigh != MPI_COMM_NULL) {
    MPI_Comm_free(&(data->commHigh));
  }
  delete data;
}

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

void destroyRSDtree(RSDnode *root) {
  if(root->child) {
    destroyRSDtree(root->child);
  }
  delete root;  
}

void createLowAndHighComms(LocalData* data) {
  int rank, npes;
  MPI_Comm_rank(data->commAll, &rank);
  MPI_Comm_size(data->commAll, &npes);

  MPI_Group groupAll;
  MPI_Comm_group(data->commAll, &groupAll);

  if((rank%2) == 0) {
    if(rank < (npes - 1)) {
      int lowRanks[2];
      lowRanks[0] = rank;
      lowRanks[1] = rank + 1;
      MPI_Group lowGroup;
      MPI_Group_incl(groupAll, 2, lowRanks, &lowGroup);
      MPI_Comm_create(data->commAll, lowGroup, &(data->commLow));
      MPI_Group_free(&lowGroup);
    } else {
      MPI_Comm_create(data->commAll, MPI_GROUP_EMPTY, &(data->commLow));
      assert((data->commLow) == MPI_COMM_NULL);
    }
  } else {
    int highRanks[2];
    highRanks[0] = rank - 1;
    highRanks[1] = rank;
    MPI_Group highGroup;
    MPI_Group_incl(groupAll, 2, highRanks, &highGroup);
    MPI_Comm_create(data->commAll, highGroup, &(data->commHigh));
    MPI_Group_free(&highGroup);
  }

  if((rank%2) == 0) {
    if(rank > 0) {
      int highRanks[2];
      highRanks[0] = rank - 1;
      highRanks[1] = rank;
      MPI_Group highGroup;
      MPI_Group_incl(groupAll, 2, highRanks, &highGroup);
      MPI_Comm_create(data->commAll, highGroup, &(data->commHigh));
      MPI_Group_free(&highGroup);
    } else {
      MPI_Comm_create(data->commAll, MPI_GROUP_EMPTY, &(data->commHigh));
      assert((data->commHigh) == MPI_COMM_NULL);
    }
  } else {
    if(rank < (npes - 1)) {
      int lowRanks[2];
      lowRanks[0] = rank;
      lowRanks[1] = rank + 1;
      MPI_Group lowGroup;
      MPI_Group_incl(groupAll, 2, lowRanks, &lowGroup);
      MPI_Comm_create(data->commAll, lowGroup, &(data->commLow));
      MPI_Group_free(&lowGroup);
    } else {
      MPI_Comm_create(data->commAll, MPI_GROUP_EMPTY, &(data->commLow));
      assert((data->commLow) == MPI_COMM_NULL);
    }
  }

  MPI_Group_free(&groupAll);
}

void createOuterKsp(OuterContext* ctx) {
  KSPCreate((ctx->data)->commAll, &(ctx->outerKsp));
  PetscObjectIncrementTabLevel((PetscObject)(ctx->outerKsp), PETSC_NULL, 0);
  KSPSetType(ctx->outerKsp, KSPFGMRES);
  KSPSetTolerances(ctx->outerKsp, 1.0e-12, 1.0e-12, PETSC_DEFAULT, 50);
  KSPSetPC(ctx->outerKsp, ctx->outerPC);
  KSPSetOptionsPrefix(ctx->outerKsp, "outer_");
  KSPSetFromOptions(ctx->outerKsp);
  KSPSetOperators(ctx->outerKsp, ctx->outerMat,
      ctx->outerMat, SAME_NONZERO_PATTERN);
  KSPSetUp(ctx->outerKsp);
}

void createOuterMat(OuterContext* ctx) {
  int rank;
  MPI_Comm_rank(((ctx->data)->commAll), &rank);

  int onx;
  if(rank == 0) {
    onx = (ctx->data)->N;
  } else {
    onx = ((ctx->data)->N) - 1;
  }

  int locSize = (onx*((ctx->data)->N));

  Mat mat;
  MatCreateShell(((ctx->data)->commAll), locSize, locSize,
      PETSC_DETERMINE, PETSC_DETERMINE, ctx, &mat);
  MatShellSetOperation(mat, MATOP_MULT, (void(*)(void))(&outerMatMult));

  ctx->outerMat = mat;
}

void createSchurMat(LocalData* data) {
  int rank, npes;
  MPI_Comm_rank((data->commAll), &rank);
  MPI_Comm_size((data->commAll), &npes);

  Mat lowMat = PETSC_NULL;
  Mat highMat = PETSC_NULL;

  if((rank%2) == 0) {
    if(rank < (npes - 1)) {
      MatCreateShell((data->commLow), (data->N), (data->N),
          PETSC_DETERMINE, PETSC_DETERMINE, data, &lowMat);
    }
  } else {
    MatCreateShell((data->commHigh), 0, 0,
        PETSC_DETERMINE, PETSC_DETERMINE, data, &highMat);
  }

  if((rank%2) == 0) {
    if(rank > 0) {
      MatCreateShell((data->commHigh), 0, 0,
          PETSC_DETERMINE, PETSC_DETERMINE, data, &highMat);
    }
  } else {
    if(rank < (npes - 1)) {
      MatCreateShell((data->commLow), (data->N), (data->N),
          PETSC_DETERMINE, PETSC_DETERMINE, data, &lowMat);
    }
  }

  if(lowMat) {
    MatShellSetOperation(lowMat, MATOP_MULT, (void(*)(void))(&lowSchurMatMult));
  }

  if(highMat) {
    MatShellSetOperation(highMat, MATOP_MULT, (void(*)(void))(&highSchurMatMult));
  }

  data->lowSchurMat = lowMat;
  data->highSchurMat = highMat;
}

void createMG(LocalData* data) {
  assert(data != NULL);
  assert(data->N >= 16);
  int nlevels = (std::floor(log2(data->N))) - 2;
  assert((8<<(nlevels - 1)) == ((data->N) - 1));

  DMMGCreate(PETSC_COMM_SELF, -nlevels, PETSC_NULL, &(data->mgObj));
  DMMGSetOptionsPrefix(data->mgObj, "loc_");

  DA da;
  DACreate2d(PETSC_COMM_SELF, DA_NONPERIODIC, DA_STENCIL_BOX, 9, 9,
      PETSC_DECIDE, PETSC_DECIDE, 1, 1, PETSC_NULL, PETSC_NULL, &da);
  DMMGSetDM((data->mgObj), (DM)da);
  DADestroy(da);

  DMMGSetKSP((data->mgObj), PETSC_NULL, &computeMGmatrix);
  PetscObjectIncrementTabLevel((PetscObject)(DMMGGetKSP(data->mgObj)), PETSC_NULL, 2);
}

void createOuterPC(OuterContext* ctx) {
  PCCreate(((ctx->data)->commAll), &(ctx->outerPC));
  PCSetType(ctx->outerPC, PCSHELL);
  PCShellSetName(ctx->outerPC, "RSD");
  PCShellSetContext(ctx->outerPC, ctx);
  PCShellSetApply(ctx->outerPC, &outerPCapply);
}

PetscErrorCode computeMGmatrix(DMMG dmmg, Mat J, Mat B) {
  assert(J == B);

  DA da = (DA)(dmmg->dm);

  int N;
  DAGetInfo(da, PETSC_NULL, &N, PETSC_NULL, PETSC_NULL, 
      PETSC_NULL, PETSC_NULL, PETSC_NULL, 
      PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

  MatZeroEntries(J);

  int Ne = N - 1;
  for(int yi = 0; yi < Ne; ++yi) {
    for(int xi = 0; xi < Ne; ++xi) {
      int dofs[4];
      dofs[0] = (yi*N) + xi;
      dofs[1] = (yi*N) + xi + 1;
      dofs[2] = ((yi + 1)*N) + xi;
      dofs[3] = ((yi + 1)*N) + xi + 1;
      for(int j = 0; j < 4; j++) {
        for(int i = 0; i < 4; i++) {
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
    int bndDof = (yi*N) + xi;
    int nhDofs[] = {-1, -1, -1, -1, -1, -1, -1, -1};
    if(yi > 0) {
      if(xi > 0) {
        nhDofs[0] = ((yi - 1)*N) + xi - 1;
      }
      nhDofs[1] = ((yi - 1)*N) + xi;
      if(xi < (N - 1)) {
        nhDofs[2] = ((yi - 1)*N) + xi + 1;
      }
    }
    if(xi > 0) {
      nhDofs[3] = (yi*N) + xi - 1;
    }
    if(xi < (N - 1)) {
      nhDofs[4] = (yi*N) + xi + 1;
    }
    if(yi < (N - 1)) {
      if(xi > 0) {
        nhDofs[5] = ((yi + 1)*N) + xi - 1;
      }
      nhDofs[6] = ((yi + 1)*N) + xi;   
      if(xi < (N - 1)) {
        nhDofs[7] = ((yi + 1)*N) + xi + 1;
      }
    }
    for(int i = 0; i < 8; ++i) {
      if(nhDofs[i] != -1) {
        MatSetValue(J, bndDof, nhDofs[i], 0.0, INSERT_VALUES);
        MatSetValue(J, nhDofs[i], bndDof, 0.0, INSERT_VALUES);
      }
    }//end i
    MatSetValue(J, bndDof, bndDof, 1.0, INSERT_VALUES);
  }//end yi

  //Right
  for(int yi = 0; yi < N; ++yi) {
    int xi = (N - 1);
    int bndDof = (yi*N) + xi;
    int nhDofs[] = {-1, -1, -1, -1, -1, -1, -1, -1};
    if(yi > 0) {
      if(xi > 0) {
        nhDofs[0] = ((yi - 1)*N) + xi - 1;
      }
      nhDofs[1] = ((yi - 1)*N) + xi;
      if(xi < (N - 1)) {
        nhDofs[2] = ((yi - 1)*N) + xi + 1;
      }
    }
    if(xi > 0) {
      nhDofs[3] = (yi*N) + xi - 1;
    }
    if(xi < (N - 1)) {
      nhDofs[4] = (yi*N) + xi + 1;
    }
    if(yi < (N - 1)) {
      if(xi > 0) {
        nhDofs[5] = ((yi + 1)*N) + xi - 1;
      }
      nhDofs[6] = ((yi + 1)*N) + xi;   
      if(xi < (N - 1)) {
        nhDofs[7] = ((yi + 1)*N) + xi + 1;
      }
    }
    for(int i = 0; i < 8; ++i) {
      if(nhDofs[i] != -1) {
        MatSetValue(J, bndDof, nhDofs[i], 0.0, INSERT_VALUES);
        MatSetValue(J, nhDofs[i], bndDof, 0.0, INSERT_VALUES);
      }
    }//end i
    MatSetValue(J, bndDof, bndDof, 1.0, INSERT_VALUES);
  }//end yi

  //Top
  for(int xi = 0; xi < N; ++xi) {
    int yi = (N - 1);
    int bndDof = (yi*N) + xi;
    int nhDofs[] = {-1, -1, -1, -1, -1, -1, -1, -1};
    if(yi > 0) {
      if(xi > 0) {
        nhDofs[0] = ((yi - 1)*N) + xi - 1;
      }
      nhDofs[1] = ((yi - 1)*N) + xi;
      if(xi < (N - 1)) {
        nhDofs[2] = ((yi - 1)*N) + xi + 1;
      }
    }
    if(xi > 0) {
      nhDofs[3] = (yi*N) + xi - 1;
    }
    if(xi < (N - 1)) {
      nhDofs[4] = (yi*N) + xi + 1;
    }
    if(yi < (N - 1)) {
      if(xi > 0) {
        nhDofs[5] = ((yi + 1)*N) + xi - 1;
      }
      nhDofs[6] = ((yi + 1)*N) + xi;   
      if(xi < (N - 1)) {
        nhDofs[7] = ((yi + 1)*N) + xi + 1;
      }
    }
    for(int i = 0; i < 8; ++i) {
      if(nhDofs[i] != -1) {
        MatSetValue(J, bndDof, nhDofs[i], 0.0, INSERT_VALUES);
        MatSetValue(J, nhDofs[i], bndDof, 0.0, INSERT_VALUES);
      }
    }//end i
    MatSetValue(J, bndDof, bndDof, 1.0, INSERT_VALUES);
  }//end xi

  //Bottom
  for(int xi = 0; xi < N; ++xi) {
    int yi = 0;
    int bndDof = (yi*N) + xi;
    int nhDofs[] = {-1, -1, -1, -1, -1, -1, -1, -1};
    if(yi > 0) {
      if(xi > 0) {
        nhDofs[0] = ((yi - 1)*N) + xi - 1;
      }
      nhDofs[1] = ((yi - 1)*N) + xi;
      if(xi < (N - 1)) {
        nhDofs[2] = ((yi - 1)*N) + xi + 1;
      }
    }
    if(xi > 0) {
      nhDofs[3] = (yi*N) + xi - 1;
    }
    if(xi < (N - 1)) {
      nhDofs[4] = (yi*N) + xi + 1;
    }
    if(yi < (N - 1)) {
      if(xi > 0) {
        nhDofs[5] = ((yi + 1)*N) + xi - 1;
      }
      nhDofs[6] = ((yi + 1)*N) + xi;   
      if(xi < (N - 1)) {
        nhDofs[7] = ((yi + 1)*N) + xi + 1;
      }
    }
    for(int i = 0; i < 8; ++i) {
      if(nhDofs[i] != -1) {
        MatSetValue(J, bndDof, nhDofs[i], 0.0, INSERT_VALUES);
        MatSetValue(J, nhDofs[i], bndDof, 0.0, INSERT_VALUES);
      }
    }//end i
    MatSetValue(J, bndDof, bndDof, 1.0, INSERT_VALUES);
  }//end xi

  MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);

  return 0;
}

PetscErrorCode lowSchurMatMult(Mat mat, Vec in, Vec out) {
  LocalData* data;
  MatShellGetContext(mat, (void**)(&data));

  Vec inSeq, outSeq;
  VecCreateSeq(PETSC_COMM_SELF, (data->N), &inSeq);
  VecDuplicate(inSeq, &outSeq);

  PetscScalar *inArr;
  PetscScalar *outArr;

  VecGetArray(in, &inArr);
  VecGetArray(out, &outArr);

  VecPlaceArray(inSeq, inArr);
  VecPlaceArray(outSeq, outArr);

  schurMatVec(data, true, inSeq, outSeq);

  VecResetArray(inSeq);
  VecResetArray(outSeq);

  VecRestoreArray(in, &inArr);
  VecRestoreArray(out, &outArr);

  VecDestroy(inSeq);
  VecDestroy(outSeq);

  return 0;
}

PetscErrorCode highSchurMatMult(Mat mat, Vec in, Vec out) {
  LocalData* data;
  MatShellGetContext(mat, (void**)(&data));

  schurMatVec(data, false, PETSC_NULL, PETSC_NULL);

  return 0;
}

PetscErrorCode outerMatMult(Mat mat, Vec in, Vec out) {
  OuterContext* ctx;
  MatShellGetContext(mat, (void**)(&ctx));

  PetscInt localSize;
  VecGetLocalSize(in, &localSize);

  Vec inSeq, outSeq;
  VecCreateSeq(PETSC_COMM_SELF, localSize, &inSeq);
  VecDuplicate(inSeq, &outSeq);

  PetscScalar *inArr;
  PetscScalar *outArr;

  VecGetArray(in, &inArr);
  VecGetArray(out, &outArr);

  VecPlaceArray(inSeq, inArr);
  VecPlaceArray(outSeq, outArr);

  KmatVec((ctx->data), (ctx->root), inSeq, outSeq);

  VecResetArray(inSeq);
  VecResetArray(outSeq);

  VecRestoreArray(in, &inArr);
  VecRestoreArray(out, &outArr);

  VecDestroy(inSeq);
  VecDestroy(outSeq);

  return 0;
}

PetscErrorCode outerPCapply(void* ptr, Vec in, Vec out) {
  OuterContext* ctx = static_cast<OuterContext*>(ptr);

  PetscInt localSize;
  VecGetLocalSize(in, &localSize);

  Vec inSeq, outSeq;
  VecCreateSeq(PETSC_COMM_SELF, localSize, &inSeq);
  VecDuplicate(inSeq, &outSeq);

  PetscScalar *inArr;
  PetscScalar *outArr;

  VecGetArray(in, &inArr);
  VecGetArray(out, &outArr);

  VecPlaceArray(inSeq, inArr);
  VecPlaceArray(outSeq, outArr);

  RSDapplyInverse((ctx->data), (ctx->root), inSeq, outSeq);

  VecResetArray(inSeq);
  VecResetArray(outSeq);

  VecRestoreArray(in, &inArr);
  VecRestoreArray(out, &outArr);

  VecDestroy(inSeq);
  VecDestroy(outSeq);

  return 0;
}

void createInnerKsp(LocalData* data) {
  int rank, npes;
  MPI_Comm_rank(data->commAll, &rank);
  MPI_Comm_size(data->commAll, &npes);

  if((rank%2) == 0) {
    if(rank < (npes - 1)) {
      KSPCreate(data->commLow, &(data->lowSchurKsp));
      PetscObjectIncrementTabLevel((PetscObject)(data->lowSchurKsp), PETSC_NULL, 1);
      KSPSetType(data->lowSchurKsp, KSPFGMRES);
      KSPSetTolerances(data->lowSchurKsp, 1.0e-12, 1.0e-12, PETSC_DEFAULT, 2);
      KSPSetOptionsPrefix(data->lowSchurKsp, "inner_");
      PC pc;
      KSPGetPC(data->lowSchurKsp, &pc);
      PCSetType(pc, PCNONE);
      KSPSetFromOptions(data->lowSchurKsp);
      KSPSetOperators(data->lowSchurKsp, data->lowSchurMat,
          data->lowSchurMat, SAME_NONZERO_PATTERN);
      KSPSetUp(data->lowSchurKsp);
    } else {
      data->lowSchurKsp = PETSC_NULL;
    }
  } else {
    KSPCreate(data->commHigh, &(data->highSchurKsp));
    PetscObjectIncrementTabLevel((PetscObject)(data->highSchurKsp), PETSC_NULL, 1);
    KSPSetType(data->highSchurKsp, KSPFGMRES);
    KSPSetTolerances(data->highSchurKsp, 1.0e-12, 1.0e-12, PETSC_DEFAULT, 2);
    KSPSetOptionsPrefix(data->highSchurKsp, "inner_");
    PC pc;
    KSPGetPC(data->highSchurKsp, &pc);
    PCSetType(pc, PCNONE);
    KSPSetFromOptions(data->highSchurKsp);
    KSPSetOperators(data->highSchurKsp, data->highSchurMat,
        data->highSchurMat, SAME_NONZERO_PATTERN);
    KSPSetUp(data->highSchurKsp);
  }

  if((rank%2) == 0) {
    if(rank > 0) {
      KSPCreate(data->commHigh, &(data->highSchurKsp));
      PetscObjectIncrementTabLevel((PetscObject)(data->highSchurKsp), PETSC_NULL, 1);
      KSPSetType(data->highSchurKsp, KSPFGMRES);
      KSPSetTolerances(data->highSchurKsp, 1.0e-12, 1.0e-12, PETSC_DEFAULT, 2);
      KSPSetOptionsPrefix(data->highSchurKsp, "inner_");
      PC pc;
      KSPGetPC(data->highSchurKsp, &pc);
      PCSetType(pc, PCNONE);
      KSPSetFromOptions(data->highSchurKsp);
      KSPSetOperators(data->highSchurKsp, data->highSchurMat,
          data->highSchurMat, SAME_NONZERO_PATTERN);
      KSPSetUp(data->highSchurKsp);
    } else {
      data->highSchurKsp = PETSC_NULL;
    }
  } else {
    if(rank < (npes - 1)) {
      KSPCreate(data->commLow, &(data->lowSchurKsp));
      PetscObjectIncrementTabLevel((PetscObject)(data->lowSchurKsp), PETSC_NULL, 1);
      KSPSetType(data->lowSchurKsp, KSPFGMRES);
      KSPSetTolerances(data->lowSchurKsp, 1.0e-12, 1.0e-12, PETSC_DEFAULT, 2);
      KSPSetOptionsPrefix(data->lowSchurKsp, "inner_");
      PC pc;
      KSPGetPC(data->lowSchurKsp, &pc);
      PCSetType(pc, PCNONE);
      KSPSetFromOptions(data->lowSchurKsp);
      KSPSetOperators(data->lowSchurKsp, data->lowSchurMat,
          data->lowSchurMat, SAME_NONZERO_PATTERN);
      KSPSetUp(data->lowSchurKsp);
    } else {
      data->lowSchurKsp = PETSC_NULL;
    }
  }

}

void RSDapplyInverse(LocalData* data, RSDnode* root, Vec f, Vec u) {
  if(root->child) {
    if(root->rankForCurrLevel == (((root->npesForCurrLevel)/2) - 1)) {
      Vec fL, fStarHigh;

      MatGetVecs(data->Ksl, &fL, &fStarHigh);

      PetscScalar* recvArr7;
      VecGetArray(fStarHigh, &recvArr7);

      PetscInt Ssize;
      VecGetSize(fStarHigh, &Ssize);

      MPI_Request recvRequest7;
      MPI_Irecv(recvArr7, Ssize, MPI_DOUBLE, 1, 7, data->commLow, &recvRequest7);

      Vec fStar, gS, uS, gL;
      VecDuplicate(fStarHigh, &fStar);
      VecDuplicate(fStar, &gS);
      VecDuplicate(fStar, &uS);
      VecDuplicate(fL, &gL);

      Vec fTmp;
      VecDuplicate(u, &fTmp);

      RSDapplyInverse(data, root->child, f, fTmp);

      map<O, L>(data, fTmp, fL);

      map<O, S>(data, f, gS);

      MatMult(data->Ksl, fL, fStar);

      MPI_Status recvStatus7;
      MPI_Wait(&recvRequest7, &recvStatus7);

      VecRestoreArray(fStarHigh, &recvArr7);

      VecAXPBYPCZ(gS, -1.0, -1.0, 1.0, fStar, fStarHigh);

      schurSolve(data, true, gS, uS);

      PetscScalar* sendArr8;
      VecGetArray(uS, &sendArr8);

      MPI_Request sendRequest8;
      MPI_Isend(sendArr8, Ssize, MPI_DOUBLE, 1, 8, data->commLow, &sendRequest8);

      MatMult(data->Kls, uS, gL);

      Vec gRhs = DMMGGetRHS(data->mgObj);
      Vec gSol = DMMGGetx(data->mgObj);

      VecZeroEntries(gRhs);
      map<L, MG>(data, gL, gRhs);

      KSPSolve(DMMGGetKSP(data->mgObj), gRhs, gSol);

      VecZeroEntries(u);
      map<MG, O>(data, gSol, u);

      VecScale(u, -1.0);
      VecAXPY(u, 1.0, fTmp);

      map<S, O>(data, uS, u);

      VecDestroy(gL);
      VecDestroy(fTmp);
      VecDestroy(fL);
      VecDestroy(fStar);
      VecDestroy(fStarHigh);
      VecDestroy(gS);

      MPI_Status sendStatus8;
      MPI_Wait(&sendRequest8, &sendStatus8);

      VecRestoreArray(uS, &sendArr8);

      VecDestroy(uS);
    } else if(root->rankForCurrLevel == ((root->npesForCurrLevel)/2)) {
      Vec fTmp;
      VecDuplicate(u, &fTmp);

      RSDapplyInverse(data, root->child, f, fTmp);

      Vec fH, fStar; 
      MatGetVecs(data->Ksh, &fH, &fStar);

      map<O, H>(data, fTmp, fH);

      MatMult(data->Ksh, fH, fStar);

      PetscInt Ssize;
      VecGetSize(fStar, &Ssize);

      PetscScalar* sendArr7;
      VecGetArray(fStar, &sendArr7);

      MPI_Request sendRequest7;
      MPI_Isend(sendArr7, Ssize, MPI_DOUBLE, 0, 7, data->commHigh, &sendRequest7);

      Vec uS, gH;
      VecDuplicate(fStar, &uS);
      VecDuplicate(fH, &gH);

      schurSolve(data, false, PETSC_NULL, PETSC_NULL);

      MPI_Status sendStatus7;
      MPI_Wait(&sendRequest7, &sendStatus7);

      VecRestoreArray(fStar, &sendArr7);

      PetscScalar* recvArr8;
      VecGetArray(uS, &recvArr8);

      MPI_Request recvRequest8;
      MPI_Irecv(recvArr8, Ssize, MPI_DOUBLE, 0, 8, data->commHigh, &recvRequest8);

      Vec gRhs = DMMGGetRHS(data->mgObj);
      Vec gSol = DMMGGetx(data->mgObj);

      VecZeroEntries(gRhs);
      VecZeroEntries(u);

      MPI_Status recvStatus8;
      MPI_Wait(&recvRequest8, &recvStatus8);

      VecRestoreArray(uS, &recvArr8);

      MatMult(data->Khs, uS, gH);

      map<H, MG>(data, gH, gRhs);

      KSPSolve(DMMGGetKSP(data->mgObj), gRhs, gSol);

      map<MG, O>(data, gSol, u);
      VecScale(u, -1.0);
      VecAXPY(u, 1.0, fTmp);

      VecDestroy(gH);
      VecDestroy(fTmp);
      VecDestroy(fH);
      VecDestroy(fStar);
      VecDestroy(uS);
    } else {
      RSDapplyInverse(data, root->child, f, u);
    }
  } else {
    Vec fMg = DMMGGetRHS(data->mgObj);
    Vec uMg = DMMGGetx(data->mgObj);

    VecZeroEntries(fMg);
    map<O, MG>(data, f, fMg);

    KSPSolve(DMMGGetKSP(data->mgObj), fMg, uMg);

    VecZeroEntries(u);
    map<MG, O>(data, uMg, u);
  }
}

void KmatVec(LocalData* data, RSDnode* root, Vec uIn, Vec uOut) {
  if(root->child) {
    if(root->rankForCurrLevel == (((root->npesForCurrLevel)/2) - 1)) {
      Vec uS, wS;
      MatGetVecs(data->Kssl, &uS, &wS);

      map<O, S>(data, uIn, uS);

      PetscInt Ssize;
      VecGetSize(uS, &Ssize);

      PetscScalar* arr;
      VecGetArray(uS, &arr);

      MPI_Request sendRequest5;
      MPI_Isend(arr, Ssize, MPI_DOUBLE, 1, 5, data->commLow, &sendRequest5);
      MPI_Status sendStatus5;
      MPI_Wait(&sendRequest5, &sendStatus5);

      VecRestoreArray(uS, &arr);

      Vec uL, bS, cL, yS;
      MatGetVecs(data->Ksl, &uL, &bS);
      VecDuplicate(uL, &cL); 
      VecDuplicate(bS, &yS); 

      map<O, L>(data, uIn, uL);

      MatMult(data->Kssl, uS, wS);

      MatMult(data->Ksl, uL, bS);

      MatMult(data->Kls, uS, cL);

      VecWAXPY(yS, 1.0, wS, bS);

      KmatVec(data, root->child, uIn, uOut);

      Vec cO;
      VecDuplicate(uOut, &cO);

      VecZeroEntries(cO);
      map<L, O>(data, cL, cO);

      VecAXPY(uOut, 1.0, cO);

      Vec uSout;
      VecDuplicate(uS, &uSout);
      VecGetArray(uSout, &arr);

      MPI_Request recvRequest6;
      MPI_Irecv(arr, Ssize, MPI_DOUBLE, 1, 6, data->commLow, &recvRequest6);
      MPI_Status recvStatus6;
      MPI_Wait(&recvRequest6, &recvStatus6);

      VecRestoreArray(uSout, &arr);

      VecAXPY(uSout, 1.0, yS);

      map<S, O>(data, uSout, uOut);

      VecDestroy(uSout);
      VecDestroy(cO);
      VecDestroy(cL);
      VecDestroy(uL);
      VecDestroy(bS);
      VecDestroy(yS);
      VecDestroy(uS);
      VecDestroy(wS);
    } else if(root->rankForCurrLevel == ((root->npesForCurrLevel)/2)) {
      Vec uS, wS;
      MatGetVecs(data->Kssh, &uS, &wS);

      PetscInt Ssize;
      VecGetSize(uS, &Ssize);

      PetscScalar *arr;
      VecGetArray(uS, &arr);

      MPI_Request recvRequest5;
      MPI_Irecv(arr, Ssize, MPI_DOUBLE, 0, 5, data->commHigh, &recvRequest5);
      MPI_Status recvStatus5;
      MPI_Wait(&recvRequest5, &recvStatus5);

      VecRestoreArray(uS, &arr);

      Vec uH, bS, cH, yS;
      MatGetVecs(data->Ksh, &uH, &bS);
      VecDuplicate(uH, &cH); 
      VecDuplicate(bS, &yS); 

      map<O, H>(data, uIn, uH);

      MatMult(data->Kssh, uS, wS);

      MatMult(data->Ksh, uH, bS);

      MatMult(data->Khs, uS, cH);

      VecWAXPY(yS, 1.0, wS, bS);

      KmatVec(data, root->child, uIn, uOut);

      Vec cO;
      VecDuplicate(uOut, &cO);

      VecZeroEntries(cO);
      map<H, O>(data, cH, cO);

      VecAXPY(uOut, 1.0, cO);

      VecGetArray(yS, &arr);

      MPI_Request sendRequest6;
      MPI_Isend(arr, Ssize, MPI_DOUBLE, 0, 6, data->commHigh, &sendRequest6);
      MPI_Status sendStatus6;
      MPI_Wait(&sendRequest6, &sendStatus6);

      VecRestoreArray(yS, &arr);

      VecDestroy(cO);
      VecDestroy(cH);
      VecDestroy(uH);
      VecDestroy(bS);
      VecDestroy(yS);
      VecDestroy(uS);
      VecDestroy(wS);
    } else {
      KmatVec(data, root->child, uIn, uOut);
    }
  } else {
    Vec uInMg, uOutMg;
    VecDuplicate(DMMGGetx(data->mgObj), &uInMg);
    VecDuplicate(uInMg, &uOutMg);

    VecZeroEntries(uInMg);
    map<O, MG>(data, uIn, uInMg);

    MatMult(DMMGGetJ(data->mgObj), uInMg, uOutMg);

    VecZeroEntries(uOut);
    map<MG, O>(data, uOutMg, uOut);

    VecDestroy(uInMg);
    VecDestroy(uOutMg);
  }
}

void schurMatVec(LocalData* data, bool isLow, Vec uSin, Vec uSout) {
  if(isLow) {
    PetscInt Ssize;
    VecGetSize(uSin, &Ssize);

    PetscScalar* arr;
    VecGetArray(uSin, &arr);

    MPI_Request sendRequest3;
    MPI_Isend(arr, Ssize, MPI_DOUBLE, 1, 3, data->commLow, &sendRequest3);
    MPI_Status sendStatus3;
    MPI_Wait(&sendRequest3, &sendStatus3);

    VecRestoreArray(uSin, &arr);

    Vec uL;
    MatGetVecs(data->Kssl, PETSC_NULL, &uL);

    Vec vL;
    MatGetVecs(data->Kls, PETSC_NULL, &vL);

    Vec wL, wS;
    MatGetVecs(data->Ksl, &wL, &wS);

    Vec uStarL;
    VecDuplicate(uL, &uStarL);

    MatMult(data->Kssl, uSin, uL);

    MatMult(data->Kls, uSin, vL);

    Vec rhsMg = DMMGGetRHS(data->mgObj);
    Vec solMg = DMMGGetx(data->mgObj);

    VecZeroEntries(rhsMg);
    map<L, MG>(data, vL, rhsMg);

    KSPSolve(DMMGGetKSP(data->mgObj), rhsMg, solMg);

    map<MG, L>(data, solMg, wL);

    MatMult(data->Ksl, wL, wS);

    VecWAXPY(uStarL, -1.0, wS, uL);

    VecGetArray(uSout, &arr);

    MPI_Request recvRequest4;
    MPI_Irecv(arr, Ssize, MPI_DOUBLE, 1, 4, data->commLow, &recvRequest4);
    MPI_Status recvStatus4;
    MPI_Wait(&recvRequest4, &recvStatus4);

    VecRestoreArray(uSout, &arr);

    VecAXPY(uSout, 1.0, uStarL);

    VecDestroy(uL);
    VecDestroy(vL);
    VecDestroy(wL);
    VecDestroy(wS);
    VecDestroy(uStarL);
  } else {
    Vec uSinCopy, uH;
    MatGetVecs(data->Kssh, &uSinCopy, &uH);

    PetscInt Ssize;
    VecGetSize(uSinCopy, &Ssize);

    PetscScalar* arr;
    VecGetArray(uSinCopy, &arr);

    MPI_Request recvRequest3;
    MPI_Irecv(arr, Ssize, MPI_DOUBLE, 0, 3, data->commHigh, &recvRequest3);
    MPI_Status recvStatus3;
    MPI_Wait(&recvRequest3, &recvStatus3);

    VecRestoreArray(uSinCopy, &arr);

    Vec vH;
    MatGetVecs(data->Khs, PETSC_NULL, &vH);

    Vec wH, wS;
    MatGetVecs(data->Ksh, &wH, &wS);

    Vec uStarH;
    VecDuplicate(uH, &uStarH);

    MatMult(data->Kssh, uSinCopy, uH);

    MatMult(data->Khs, uSinCopy, vH);

    Vec rhsMg = DMMGGetRHS(data->mgObj);
    Vec solMg = DMMGGetx(data->mgObj);

    VecZeroEntries(rhsMg);
    map<H, MG>(data, vH, rhsMg);

    KSPSolve(DMMGGetKSP(data->mgObj), rhsMg, solMg);

    map<MG, H>(data, solMg, wH);

    MatMult(data->Ksh, wH, wS);

    VecWAXPY(uStarH, -1.0, wS, uH);

    VecGetArray(uStarH, &arr);

    MPI_Request sendRequest4;
    MPI_Isend(arr, Ssize, MPI_DOUBLE, 0, 4, data->commHigh, &sendRequest4);
    MPI_Status sendStatus4;
    MPI_Wait(&sendRequest4, &sendStatus4);

    VecRestoreArray(uStarH, &arr);

    VecDestroy(uSinCopy);
    VecDestroy(uH);
    VecDestroy(vH);
    VecDestroy(wH);
    VecDestroy(wS);
    VecDestroy(uStarH);
  }
}

void schurSolve(LocalData* data, bool isLow, Vec rhs, Vec sol) {
  Vec rhsKsp, solKsp;

  if(isLow) {
    PetscInt localSize;
    VecGetLocalSize(rhs, &localSize);
    VecCreateMPI(data->commLow, localSize, PETSC_DETERMINE, &rhsKsp);
  } else {
    VecCreateMPI(data->commHigh, 0, PETSC_DETERMINE, &rhsKsp);
  }
  VecDuplicate(rhsKsp, &solKsp);

  PetscScalar *rhsArr;
  PetscScalar *solArr;

  if(isLow) {
    VecGetArray(rhs, &rhsArr);
    VecGetArray(sol, &solArr);

    VecPlaceArray(rhsKsp, rhsArr);
    VecPlaceArray(solKsp, solArr);
  }

  if(isLow) {
    KSPSolve(data->lowSchurKsp, rhsKsp, solKsp);
  } else {
    KSPSolve(data->highSchurKsp, rhsKsp, solKsp);
  }

  if(isLow) {
    VecResetArray(rhsKsp);
    VecResetArray(solKsp);

    VecRestoreArray(rhs, &rhsArr);
    VecRestoreArray(sol, &solArr);
  }

  VecDestroy(rhsKsp);
  VecDestroy(solKsp);
}

double dPhidPsi(int i, double eta) {
  if(i == 0) {
    return (-(1.0 - eta)/4.0);
  } else if(i == 1) {
    return ((1.0 - eta)/4.0);
  } else if(i == 2) {
    return (-(1.0 + eta)/4.0);
  } else if(i == 3) {
    return ((1.0 + eta)/4.0);
  } else {
    assert(false);
  }
  return 0;
}

double dPhidEta(int i, double psi) {
  if(i == 0) {
    return (-(1.0 - psi)/4.0);
  } else if(i == 1) {
    return (-(1.0 + psi)/4.0);
  } else if(i == 2) {
    return ((1.0 - psi)/4.0);
  } else if(i == 3) {
    return ((1.0 + psi)/4.0);
  } else {
    assert(false);
  }
  return 0;

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

void computeStencil() {
  const double gaussPts[] = { (1.0/sqrt(3.0)), (-1.0/sqrt(3.0)) };
  for(int j = 0; j < 4; ++j) {
    for(int i = 0; i < 4; ++i) {
      stencil[j][i] = 0.0;
      for(int n = 0; n < 2; ++n) {
        double eta = gaussPts[n];
        for(int m = 0; m < 2; ++m) {
          double psi = gaussPts[m];
          stencil[j][i] += ( (dPhidPsi(j, eta)*dPhidPsi(i, eta)) + (dPhidEta(j, psi)*dPhidEta(i, psi)) );
        }//end m
      }//end n
    }//end i
  }//end j
}

void createLocalMatrices(LocalData* data) {
  int rank, npes;
  MPI_Comm_rank(data->commAll, &rank);
  MPI_Comm_size(data->commAll, &npes);

  int Ne = (data->N) - 1;

  if(rank > 0) {
    MatCreateSeqAIJ(PETSC_COMM_SELF, data->N, data->N, 9, PETSC_NULL, &(data->Kssh));
    MatCreateSeqAIJ(PETSC_COMM_SELF, data->N, data->N, 9, PETSC_NULL, &(data->Ksh));
    MatCreateSeqAIJ(PETSC_COMM_SELF, data->N, data->N, 9, PETSC_NULL, &(data->Khs));
    MatCreateSeqAIJ(PETSC_COMM_SELF, data->N, data->N, 9, PETSC_NULL, &(data->Khh));
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
    MatCreateSeqAIJ(PETSC_COMM_SELF, data->N, data->N, 9, PETSC_NULL, &(data->Kssl));
    MatCreateSeqAIJ(PETSC_COMM_SELF, data->N, data->N, 9, PETSC_NULL, &(data->Ksl));
    MatCreateSeqAIJ(PETSC_COMM_SELF, data->N, data->N, 9, PETSC_NULL, &(data->Kls));
    MatCreateSeqAIJ(PETSC_COMM_SELF, data->N, data->N, 9, PETSC_NULL, &(data->Kll));
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


