
#include "schur.h"
#include "schurMaps.h"

PetscErrorCode lowSchurMatMult(Mat mat, Vec in, Vec out) {
  LocalData* data;
  MatShellGetContext(mat, (void**)(&data));

  PetscInt locSize; 
  VecGetLocalSize(in, &locSize);

  Vec inSeq, outSeq;
  VecCreateSeq(PETSC_COMM_SELF, locSize, &inSeq);
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

void RSDapplyInverse(LocalData* data, RSDnode* root, Vec f, Vec u) {
  if(root->child) {
    if(root->rankForCurrLevel == (((root->npesForCurrLevel)/2) - 1)) {
      Vec fStarHigh;
      MatGetVecs(data->Ksl, PETSC_NULL, &fStarHigh);

      PetscScalar* recvArr7;
      VecGetArray(fStarHigh, &recvArr7);

      PetscInt Ssize;
      VecGetSize(fStarHigh, &Ssize);

      MPI_Request recvRequest7;
      MPI_Irecv(recvArr7, Ssize, MPI_DOUBLE, 1, 7, data->commLow, &recvRequest7);

      Vec fTmp;
      VecDuplicate(u, &fTmp);

      RSDapplyInverse(data, root->child, f, fTmp);

      Vec fL;
      MatGetVecs(data->Ksl, &fL, PETSC_NULL);

      Vec fStar, gS, uS; 
      VecDuplicate(fStarHigh, &fStar);
      VecDuplicate(fStar, &gS);
      VecDuplicate(fStar, &uS);

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

      Vec gL;
      VecDuplicate(fL, &gL);

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
      Vec uS; 
      MatGetVecs(data->Ksh, PETSC_NULL, &uS);

      PetscInt Ssize;
      VecGetSize(uS, &Ssize);

      PetscScalar* recvArr8;
      VecGetArray(uS, &recvArr8);

      MPI_Request recvRequest8;
      MPI_Irecv(recvArr8, Ssize, MPI_DOUBLE, 0, 8, data->commHigh, &recvRequest8);

      Vec fTmp;
      VecDuplicate(u, &fTmp);

      RSDapplyInverse(data, root->child, f, fTmp);

      Vec fH; 
      MatGetVecs(data->Ksh, &fH, PETSC_NULL);

      map<O, H>(data, fTmp, fH);

      Vec fStar;
      VecDuplicate(uS, &fStar);

      MatMult(data->Ksh, fH, fStar);

      PetscScalar* sendArr7;
      VecGetArray(fStar, &sendArr7);

      MPI_Request sendRequest7;
      MPI_Isend(sendArr7, Ssize, MPI_DOUBLE, 0, 7, data->commHigh, &sendRequest7);

      schurSolve(data, false, PETSC_NULL, PETSC_NULL);

      Vec gH;
      VecDuplicate(fH, &gH);

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
      VecDestroy(uS);

      MPI_Status sendStatus7;
      MPI_Wait(&sendRequest7, &sendStatus7);

      VecRestoreArray(fStar, &sendArr7);

      VecDestroy(fStar);
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
      Vec uSout;
      MatGetVecs(data->Kssl, &uSout, PETSC_NULL);

      PetscInt Ssize;
      VecGetSize(uSout, &Ssize);

      PetscScalar* recvArr6;
      VecGetArray(uSout, &recvArr6);

      MPI_Request recvRequest6;
      MPI_Irecv(recvArr6, Ssize, MPI_DOUBLE, 1, 6, data->commLow, &recvRequest6);

      Vec uS;
      VecDuplicate(uSout, &uS);

      map<O, S>(data, uIn, uS);

      PetscScalar* sendArr5;
      VecGetArray(uS, &sendArr5);

      MPI_Request sendRequest5;
      MPI_Isend(sendArr5, Ssize, MPI_DOUBLE, 1, 5, data->commLow, &sendRequest5);

      KmatVec(data, root->child, uIn, uOut);

      Vec wS; 
      VecDuplicate(uS, &wS);

      MatMult(data->Kssl, uS, wS);

      Vec uL, bS;
      MatGetVecs(data->Ksl, &uL, &bS);

      map<O, L>(data, uIn, uL);

      MatMult(data->Ksl, uL, bS);

      Vec yS;
      VecDuplicate(bS, &yS); 
      VecWAXPY(yS, 1.0, wS, bS);

      Vec cL;
      VecDuplicate(uL, &cL); 

      MatMult(data->Kls, uS, cL);

      Vec cO;
      VecDuplicate(uOut, &cO);

      VecZeroEntries(cO);
      map<L, O>(data, cL, cO);

      VecAXPY(uOut, 1.0, cO);

      MPI_Status recvStatus6;
      MPI_Wait(&recvRequest6, &recvStatus6);

      VecRestoreArray(uSout, &recvArr6);

      VecAXPY(uSout, 1.0, yS);

      map<S, O>(data, uSout, uOut);

      VecDestroy(uSout);
      VecDestroy(cO);
      VecDestroy(cL);
      VecDestroy(uL);
      VecDestroy(bS);
      VecDestroy(yS);
      VecDestroy(wS);

      MPI_Status sendStatus5;
      MPI_Wait(&sendRequest5, &sendStatus5);

      VecRestoreArray(uS, &sendArr5);

      VecDestroy(uS);
    } else if(root->rankForCurrLevel == ((root->npesForCurrLevel)/2)) {
      Vec uS;
      MatGetVecs(data->Kssh, &uS, PETSC_NULL);

      PetscInt Ssize;
      VecGetSize(uS, &Ssize);

      PetscScalar *recvArr5;
      VecGetArray(uS, &recvArr5);

      MPI_Request recvRequest5;
      MPI_Irecv(recvArr5, Ssize, MPI_DOUBLE, 0, 5, data->commHigh, &recvRequest5);

      KmatVec(data, root->child, uIn, uOut);

      Vec wS;
      VecDuplicate(uS, &wS);

      Vec uH, bS, cH, yS;
      MatGetVecs(data->Ksh, &uH, &bS);
      VecDuplicate(uH, &cH); 
      VecDuplicate(bS, &yS); 

      map<O, H>(data, uIn, uH);

      MatMult(data->Ksh, uH, bS);

      Vec cO;
      VecDuplicate(uOut, &cO);
      VecZeroEntries(cO);

      MPI_Status recvStatus5;
      MPI_Wait(&recvRequest5, &recvStatus5);

      VecRestoreArray(uS, &recvArr5);

      MatMult(data->Kssh, uS, wS);

      VecWAXPY(yS, 1.0, wS, bS);

      PetscScalar *sendArr6;
      VecGetArray(yS, &sendArr6);

      MPI_Request sendRequest6;
      MPI_Isend(sendArr6, Ssize, MPI_DOUBLE, 0, 6, data->commHigh, &sendRequest6);

      MatMult(data->Khs, uS, cH);

      map<H, O>(data, cH, cO);

      VecAXPY(uOut, 1.0, cO);

      VecDestroy(cO);
      VecDestroy(cH);
      VecDestroy(uH);
      VecDestroy(bS);
      VecDestroy(uS);
      VecDestroy(wS);

      MPI_Status sendStatus6;
      MPI_Wait(&sendRequest6, &sendStatus6);

      VecRestoreArray(yS, &sendArr6);

      VecDestroy(yS);
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

    PetscScalar* recvArr4;
    VecGetArray(uSout, &recvArr4);

    MPI_Request recvRequest4;
    MPI_Irecv(recvArr4, Ssize, MPI_DOUBLE, 1, 4, data->commLow, &recvRequest4);

    PetscScalar* sendArr3;
    VecGetArray(uSin, &sendArr3);

    MPI_Request sendRequest3;
    MPI_Isend(sendArr3, Ssize, MPI_DOUBLE, 1, 3, data->commLow, &sendRequest3);

    Vec uL;
    MatGetVecs(data->Kssl, PETSC_NULL, &uL);

    MatMult(data->Kssl, uSin, uL);

    Vec vL;
    MatGetVecs(data->Kls, PETSC_NULL, &vL);

    MatMult(data->Kls, uSin, vL);

    Vec rhsMg = DMMGGetRHS(data->mgObj);
    Vec solMg = DMMGGetx(data->mgObj);

    VecZeroEntries(rhsMg);
    map<L, MG>(data, vL, rhsMg);

    KSPSolve(DMMGGetKSP(data->mgObj), rhsMg, solMg);

    Vec wL, wS;
    MatGetVecs(data->Ksl, &wL, &wS);

    map<MG, L>(data, solMg, wL);

    MatMult(data->Ksl, wL, wS);

    Vec uStarL;
    VecDuplicate(uL, &uStarL);

    VecWAXPY(uStarL, -1.0, wS, uL);

    VecDestroy(uL);
    VecDestroy(vL);
    VecDestroy(wL);
    VecDestroy(wS);

    MPI_Status recvStatus4;
    MPI_Wait(&recvRequest4, &recvStatus4);

    VecRestoreArray(uSout, &recvArr4);

    VecAXPY(uSout, 1.0, uStarL);

    VecDestroy(uStarL);

    MPI_Status sendStatus3;
    MPI_Wait(&sendRequest3, &sendStatus3);

    VecRestoreArray(uSin, &sendArr3);
  } else {
    Vec uSinCopy;
    MatGetVecs(data->Kssh, &uSinCopy, PETSC_NULL);

    PetscInt Ssize;
    VecGetSize(uSinCopy, &Ssize);

    PetscScalar* recvArr3;
    VecGetArray(uSinCopy, &recvArr3);

    MPI_Request recvRequest3;
    MPI_Irecv(recvArr3, Ssize, MPI_DOUBLE, 0, 3, data->commHigh, &recvRequest3);

    Vec uH;
    MatGetVecs(data->Kssh, PETSC_NULL, &uH);

    Vec vH;
    MatGetVecs(data->Khs, PETSC_NULL, &vH);

    Vec wH, wS;
    MatGetVecs(data->Ksh, &wH, &wS);

    Vec uStarH;
    VecDuplicate(uH, &uStarH);

    Vec rhsMg = DMMGGetRHS(data->mgObj);
    Vec solMg = DMMGGetx(data->mgObj);

    VecZeroEntries(rhsMg);

    MPI_Status recvStatus3;
    MPI_Wait(&recvRequest3, &recvStatus3);

    VecRestoreArray(uSinCopy, &recvArr3);

    MatMult(data->Kssh, uSinCopy, uH);

    MatMult(data->Khs, uSinCopy, vH);

    map<H, MG>(data, vH, rhsMg);

    KSPSolve(DMMGGetKSP(data->mgObj), rhsMg, solMg);

    map<MG, H>(data, solMg, wH);

    MatMult(data->Ksh, wH, wS);

    VecWAXPY(uStarH, -1.0, wS, uH);

    PetscScalar* sendArr4;
    VecGetArray(uStarH, &sendArr4);

    MPI_Request sendRequest4;
    MPI_Isend(sendArr4, Ssize, MPI_DOUBLE, 0, 4, data->commHigh, &sendRequest4);

    VecDestroy(uSinCopy);
    VecDestroy(uH);
    VecDestroy(vH);
    VecDestroy(wH);
    VecDestroy(wS);

    MPI_Status sendStatus4;
    MPI_Wait(&sendRequest4, &sendStatus4);

    VecRestoreArray(uStarH, &sendArr4);

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



