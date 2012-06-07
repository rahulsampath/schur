
#include "schur.h"
#include "schurMaps.h"

PetscErrorCode lowSchurMatMult(Mat mat, Vec in, Vec out) {
  LocalData* data;
  MatShellGetContext(mat, (void**)(&data));

  VecBufType1* buf = data->buf1;

  Vec inSeq;
  if(buf->inSeq) {
    inSeq = buf->inSeq;
  } else {
    PetscInt locSize; 
    VecGetLocalSize(in, &locSize);
    VecCreateSeq(PETSC_COMM_SELF, locSize, &inSeq);
    buf->inSeq = inSeq;
  }

  Vec outSeq;
  if(buf->outSeq) {
    outSeq = buf->outSeq;
  } else {
    VecDuplicate(inSeq, &outSeq);
    buf->outSeq = outSeq;
  }

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

  VecBufType1* buf = (ctx->data)->buf2;

  Vec inSeq;
  if(buf->inSeq) {
    inSeq = buf->inSeq;
  } else {
    PetscInt locSize; 
    VecGetLocalSize(in, &locSize);
    VecCreateSeq(PETSC_COMM_SELF, locSize, &inSeq);
    buf->inSeq = inSeq;
  }

  Vec outSeq;
  if(buf->outSeq) {
    outSeq = buf->outSeq;
  } else {
    VecDuplicate(inSeq, &outSeq);
    buf->outSeq = outSeq;
  }

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

  return 0;
}

PetscErrorCode outerPCapply(void* ptr, Vec in, Vec out) {
  OuterContext* ctx = static_cast<OuterContext*>(ptr);

  VecBufType1* buf = (ctx->data)->buf3;

  Vec inSeq;
  if(buf->inSeq) {
    inSeq = buf->inSeq;
  } else {
    PetscInt locSize; 
    VecGetLocalSize(in, &locSize);
    VecCreateSeq(PETSC_COMM_SELF, locSize, &inSeq);
    buf->inSeq = inSeq;
  }

  Vec outSeq;
  if(buf->outSeq) {
    outSeq = buf->outSeq;
  } else {
    VecDuplicate(inSeq, &outSeq);
    buf->outSeq = outSeq;
  }

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

  return 0;
}

void RSDapplyInverse(LocalData* data, RSDnode* root, Vec f, Vec u) {
  VecBufType5* buf = data->buf7;

  if(root->child) {
    if(root->rankForCurrLevel == (((root->npesForCurrLevel)/2) - 1)) {
      Vec fStarHcopy;
      if(buf->fStarHcopy) {
        fStarHcopy = buf->fStarHcopy;
      } else {
        MatGetVecs(data->Ksl, PETSC_NULL, &fStarHcopy);
        buf->fStarHcopy = fStarHcopy;
      }

      PetscScalar* recvArr7;
      VecGetArray(fStarHcopy, &recvArr7);

      PetscInt Ssize;
      VecGetSize(fStarHcopy, &Ssize);

      MPI_Request recvRequest7;
      MPI_Irecv(recvArr7, Ssize, MPI_DOUBLE, 1, 7, data->commLow, &recvRequest7);

      Vec fTmp;
      if(buf->fTmpL) {
        fTmp = buf->fTmpL;
      } else {
        VecDuplicate(u, &fTmp);
        buf->fTmpL = fTmp;
      }

      RSDapplyInverse(data, root->child, f, fTmp);

      Vec fL;
      if(buf->fL) {
        fL = buf->fL;
      } else {
        MatGetVecs(data->Ksl, &fL, PETSC_NULL);
        buf->fL = fL;
      }

      Vec fStar;
      if(buf->fStarL) {
        fStar = buf->fStarL;
      } else {
        VecDuplicate(fStarHcopy, &fStar);
        buf->fStarL = fStar;
      }

      Vec gS; 
      if(buf->gS) {
        gS = buf->gS;
      } else {
        VecDuplicate(fStar, &gS);
        buf->gS = gS;
      }

      Vec uS; 
      if(buf->uSl) {
        uS = buf->uSl;
      } else {
        VecDuplicate(fStar, &uS);
        buf->uSl = uS;
      }

      map<O, L>(data, fTmp, fL);

      map<O, S>(data, f, gS);

      MatMult(data->Ksl, fL, fStar);

      MPI_Status recvStatus7;
      MPI_Wait(&recvRequest7, &recvStatus7);

      VecRestoreArray(fStarHcopy, &recvArr7);

      VecAXPBYPCZ(gS, -1.0, -1.0, 1.0, fStar, fStarHcopy);

      schurSolve(data, true, gS, uS);

      PetscScalar* sendArr8;
      VecGetArray(uS, &sendArr8);

      MPI_Request sendRequest8;
      MPI_Isend(sendArr8, Ssize, MPI_DOUBLE, 1, 8, data->commLow, &sendRequest8);

      Vec gL;
      if(buf->gL) {
        gL = buf->gL;
      } else {
        VecDuplicate(fL, &gL);
        buf->gL = gL;
      }

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

      MPI_Status sendStatus8;
      MPI_Wait(&sendRequest8, &sendStatus8);

      VecRestoreArray(uS, &sendArr8);
    } else if(root->rankForCurrLevel == ((root->npesForCurrLevel)/2)) {
      Vec uS; 
      if(buf->uSh) {
        uS = buf->uSh;
      } else {
        MatGetVecs(data->Ksh, PETSC_NULL, &uS);
        buf->uSh = uS;
      }

      PetscInt Ssize;
      VecGetSize(uS, &Ssize);

      PetscScalar* recvArr8;
      VecGetArray(uS, &recvArr8);

      MPI_Request recvRequest8;
      MPI_Irecv(recvArr8, Ssize, MPI_DOUBLE, 0, 8, data->commHigh, &recvRequest8);

      Vec fTmp;
      if(buf->fTmpH) {
        fTmp = buf->fTmpH;
      } else {
        VecDuplicate(u, &fTmp);
        buf->fTmpH = fTmp;
      }

      RSDapplyInverse(data, root->child, f, fTmp);

      Vec fH; 
      if(buf->fH) {
        fH = buf->fH;
      } else {
        MatGetVecs(data->Ksh, &fH, PETSC_NULL);
        buf->fH = fH;
      }

      map<O, H>(data, fTmp, fH);

      Vec fStar;
      if(buf->fStarH) {
        fStar = buf->fStarH;
      } else {
        VecDuplicate(uS, &fStar);
        buf->fStarH = fStar;
      }

      MatMult(data->Ksh, fH, fStar);

      PetscScalar* sendArr7;
      VecGetArray(fStar, &sendArr7);

      MPI_Request sendRequest7;
      MPI_Isend(sendArr7, Ssize, MPI_DOUBLE, 0, 7, data->commHigh, &sendRequest7);

      schurSolve(data, false, PETSC_NULL, PETSC_NULL);

      Vec gH;
      if(buf->gH) {
        gH = buf->gH;
      } else {
        VecDuplicate(fH, &gH);
        buf->gH = gH;
      }

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

      MPI_Status sendStatus7;
      MPI_Wait(&sendRequest7, &sendStatus7);

      VecRestoreArray(fStar, &sendArr7);
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
  VecBufType4* buf = data->buf6;

  if(root->child) {
    if(root->rankForCurrLevel == (((root->npesForCurrLevel)/2) - 1)) {
      Vec uSout;
      if(buf->uSout) {
        uSout = buf->uSout;
      } else {
        MatGetVecs(data->Kssl, &uSout, PETSC_NULL);
        buf->uSout = uSout;
      }

      PetscInt Ssize;
      VecGetSize(uSout, &Ssize);

      PetscScalar* recvArr6;
      VecGetArray(uSout, &recvArr6);

      MPI_Request recvRequest6;
      MPI_Irecv(recvArr6, Ssize, MPI_DOUBLE, 1, 6, data->commLow, &recvRequest6);

      Vec uS;
      if(buf->uSl) {
        uS = buf->uSl;
      } else {
        VecDuplicate(uSout, &uS);
        buf->uSl = uS;
      }

      map<O, S>(data, uIn, uS);

      PetscScalar* sendArr5;
      VecGetArray(uS, &sendArr5);

      MPI_Request sendRequest5;
      MPI_Isend(sendArr5, Ssize, MPI_DOUBLE, 1, 5, data->commLow, &sendRequest5);

      KmatVec(data, root->child, uIn, uOut);

      Vec wS;
      if(buf->wSl) {
        wS = buf->wSl;
      } else {
        VecDuplicate(uS, &wS);
        buf->wSl = wS;
      }

      MatMult(data->Kssl, uS, wS);

      Vec uL; 
      if(buf->uL) {
        uL = buf->uL;
      } else {
        MatGetVecs(data->Ksl, &uL, PETSC_NULL);
        buf->uL = uL;
      }

      Vec bS;
      if(buf->bSl) {
        bS = buf->bSl;
      } else {
        MatGetVecs(data->Ksl, PETSC_NULL, &bS);
        buf->bSl = bS;
      }

      map<O, L>(data, uIn, uL);

      MatMult(data->Ksl, uL, bS);

      Vec yS;
      if(buf->ySl) {
        yS = buf->ySl;
      } else {
        VecDuplicate(bS, &yS); 
        buf->ySl = yS;
      }

      VecWAXPY(yS, 1.0, wS, bS);

      Vec cL;
      if(buf->cL) {
        cL = buf->cL;
      } else {
        VecDuplicate(uL, &cL); 
        buf->cL = cL;
      }

      MatMult(data->Kls, uS, cL);

      Vec cO;
      if(buf->cOl) {
        cO = buf->cOl;
      } else {
        VecDuplicate(uOut, &cO);
        buf->cOl = cO;
      }

      VecZeroEntries(cO);
      map<L, O>(data, cL, cO);

      VecAXPY(uOut, 1.0, cO);

      MPI_Status recvStatus6;
      MPI_Wait(&recvRequest6, &recvStatus6);

      VecRestoreArray(uSout, &recvArr6);

      VecAXPY(uSout, 1.0, yS);

      map<S, O>(data, uSout, uOut);

      MPI_Status sendStatus5;
      MPI_Wait(&sendRequest5, &sendStatus5);

      VecRestoreArray(uS, &sendArr5);
    } else if(root->rankForCurrLevel == ((root->npesForCurrLevel)/2)) {
      Vec uS;
      if(buf->uSh) {
        uS = buf->uSh;
      } else {
        MatGetVecs(data->Kssh, &uS, PETSC_NULL);
        buf->uSh = uS;
      }

      PetscInt Ssize;
      VecGetSize(uS, &Ssize);

      PetscScalar *recvArr5;
      VecGetArray(uS, &recvArr5);

      MPI_Request recvRequest5;
      MPI_Irecv(recvArr5, Ssize, MPI_DOUBLE, 0, 5, data->commHigh, &recvRequest5);

      KmatVec(data, root->child, uIn, uOut);

      Vec wS;
      if(buf->wSh) {
        wS = buf->wSh;
      } else {
        VecDuplicate(uS, &wS);
        buf->wSh = wS;
      }

      Vec uH;
      if(buf->uH) {
        uH = buf->uH;
      } else {
        MatGetVecs(data->Ksh, &uH, PETSC_NULL);
        buf->uH = uH;
      }

      Vec bS; 
      if(buf->bSh) {
        bS = buf->bSh;
      } else {
        MatGetVecs(data->Ksh, PETSC_NULL, &bS);
        buf->bSh = bS;
      }

      Vec cH; 
      if(buf->cH) {
        cH = buf->cH;
      } else {
        VecDuplicate(uH, &cH); 
        buf->cH = cH;
      }

      Vec yS;
      if(buf->ySh) {
        yS = buf->ySh;
      } else {
        VecDuplicate(bS, &yS); 
        buf->ySh = yS;
      }

      map<O, H>(data, uIn, uH);

      MatMult(data->Ksh, uH, bS);

      Vec cO;
      if(buf->cOh) {
        cO = buf->cOh;
      } else {
        VecDuplicate(uOut, &cO);
        buf->cOh = cO;
      }

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

      MPI_Status sendStatus6;
      MPI_Wait(&sendRequest6, &sendStatus6);

      VecRestoreArray(yS, &sendArr6);
    } else {
      KmatVec(data, root->child, uIn, uOut);
    }
  } else {
    Vec uInMg = DMMGGetx(data->mgObj);
    Vec uOutMg = DMMGGetRHS(data->mgObj);

    VecZeroEntries(uInMg);
    map<O, MG>(data, uIn, uInMg);

    MatMult(DMMGGetJ(data->mgObj), uInMg, uOutMg);

    VecZeroEntries(uOut);
    map<MG, O>(data, uOutMg, uOut);
  }
}

void schurMatVec(LocalData* data, bool isLow, Vec uSin, Vec uSout) {
  VecBufType3* buf = data->buf5;

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
    if(buf->uL) {
      uL = buf->uL;
    } else {
      MatGetVecs(data->Kssl, PETSC_NULL, &uL);
      buf->uL = uL;
    }

    MatMult(data->Kssl, uSin, uL);

    Vec vL;
    if(buf->vL) {
      vL = buf->vL;
    } else {
      MatGetVecs(data->Kls, PETSC_NULL, &vL);
      buf->vL = vL;
    }

    MatMult(data->Kls, uSin, vL);

    Vec rhsMg = DMMGGetRHS(data->mgObj);
    Vec solMg = DMMGGetx(data->mgObj);

    VecZeroEntries(rhsMg);
    map<L, MG>(data, vL, rhsMg);

    KSPSolve(DMMGGetKSP(data->mgObj), rhsMg, solMg);

    Vec wL;
    if(buf->wL) {
      wL = buf->wL;
    } else {
      MatGetVecs(data->Ksl, &wL, PETSC_NULL);
      buf->wL = wL;
    }

    Vec wS;
    if(buf->wSl) {
      wS = buf->wSl;
    } else {
      MatGetVecs(data->Ksl, PETSC_NULL, &wS);
      buf->wSl = wS;
    }

    map<MG, L>(data, solMg, wL);

    MatMult(data->Ksl, wL, wS);

    Vec uStarL;
    if(buf->uStarL) {
      uStarL = buf->uStarL;
    } else {
      VecDuplicate(uL, &uStarL);
      buf->uStarL = uStarL;
    }

    VecWAXPY(uStarL, -1.0, wS, uL);

    MPI_Status recvStatus4;
    MPI_Wait(&recvRequest4, &recvStatus4);

    VecRestoreArray(uSout, &recvArr4);

    VecAXPY(uSout, 1.0, uStarL);

    MPI_Status sendStatus3;
    MPI_Wait(&sendRequest3, &sendStatus3);

    VecRestoreArray(uSin, &sendArr3);
  } else {
    Vec uSinCopy;
    if(buf->uSinCopy) {
      uSinCopy = buf->uSinCopy;
    } else {
      MatGetVecs(data->Kssh, &uSinCopy, PETSC_NULL);
      buf->uSinCopy = uSinCopy;
    }

    PetscInt Ssize;
    VecGetSize(uSinCopy, &Ssize);

    PetscScalar* recvArr3;
    VecGetArray(uSinCopy, &recvArr3);

    MPI_Request recvRequest3;
    MPI_Irecv(recvArr3, Ssize, MPI_DOUBLE, 0, 3, data->commHigh, &recvRequest3);

    Vec uH;
    if(buf->uH) {
      uH = buf->uH;
    } else {
      MatGetVecs(data->Kssh, PETSC_NULL, &uH);
      buf->uH = uH;
    }

    Vec vH;
    if(buf->vH) {
      vH = buf->vH;
    } else {
      MatGetVecs(data->Khs, PETSC_NULL, &vH);
      buf->vH = vH;
    }

    Vec wH;
    if(buf->wH) {
      wH = buf->wH;
    } else {
      MatGetVecs(data->Ksh, &wH, PETSC_NULL);
      buf->wH = wH;
    }

    Vec wS;
    if(buf->wSh) {
      wS = buf->wSh;
    } else {
      MatGetVecs(data->Ksh, PETSC_NULL, &wS);
      buf->wSh = wS;
    }

    Vec uStarH;
    if(buf->uStarH) {
      uStarH = buf->uStarH;
    } else {
      VecDuplicate(uH, &uStarH);
      buf->uStarH = uStarH;
    }

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

    MPI_Status sendStatus4;
    MPI_Wait(&sendRequest4, &sendStatus4);

    VecRestoreArray(uStarH, &sendArr4);
  }
}

void schurSolve(LocalData* data, bool isLow, Vec rhs, Vec sol) {
  VecBufType2* buf = data->buf4;

  Vec rhsKsp, solKsp;
  if(isLow) {
    if(buf->rhsKspL) {
      rhsKsp = buf->rhsKspL;
    } else {
      PetscInt localSize;
      VecGetLocalSize(rhs, &localSize);
      VecCreateMPI(data->commLow, localSize, PETSC_DETERMINE, &rhsKsp);
      buf->rhsKspL = rhsKsp;
    }
    if(buf->solKspL) {
      solKsp = buf->solKspL;
    } else {
      VecDuplicate(rhsKsp, &solKsp);
      buf->solKspL = solKsp;
    }
  } else {
    if(buf->rhsKspH) {
      rhsKsp = buf->rhsKspH;
    } else {
      VecCreateMPI(data->commHigh, 0, PETSC_DETERMINE, &rhsKsp);
      buf->rhsKspH = rhsKsp;
    }
    if(buf->solKspH) {
      solKsp = buf->solKspH;
    } else {
      VecDuplicate(rhsKsp, &solKsp);
      buf->solKspH = solKsp;
    }
  }

  if(isLow) {
    PetscScalar *rhsArr;
    PetscScalar *solArr;

    VecGetArray(rhs, &rhsArr);
    VecGetArray(sol, &solArr);

    VecPlaceArray(rhsKsp, rhsArr);
    VecPlaceArray(solKsp, solArr);

    KSPSolve(data->lowSchurKsp, rhsKsp, solKsp);

    VecResetArray(rhsKsp);
    VecResetArray(solKsp);

    VecRestoreArray(rhs, &rhsArr);
    VecRestoreArray(sol, &solArr);
  } else {
    KSPSolve(data->highSchurKsp, rhsKsp, solKsp);
  }
}



