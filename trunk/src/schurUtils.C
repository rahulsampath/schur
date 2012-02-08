
#include "mpi.h"
#include "schur.h"
#include <cassert>
#include "petscdmmg.h"
#include <vector>

void RSDapplyInverse(LocalData* data, RSDnode* root, Vec f, Vec u) {
  if(root->child) {
    if(root->rankForCurrLevel == (((root->npesForCurrLevel)/2) - 1)) {
      Vec fTmp;
      VecDuplicate(u, &fTmp);

      Vec fL, fStar, fStarHigh, gS, uS, gL;
      MatGetVecs(data->Ksl, &fL, &fStar);
      VecDuplicate(fStar, &fStarHigh);
      VecDuplicate(fStar, &gS);
      VecDuplicate(fStar, &uS);
      VecDuplicate(fL, &gL);

      RSDapplyInverse(data, root->child, f, fTmp);

      map<O, L>(data, fTmp, fL);

      map<O, S>(data, f, gS);

      MatMult(data->Ksl, fL, fStar);

      PetscInt Ssize;
      VecGetSize(fStarHigh, &Ssize);

      PetscScalar* arr;
      MPI_Status status;
      VecGetArray(fStarHigh, &arr);
      MPI_Recv(arr, Ssize, MPI_DOUBLE, 1, 7, data->commLow, &status);
      VecRestoreArray(fStarHigh, &arr);

      VecAXPBYPCZ(gS, -1.0, -1.0, 1.0, fStar, fStarHigh);

      schurSolve(data, true, gS, uS);

      VecGetArray(uS, &arr);
      MPI_Send(arr, Ssize, MPI_DOUBLE, 1, 8, data->commLow);
      VecRestoreArray(uS, &arr);

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
      VecDestroy(uS);
    } else if(root->rankForCurrLevel == ((root->npesForCurrLevel)/2)) {
      Vec fTmp;
      VecDuplicate(u, &fTmp);

      Vec fH, fStar, uS, gH;
      MatGetVecs(data->Ksh, &fH, &fStar);
      VecDuplicate(fStar, &uS);
      VecDuplicate(fH, &gH);

      RSDapplyInverse(data, root->child, f, fTmp);

      map<O, H>(data, fTmp, fH);

      MatMult(data->Ksh, fH, fStar);

      PetscInt Ssize;
      VecGetSize(fStar, &Ssize);

      PetscScalar* arr;
      VecGetArray(fStar, &arr);
      MPI_Send(arr, Ssize, MPI_DOUBLE, 0, 7, data->commHigh);
      VecRestoreArray(fStar, &arr);

      schurSolve(data, false, PETSC_NULL, PETSC_NULL);

      MPI_Status status;
      VecGetArray(uS, &arr);
      MPI_Recv(arr, Ssize, MPI_DOUBLE, 0, 8, data->commHigh, &status);
      VecRestoreArray(uS, &arr);

      MatMult(data->Khs, uS, gH);

      Vec gRhs = DMMGGetRHS(data->mgObj);
      Vec gSol = DMMGGetx(data->mgObj);

      VecZeroEntries(gRhs);
      map<H, MG>(data, gH, gRhs);

      KSPSolve(DMMGGetKSP(data->mgObj), gRhs, gSol);

      VecZeroEntries(u);
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
      MPI_Send(arr, Ssize, MPI_DOUBLE, 1, 5, data->commLow);
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

      MPI_Status status;
      Vec uSout;
      VecDuplicate(uS, &uSout);
      VecGetArray(uSout, &arr);
      MPI_Recv(arr, Ssize, MPI_DOUBLE, 1, 6, data->commLow, &status);
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

      MPI_Status status;
      PetscScalar *arr;
      VecGetArray(uS, &arr);
      MPI_Recv(arr, Ssize, MPI_DOUBLE, 0, 5, data->commHigh, &status);
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
      MPI_Send(arr, Ssize, MPI_DOUBLE, 0, 6, data->commHigh);
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
    MPI_Send(arr, Ssize, MPI_DOUBLE, 1, 3, data->commLow);
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

    MPI_Status status;
    VecGetArray(uSout, &arr);
    MPI_Recv(arr, Ssize, MPI_DOUBLE, 1, 4, data->commLow, &status);
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
    MPI_Status status;
    VecGetArray(uSinCopy, &arr);
    MPI_Recv(arr, Ssize, MPI_DOUBLE, 0, 3, data->commHigh, &status);
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
    MPI_Send(arr, Ssize, MPI_DOUBLE, 0, 4, data->commHigh);
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
}

void mgSolve(LocalData* data, Vec rhs, Vec sol) {
  //To be implemented
}

void mgMatMult(LocalData* data, Vec in, Vec out) {
  //To be implemented
}

void computeSchurDiag(LocalData* data) {
  int rank, npes;
  MPI_Comm_rank(data->commAll, &rank);
  MPI_Comm_size(data->commAll, &npes);

  MPI_Request requestH;
  std::vector<double> dH;
  if(rank > 0) {
    Vec diagH, diagSh;
    std::vector<double> dHstar;
    int Ssize, Hsize;
    PetscScalar *arr;

    MatGetVecs(data->Khh, PETSC_NULL, &diagH);
    MatGetVecs(data->Kssh, PETSC_NULL, &diagSh);

    MatGetDiagonal(data->Khh, diagH);
    MatGetDiagonal(data->Kssh, diagSh);

    VecGetSize(diagSh, &Ssize);
    VecGetSize(diagH, &Hsize);

    dHstar.resize(Ssize);
    VecGetArray(diagH, &arr);
    for(int i = 0; i < Ssize; i++) {
      dHstar[i] = 0.0;
      for(int k = 0; k < Hsize; k++) {
        PetscScalar val1, val2;
        MatGetValues(data->Ksh, 1, &i, 1, &k, &val1);
        MatGetValues(data->Khs, 1, &k, 1, &i, &val2);
        dHstar[i] += (val1*val2/(arr[k]));
      }//end for k
    }//end for i
    VecRestoreArray(diagH, &arr);

    dH.resize(Ssize);
    VecGetArray(diagSh, &arr);
    for(int i = 0; i < Ssize; i++) {
      dH[i] = arr[i] - dHstar[i];
    }//end for i
    VecRestoreArray(diagSh, &arr);

    MPI_Isend((&(dH[0])), Ssize, MPI_DOUBLE, 0, 2, data->commHigh, &requestH);

    VecDestroy(diagH);
    VecDestroy(diagSh);
  }

  if(rank < (npes - 1)) {
    Vec diagL, diagSl;
    std::vector<double> dLstar, dL;
    int Ssize, Lsize;
    PetscScalar *arr;

    MatGetVecs(data->Kssl, PETSC_NULL, &diagSl);
    MatGetDiagonal(data->Kssl, diagSl);
    VecGetSize(diagSl, &Ssize);

    std::vector<double> dHcopy(Ssize);

    MPI_Request requestL;
    MPI_Irecv((&(dHcopy[0])), Ssize, MPI_DOUBLE, 1, 2, data->commLow, &requestL);

    MatGetVecs(data->Kll, PETSC_NULL, &diagL);
    MatGetDiagonal(data->Kll, diagL);
    VecGetSize(diagL, &Lsize);

    dLstar.resize(Ssize);
    VecGetArray(diagL, &arr);
    for(int i = 0; i < Ssize; i++) {
      dLstar[i] = 0.0;
      for(int k = 0; k < Lsize; k++) {
        PetscScalar val1, val2;
        MatGetValues(data->Ksl, 1, &i, 1, &k, &val1);
        MatGetValues(data->Kls, 1, &k, 1, &i, &val2);
        dLstar[i] += (val1*val2/(arr[k]));
      }//end for k
    }//end for i
    VecRestoreArray(diagL, &arr);

    dL.resize(Ssize);
    VecGetArray(diagSl, &arr);
    for(int i = 0; i < Ssize; i++) {
      dL[i] = arr[i] - dLstar[i];
    }//end for i
    VecRestoreArray(diagSl, &arr);

    VecDestroy(diagL);
    VecDestroy(diagSl);

    MPI_Status status;
    MPI_Wait(&requestL, &status);

    VecGetArray(data->diagS, &arr);
    for(int i = 0; i < Ssize; i++) {
      arr[i] = dL[i] + dHcopy[i];
    }//end for i
    VecRestoreArray(data->diagS, &arr);
  }

  if(rank > 0) {
    MPI_Status status;
    MPI_Wait(&requestH, &status);
  }
}

void destroyRSDtree(RSDnode *root) {
  if(root->child) {
    destroyRSDtree(root->child);
  }
  delete root;  
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
    }
  }

  MPI_Group_free(&groupAll);
}




