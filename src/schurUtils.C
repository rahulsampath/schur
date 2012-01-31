
#include "mpi.h"
#include "schur.h"
#include <cassert>
#include <vector>

void KmatVec(LocalData* data, RSDnode* root, Vec uIn, Vec uOut) {
  if(root->child) {
    if(root->rankForCurrLevel == (((root->npesForCurrLevel)/2) - 1)) {
      PetscInt Lsize, Ssize;
      MatGetSize(data->Kls, &Lsize, &Ssize);

      Vec uS, wS;
      MatGetVecs(data->Kssl, &uS, &wS);

      map(data, O, uIn, S, uS);

      PetscScalar* arr;
      VecGetArray(uS, &arr);
      MPI_Send(arr, Ssize, MPI_DOUBLE, 1, 5, data->commLow);
      VecRestoreArray(uS, &arr);

      Vec uL, bS, cL, yS;
      MatGetVecs(data->Ksl, &uL, &bS);
      VecDuplicate(uL, &cL); 
      VecDuplicate(bS, &yS); 

      map(data, O, uIn, L, uL);

      MatMult(data->Kssl, uS, wS);

      MatMult(data->Ksl, uL, bS);

      MatMult(data->Kls, uS, cL);

      VecWAXPY(yS, 1.0, wS, bS);

      KmatVec(data, root->child, uIn, uOut);

      Vec cO;
      VecDuplicate(uOut, &cO);
      VecZeroEntries(cO);

      map(data, L, cL, O, cO);

      VecAXPY(uOut, 1.0, cO);

      MPI_Status status;
      Vec uSout;
      VecDuplicate(uS, &uSout);
      VecGetArray(uSout, &arr);
      MPI_Recv(arr, Ssize, MPI_DOUBLE, 1, 6, data->commLow, &status);
      VecRestoreArray(uSout, &arr);

      VecAXPY(uSout, 1.0, yS);

      map(data, S, uSout, O, uOut);

      VecDestroy(uSout);
      VecDestroy(cO);
      VecDestroy(cL);
      VecDestroy(uL);
      VecDestroy(bS);
      VecDestroy(yS);
      VecDestroy(uS);
      VecDestroy(wS);
    } else if(root->rankForCurrLevel == ((root->npesForCurrLevel)/2)) {
      PetscInt Hsize, Ssize;
      MatGetSize(data->Khs, &Hsize, &Ssize);

      Vec uS, wS;
      MatGetVecs(data->Kssh, &uS, &wS);

      MPI_Status status;
      PetscScalar *arr;
      VecGetArray(uS, &arr);
      MPI_Recv(arr, Ssize, MPI_DOUBLE, 0, 5, data->commHigh, &status);
      VecRestoreArray(uS, &arr);

      Vec uH, bS, cH, yS;
      MatGetVecs(data->Ksh, &uH, &bS);
      VecDuplicate(uH, &cH); 
      VecDuplicate(bS, &yS); 

      map(data, O, uIn, H, uH);

      MatMult(data->Kssh, uS, wS);

      MatMult(data->Ksh, uH, bS);

      MatMult(data->Khs, uS, cH);

      VecWAXPY(yS, 1.0, wS, bS);

      KmatVec(data, root->child, uIn, uOut);

      Vec cO;
      VecDuplicate(uOut, &cO);
      VecZeroEntries(cO);

      map(data, H, cH, O, cO);

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
    createVector(data, MG, uInMg);
    VecZeroEntries(uInMg);
    VecDuplicate(uInMg, &uOutMg);

    map(data, O, uIn, MG, uInMg);

    mgMatMult(data, uInMg, uOutMg);

    map(data, MG, uOutMg, O, uOut);

    VecDestroy(uInMg);
    VecDestroy(uOutMg);
  }
}

void schurMatVec(LocalData* data, bool isLow, Vec uSin, Vec uSout) {
  if(isLow) {
    PetscInt Lsize, Ssize;
    MatGetSize(data->Kls, &Lsize, &Ssize);

    PetscScalar* arr;
    VecGetArray(uSin, &arr);
    MPI_Send(arr, Ssize, MPI_DOUBLE, 1, 3, data->commLow);
    VecRestoreArray(uSin, &arr);

    Vec uL;
    MatGetVecs(data->Kssl, PETSC_NULL, &uL);

    Vec vL;
    MatGetVecs(data->Kls, PETSC_NULL, &vL);

    Vec rhsMg, solMg;
    createVector(data, MG, rhsMg);
    VecZeroEntries(rhsMg);
    VecDuplicate(rhsMg, &solMg);

    Vec wL, wS;
    MatGetVecs(data->Ksl, &wL, &wS);

    Vec uStarL;
    VecDuplicate(uL, &uStarL);

    MatMult(data->Kssl, uSin, uL);

    MatMult(data->Kls, uSin, vL);

    map(data, L, vL, MG, rhsMg);

    mgSolve(data, rhsMg, solMg);

    map(data, MG, solMg, L, wL);

    MatMult(data->Ksl, wL, wS);

    VecWAXPY(uStarL, -1.0, wS, uL);

    MPI_Status status;
    VecGetArray(uSout, &arr);
    MPI_Recv(arr, Ssize, MPI_DOUBLE, 1, 4, data->commLow, &status);
    VecRestoreArray(uSout, &arr);

    VecAXPY(uSout, 1.0, uStarL);

    VecDestroy(rhsMg);
    VecDestroy(solMg);
    VecDestroy(uL);
    VecDestroy(vL);
    VecDestroy(wL);
    VecDestroy(wS);
    VecDestroy(uStarL);
  } else {
    PetscInt Hsize, Ssize;
    MatGetSize(data->Khs, &Hsize, &Ssize);

    Vec uSinCopy, uH;
    MatGetVecs(data->Kssh, &uSinCopy, &uH);

    PetscScalar* arr;
    MPI_Status status;
    VecGetArray(uSinCopy, &arr);
    MPI_Recv(arr, Ssize, MPI_DOUBLE, 0, 3, data->commHigh, &status);
    VecRestoreArray(uSinCopy, &arr);

    Vec vH;
    MatGetVecs(data->Khs, PETSC_NULL, &vH);

    Vec rhsMg, solMg;
    createVector(data, MG, rhsMg);
    VecZeroEntries(rhsMg);
    VecDuplicate(rhsMg, &solMg);

    Vec wH, wS;
    MatGetVecs(data->Ksh, &wH, &wS);

    Vec uStarH;
    VecDuplicate(uH, &uStarH);

    MatMult(data->Kssh, uSinCopy, uH);

    MatMult(data->Khs, uSinCopy, vH);

    map(data, H, vH, MG, rhsMg);

    mgSolve(data, rhsMg, solMg);

    map(data, MG, solMg, H, wH);

    MatMult(data->Ksh, wH, wS);

    VecWAXPY(uStarH, -1.0, wS, uH);

    VecGetArray(uStarH, &arr);
    MPI_Send(arr, Ssize, MPI_DOUBLE, 0, 4, data->commHigh);
    VecRestoreArray(uStarH, &arr);

    VecDestroy(rhsMg);
    VecDestroy(solMg);
    VecDestroy(uSinCopy);
    VecDestroy(uH);
    VecDestroy(vH);
    VecDestroy(wH);
    VecDestroy(wS);
    VecDestroy(uStarH);
  }
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

void createVector(LocalData* data, ListType type, Vec & v) {
}

void map(LocalData* data, ListType fromType, Vec fromVec, ListType toType, Vec toVec) {
}


