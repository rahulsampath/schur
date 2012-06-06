
#include <cassert>
#include <cmath>
#include "schur.h"

extern double** stencil;
extern int DOFS_PER_NODE;

void createConvectionDiffusionStencil() {
  int N = 9;
  double PeInv = 10e-3;
  PetscOptionsGetInt(PETSC_NULL, "-N", &N, PETSC_NULL);
  const double h = 1.0/(static_cast<double>(N));
  DOFS_PER_NODE = 1;
  const double gaussPts[] = { (1.0/sqrt(3.0)), (-1.0/sqrt(3.0)) };
  typedef double* doublePtr;
  stencil = new doublePtr[4];
  for(int j = 0; j < 4; ++j) {
    stencil[j] = new double[4];
    for(int k = 0; k < 4; ++k) {
      stencil[j][k] = 0.0;
    }//end k
  }//end j

  for(int j = 0; j < 4; ++j) {
    for(int i = 0; i < 4; ++i) {
      for(int n = 0; n < 2; ++n) {
        double eta = gaussPts[n];
        for(int m = 0; m < 2; ++m) {
          double psi = gaussPts[m];
          stencil[j][i] += ( ((h/2.0)*( (dPhidPsi(j, eta)*Phi(i, psi, eta)) + (dPhidEta(j, psi)*Phi(i, psi, eta)) )) +
              (PeInv*( ( (dPhidPsi(j, eta) + dPhidEta(j, psi))*dPhidPsi(i, eta) ) + 
                       ( (dPhidPsi(j, eta) + dPhidEta(j, psi))*dPhidEta(i, psi) ) ) ) ); 
        }//end m
      }//end n
    }//end i
  }//end j
}

void createHardStencilType1() {
  const double epsilon = 0.01;
  const double kappa = 0.01;
  createHardStencil(epsilon, kappa);
}

void createHardStencilType2() {
  const double epsilon = 0.01;
  const double kappa = 100;
  createHardStencil(epsilon, kappa);
}

void createHardStencil(const double epsilon, const double kappa) {
  int N = 9;
  PetscOptionsGetInt(PETSC_NULL, "-N", &N, PETSC_NULL);
  const double h = 1.0/(static_cast<double>(N));
  DOFS_PER_NODE = 2;
  const double gaussPts[] = { (1.0/sqrt(3.0)), (-1.0/sqrt(3.0)) };
  typedef double* doublePtr;
  stencil = new doublePtr[8];
  for(int j = 0; j < 8; ++j) {
    stencil[j] = new double[8];
    for(int k = 0; k < 8; ++k) {
      stencil[j][k] = 0.0;
    }//end k
  }//end j

  for(int j = 0; j < 4; ++j) {
    for(int i = 0; i < 4; ++i) {
      for(int n = 0; n < 2; ++n) {
        double eta = gaussPts[n];
        for(int m = 0; m < 2; ++m) {
          double psi = gaussPts[m];
          //f1
          {
            int dj = 0;
            int di = 0;
            stencil[(j*2) + dj][(i*2) + di] += ( (epsilon*dPhidPsi(i, eta)*dPhidPsi(j, eta)) + (dPhidEta(i, psi)*dPhidEta(j, psi)) );
          }
          //f2
          {
            int dj = 0;
            int di = 1;
            stencil[(j*2) + dj][(i*2) + di] += (h*h*kappa*Phi(i, psi, eta)*Phi(j, psi, eta)/4.0);
          }
          //f3
          {
            int dj = 1;
            int di = 0;
            stencil[(j*2) + dj][(i*2) + di] += (-h*h*kappa*Phi(i, psi, eta)*Phi(j, psi, eta)/4.0);
          }
          //f4
          {
            int dj = 1;
            int di = 1;
            stencil[(j*2) + dj][(i*2) + di] += ( (dPhidPsi(i, eta)*dPhidPsi(j, eta)) + (epsilon*dPhidEta(i, psi)*dPhidEta(j, psi)) );
          }
        }//end m
      }//end n
    }//end i
  }//end j
}

void createPoissonStencil() {
  DOFS_PER_NODE = 1;
  const double gaussPts[] = { (1.0/sqrt(3.0)), (-1.0/sqrt(3.0)) };
  typedef double* doublePtr;
  stencil = new doublePtr[4];
  for(int j = 0; j < 4; ++j) {
    stencil[j] = new double[4];
    for(int k = 0; k < 4; ++k) {
      stencil[j][k] = 0.0;
    }//end k
  }//end j

  for(int j = 0; j < 4; ++j) {
    for(int i = 0; i < 4; ++i) {
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

void destroyStencil() {
  for(int j = 0; j < (4*DOFS_PER_NODE); ++j) {
    delete [] (stencil[j]);
  }//end j
  delete [] stencil;
}

double Phi(int i, double psi, double eta) {
  if(i == 0) {
    return ((1.0 - psi)*(1.0 - eta)/4.0);
  } else if(i == 1) {
    return ((1.0 + psi)*(1.0 - eta)/4.0);
  } else if(i == 2) {
    return ((1.0 - psi)*(1.0 + eta)/4.0);
  } else if(i == 3) {
    return ((1.0 + psi)*(1.0 + eta)/4.0);
  } else {
    assert(false);
  }
  return 0;
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



