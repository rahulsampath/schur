
#include <cassert>
#include <cmath>
#include "schur.h"

extern double** stencil;

void createHardStencil() {
  const double gaussPts[] = { (1.0/sqrt(3.0)), (-1.0/sqrt(3.0)) };
  typedef double* doublePtr;
  stencil = new doublePtr[8];
  for(int j = 0; j < 8; ++j) {
    stencil[j] = new double[8];
  }//end j

  for(int j = 0; j < 4; ++j) {
    for(int dj = 0; dj < 2; ++dj) {
      for(int i = 0; i < 4; ++i) {
        for(int di = 0; di < 2; ++di) {
          stencil[(j*2) + dj][(i*2) + di] = 0.0;
          for(int n = 0; n < 2; ++n) {
            double eta = gaussPts[n];
            for(int m = 0; m < 2; ++m) {
              double psi = gaussPts[m];
            }//end m
          }//end n
        }//end di
      }//end i
    }//end dj
  }//end j
}

void destroyHardStencil() {
  for(int j = 0; j < 8; ++j) {
    delete [] (stencil[j]);
  }//end j
  delete [] stencil;
}


void createPoissonStencil() {
  const double gaussPts[] = { (1.0/sqrt(3.0)), (-1.0/sqrt(3.0)) };
  typedef double* doublePtr;
  stencil = new doublePtr[4];
  for(int j = 0; j < 4; ++j) {
    stencil[j] = new double[4];
  }//end j

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

void destroyPoissonStencil() {
  for(int j = 0; j < 4; ++j) {
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



