
#include <cassert>
#include <cmath>

extern double stencil[4][4];

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



