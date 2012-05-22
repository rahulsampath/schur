
#include <iostream>
#include <cstdio>

int main() {
  FILE*fp = fopen("tables.tex", "w");

  int N[] = {17, 65, 257, 513};
  int P[] = {4, 64, 256, 512};
  int G[] = {1, 2, 4, 8};
  int C[] = {1, 2};

  const int Nlen = 4;
  const int Plen = 4;
  const int Glen = 4;
  const int Clen = 4;

  for(int ci = 0; ci < Clen; ++ci) {
    fprintf(fp, "Problem: %d \n\n", C[ci]);

    fprintf(fp, "Iterations \n");
    for(int ni = 0; ni < Nlen; ++ni) {
      fprintf(fp, " %d &", N[ni]);
      for(int pi = 0; pi < Plen; ++pi) {
        for(int gi = 0; gi < Glen; ++gi) {
          char fname[256];
          sprintf(fname, "rsdN%dP%dG%dC%dIter.out", N[ni], P[pi], G[gi], C[ci]);
          FILE* inp = fopen(fname, "r");
          int num;
          fscanf(inp, "%d", &num);
          fprintf(fp, " %d &", num);
          fclose(inp);
        }//end gi
      }//end pi
      fprintf(fp, "\\\\ \\hline \n ");
    }//end ni
    fprintf(fp, " \n\n ");

    fprintf(fp, "Setup Time \n");
    for(int ni = 0; ni < Nlen; ++ni) {
      fprintf(fp, " %d &", N[ni]);
      for(int pi = 0; pi < Plen; ++pi) {
        for(int gi = 0; gi < Glen; ++gi) {
          char fname[256];
          sprintf(fname, "rsdN%dP%dG%dC%dSetup.out", N[ni], P[pi], G[gi], C[ci]);
          FILE* inp = fopen(fname, "r");
          double val;
          fscanf(inp, "%lf", &val);
          fprintf(fp, " %.3lg &", val);
          fclose(inp);
        }//end gi
      }//end pi
      fprintf(fp, "\\\\ \\hline \n ");
    }//end ni
    fprintf(fp, " \n\n ");

    fprintf(fp, "Solve Time \n");
    for(int ni = 0; ni < Nlen; ++ni) {
      fprintf(fp, " %d &", N[ni]);
      for(int pi = 0; pi < Plen; ++pi) {
        for(int gi = 0; gi < Glen; ++gi) {
          char fname[256];
          sprintf(fname, "rsdN%dP%dG%dC%dSolve.out", N[ni], P[pi], G[gi], C[ci]);
          FILE* inp = fopen(fname, "r");
          double val;
          fscanf(inp, "%lf", &val);
          fprintf(fp, " %.3lg &", val);
          fclose(inp);
        }//end gi
      }//end pi
      fprintf(fp, "\\\\ \\hline \n ");
    }//end ni
    fprintf(fp, " \n\n ");

  }//end ci
  fprintf(fp, " \n\n ");

  fclose(fp);
}


