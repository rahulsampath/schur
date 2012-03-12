
#include <iostream>
#include <cstdio>

int main() {
  FILE*fp = fopen("table.tex", "w");

  //int N[] = {17, 65, 257, 513};
  //int P[] = {4, 64, 256, 512};
  int N[] = {513};
  int P[] = {4, 64, 256};
  int G[] = {1, 2, 4, 8};

  const int Nlen = 1;
  const int Plen = 3;
  const int Glen = 4;

  for(int ni = 0; ni < Nlen; ++ni) {
    fprintf(fp, " %d &", N[ni]);
    for(int pi = 0; pi < Plen; ++pi) {
      for(int gi = 0; gi < Glen; ++gi) {
        char fname[256];
        sprintf(fname, "rsdN%dP%dG%d.out", N[ni], P[pi], G[gi]);
        FILE* inp = fopen(fname, "r");
        int num;
        fscanf(inp, "%d", &num);
        fprintf(fp, " %d &", num);
        fclose(inp);
      }//end gi
    }//end pi
    fprintf(fp, "\\\\ \\hline \n ");
  }//end ni

  fclose(fp);
}



