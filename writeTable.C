
#include <iostream>
#include <cstdio>

int main() {
  FILE*fp = fopen("table.tex", "w");

  int N[] = {17, 65, 257};
  int P[] = {4, 64, 256, 512};
  int G[] = {1, 2, 4, 8};

  for(int ni = 0; ni < 3; ++ni) {
    fprintf(fp, " %d &", (N[ni] - 1));
    for(int pi = 0; pi < 4; ++pi) {
      for(int gi = 0; gi < 4; ++gi) {
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



