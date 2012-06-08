
#include <iostream>
#include <cstdio>

int main() {
  FILE*fp = fopen("tables.tex", "w");

  int N[] = {17, 33, 65, 129};
  int P[] = {8, 128, 2048, 32768};
  int G[] = {2, 4, 8};
  int C[] = {1, 2, 3, 4, 5};

  const int Nlen = 4;
  const int Plen = 4;
  const int Glen = 3;
  const int Clen = 5;

  for(int ci = 0; ci < Clen; ++ci) {
    fprintf(fp, "\% Problem: %d \n\n", C[ci]);

    fprintf(fp, "\% Iterations \n");

    fprintf(fp, "\\begin{table} \n");
    fprintf(fp, "\\scriptsize \n");
    fprintf(fp, "\\begin{center} \n");
    fprintf(fp, "\\begin{tabular}{");
    for(int j = 0; j < (Plen*Glen); ++j) {
      fprintf(fp, "|c");
    }//end j
    fprintf(fp, "|}\\hline \n");
    for(int pi = 0; pi < Plen; ++pi) {
      fprintf(fp, " & \\multicolumn{%d}{c|}{P = %d}", Glen, P[pi]);
    }//end pi
    fprintf(fp, " \\\\ \\cline{2-%d} \n", (1 + (Plen*Glen)));
    fprintf(fp, " $N$");
    for(int pi = 0; pi < Plen; ++pi) {
      fprintf(fp, " & \\multicolumn{%d}{c|}{$\\gamma$}", Glen);
    }//end pi
    fprintf(fp, " \\\\ \\cline{2-%d} \n", (1 + (Plen*Glen)));
    for(int pi = 0; pi < Plen; ++pi) {
      for(int gi = 0; gi < Glen; ++gi) {
        fprintf(fp, " & %d", G[gi]);
      }//end gi
    }//end pi
    fprintf(fp, " \\\\ \\hline \n");

    for(int ni = 0; ni < Nlen; ++ni) {
      fprintf(fp, " %d &", N[ni]);
      for(int pi = 0; pi < Plen; ++pi) {
        for(int gi = 0; gi < Glen; ++gi) {
          char fname[256];
          sprintf(fname, "rsdN%dP%dG%dC%dIter.txt", N[ni], P[pi], G[gi], C[ci]);
          FILE* inp = fopen(fname, "r");
          int num;
          fscanf(inp, "%d", &num);
          fprintf(fp, " %d", num);
          if( (pi < (Plen - 1)) || (gi < (Glen - 1)) ) {
            fprintf(fp, " &");
          }
          fclose(inp);
        }//end gi
      }//end pi
      fprintf(fp, " \\\\ \\hline \n ");
    }//end ni

    fprintf(fp, " \\end{tabular} \n");
    fprintf(fp, " \\caption{\\label{tab:prob%dIter} Problem %d. Number of outer Krylov iterations.} \n", C[ci], C[ci]);
    fprintf(fp, " \\end{center} \n");
    fprintf(fp, " \\end{table} \n");
    fprintf(fp, " \n\n ");

    fprintf(fp, "\% Setup Time \n");

    fprintf(fp, "\\begin{table} \n");
    fprintf(fp, "\\scriptsize \n");
    fprintf(fp, "\\begin{center} \n");
    fprintf(fp, "\\begin{tabular}{");
    for(int j = 0; j < (Plen*Glen); ++j) {
      fprintf(fp, "|c");
    }//end j
    fprintf(fp, "|}\\hline \n");
    for(int pi = 0; pi < Plen; ++pi) {
      fprintf(fp, " & \\multicolumn{%d}{c|}{P = %d}", Glen, P[pi]);
    }//end pi
    fprintf(fp, " \\\\ \\cline{2-%d} \n", (1 + (Plen*Glen)));
    fprintf(fp, " $N$");
    for(int pi = 0; pi < Plen; ++pi) {
      fprintf(fp, " & \\multicolumn{%d}{c|}{$\\gamma$}", Glen);
    }//end pi
    fprintf(fp, " \\\\ \\cline{2-%d} \n", (1 + (Plen*Glen)));
    for(int pi = 0; pi < Plen; ++pi) {
      for(int gi = 0; gi < Glen; ++gi) {
        fprintf(fp, " & %d", G[gi]);
      }//end gi
    }//end pi
    fprintf(fp, " \\\\ \\hline \n");

    for(int ni = 0; ni < Nlen; ++ni) {
      fprintf(fp, " %d &", N[ni]);
      for(int pi = 0; pi < Plen; ++pi) {
        for(int gi = 0; gi < Glen; ++gi) {
          char fname[256];
          sprintf(fname, "rsdN%dP%dG%dC%dSetup.txt", N[ni], P[pi], G[gi], C[ci]);
          FILE* inp = fopen(fname, "r");
          double val;
          fscanf(inp, "%lf", &val);
          fprintf(fp, " %.3lg", val);
          if( (pi < (Plen - 1)) || (gi < (Glen - 1)) ) {
            fprintf(fp, " &");
          }
          fclose(inp);
        }//end gi
      }//end pi
      fprintf(fp, " \\\\ \\hline \n ");
    }//end ni

    fprintf(fp, " \\end{tabular} \n");
    fprintf(fp, " \\caption{\\label{tab:prob%dSetup} Problem %d. Timings (in seconds) for the setup phase.} \n", C[ci], C[ci]);
    fprintf(fp, " \\end{center} \n");
    fprintf(fp, " \\end{table} \n");
    fprintf(fp, " \n\n ");

    fprintf(fp, "\% Solve Time \n");

    fprintf(fp, "\\begin{table} \n");
    fprintf(fp, "\\scriptsize \n");
    fprintf(fp, "\\begin{center} \n");
    fprintf(fp, "\\begin{tabular}{");
    for(int j = 0; j < (Plen*Glen); ++j) {
      fprintf(fp, "|c");
    }//end j
    fprintf(fp, "|}\\hline \n");
    for(int pi = 0; pi < Plen; ++pi) {
      fprintf(fp, " & \\multicolumn{%d}{c|}{P = %d}", Glen, P[pi]);
    }//end pi
    fprintf(fp, " \\\\ \\cline{2-%d} \n", (1 + (Plen*Glen)));
    fprintf(fp, " $N$");
    for(int pi = 0; pi < Plen; ++pi) {
      fprintf(fp, " & \\multicolumn{%d}{c|}{$\\gamma$}", Glen);
    }//end pi
    fprintf(fp, " \\\\ \\cline{2-%d} \n", (1 + (Plen*Glen)));
    for(int pi = 0; pi < Plen; ++pi) {
      for(int gi = 0; gi < Glen; ++gi) {
        fprintf(fp, " & %d", G[gi]);
      }//end gi
    }//end pi
    fprintf(fp, " \\\\ \\hline \n");

    for(int ni = 0; ni < Nlen; ++ni) {
      fprintf(fp, " %d &", N[ni]);
      for(int pi = 0; pi < Plen; ++pi) {
        for(int gi = 0; gi < Glen; ++gi) {
          char fname[256];
          sprintf(fname, "rsdN%dP%dG%dC%dSolve.txt", N[ni], P[pi], G[gi], C[ci]);
          FILE* inp = fopen(fname, "r");
          double val;
          fscanf(inp, "%lf", &val);
          fprintf(fp, " %.3lg", val);
          if( (pi < (Plen - 1)) || (gi < (Glen - 1)) ) {
            fprintf(fp, " &");
          }
          fclose(inp);
        }//end gi
      }//end pi
      fprintf(fp, " \\\\ \\hline \n ");
    }//end ni

    fprintf(fp, " \\end{tabular} \n");
    fprintf(fp, " \\caption{\\label{tab:prob%dSolve} Problem %d. Timings (in seconds) for the solve phase.} \n", C[ci], C[ci]);
    fprintf(fp, " \\end{center} \n");
    fprintf(fp, " \\end{table} \n");
    fprintf(fp, " \n\n ");

  }//end ci
  fprintf(fp, " \n\n ");

  fclose(fp);
}



