
SHELL = /bin/sh

include ${PETSC_DIR}/${PETSC_ARCH}/conf/petscvariables
include ${PETSC_DIR}/conf/variables
#Set PCC to point to the MPI/C++ compiler.
#PCC = mpicxx

CEXT = C
CFLAGS = -DPETSC_USE_LOG

include ./makefileCore

