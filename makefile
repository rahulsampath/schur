
SHELL = /bin/sh

include ${PETSC_DIR}/${PETSC_ARCH}/conf/petscvariables
include ${PETSC_DIR}/conf/variables

#Set MYCPP to point to c++/mpi compiler

CEXT = C
MYCPPFLAGS = -DPETSC_USE_LOG ${CXXFLAGS}

include ./makefileCore

