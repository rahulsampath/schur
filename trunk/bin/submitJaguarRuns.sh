#!/bin/bash
for P in 8 128 2048 32768
do
  for N in 17 33 65 129
  do
    for G in 1 2 4 8
    do
      for C in 1 2
      do
        export S=`echo $[2*$P]`
        echo "#!/bin/bash" > myJob.pbs
        echo "#" >> myJob.pbs
        echo "#PBS -A NFI007FP" >> myJob.pbs
        echo "#PBS -j oe" >> myJob.pbs
        echo "#PBS -l gres=widow2" >> myJob.pbs
        echo "#PBS -V" >> myJob.pbs
        echo "#PBS -l walltime=1:00:00" >> myJob.pbs
        echo "#PBS -l size="$S >> myJob.pbs
        echo "#PBS -N rsdN"$N"P"$P"G"$G"C"$C >> myJob.pbs
        echo "#" >> myJob.pbs
        echo " "  >> myJob.pbs
        echo "cd $PBS_O_WORKDIR" >> myJob.pbs
        echo " " >> myJob.pbs
        echo "aprun -n "$P" -S4 -d2 ./testrsd -N "$N" -inner_ksp_max_it "$G" -problem "$C >> myJob.pbs
        echo " " >> myJob.pbs
        echo "echo Finished" >> myJob.pbs
        echo " " >> myJob.pbs
        cat myJob.pbs
        qsub myJob.pbs
      done
    done
  done
done




