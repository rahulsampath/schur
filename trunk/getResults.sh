#!/bin/bash
for file in $(find . -type f -name 'rsdN*\.txt')
do
 grep "OuterKsp" $file | gawk '{print $4}' > $file.1.out
 grep "RsdSetUp" $file | gawk '{print $4}' > $file.2.out
 grep "CONVERGED_RTOL" $file | sed s/"Linear solve converged due to CONVERGED_RTOL iterations"/""/ > $file.3.out
 export outFileNameA=`echo $file | sed s/".txt"/"Solve.out"/`
 export outFileNameB=`echo $file | sed s/".txt"/"Setup.out"/`
 export outFileNameC=`echo $file | sed s/".txt"/"Iter.out"/`
 mv $file.1.out $outFileNameA
 mv $file.2.out $outFileNameB
 mv $file.3.out $outFileNameC
 echo processed $file
done


