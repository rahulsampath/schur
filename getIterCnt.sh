#!/bin/bash
for file in $(find . -type f -name 'rsdN*\.txt')
do
 grep "CONVERGED_RTOL" $file | sed s/"Linear solve converged due to CONVERGED_RTOL iterations"/""/ > $file.out
 export outFileName=`echo $file | sed s/".txt"/".out"/`
 mv $file.out $outFileName
 echo processed $file
done


