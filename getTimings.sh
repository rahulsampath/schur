#!/bin/bash
for file in $(find . -type f -name 'rsdN*\.txt')
do
 grep "OuterKsp" $file | gawk '{print $4}' > $file.1.out
 grep "RsdSetUp" $file | gawk '{print $4}' > $file.2.out
 export outFileNameA=`echo $file | sed s/".txt"/"Solve.out"/`
 export outFileNameB=`echo $file | sed s/".txt"/"Setup.out"/`
 mv $file.1.out $outFileNameA
 mv $file.2.out $outFileNameB
 echo processed $file
done


