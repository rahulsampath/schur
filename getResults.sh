#!/bin/bash
for file in $(find . -type f -name 'rsdN*\.o*')
do
 export fileBase=.`echo $file | cut -d '.' -f2 | cut -d '.' -f1`
 echo "FileBase:" $fileBase
 grep "OuterKsp" $file | gawk '{print $4}' > $fileBase.1.txt
 grep "RsdSetUp" $file | gawk '{print $4}' > $fileBase.2.txt
 grep "CONVERGED_RTOL" $file | sed s/"Linear solve converged due to CONVERGED_RTOL iterations"/""/ > $fileBase.3.txt
 export outFileNameA=`echo ${fileBase}"Solve.txt"`
 echo "SolveFile:" $outFileNameA
 export outFileNameB=`echo ${fileBase}"Setup.txt"`
 echo "SetupFile:" $outFileNameB
 export outFileNameC=`echo ${fileBase}"Iter.txt"`
 echo "IterFile:" $outFileNameC
 mv $fileBase.1.txt $outFileNameA
 mv $fileBase.2.txt $outFileNameB
 mv $fileBase.3.txt $outFileNameC
 echo processed $file
done


