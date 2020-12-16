#usr/bin/env bash

for n in 100 1000;
do
  for d in 1 2;
  do
    for k in 2 10
    do
      for m in 1 2
      do
	echo -n $n $d $k $m ''
        cat results_${n}_${d}_${k}_${m}.txt | Rscript test.r
        echo ""
      done >> results_compiled.txt
    done
  done
done
