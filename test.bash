#usr/bin/env bash

echo -n "all: " > myjobs.makefile
for n in 100 1000;
do
  for d in 1 2;
  do
    for k in 2 10
    do
      for m in 1 2
      do
        echo -n "target_${n}_${d}_${k}_${m} "
      done >> myjobs.makefile
    done
  done
done

echo >> myjobs.makefile
for n in 100 1000;
do
  for d in 1 2;
  do
    for k in 2 10
    do
      for m in 1 2
      do
	echo -e "target_${n}_${d}_${k}_${m}:"
        for((x=1;x<=100;x++));
        do
          echo -e "\tpython3 kmeans.py $n $d $k $m $x $x >> Results/results_${n}_${d}_${k}_${m}.txt"
        done >> myjobs.makefile
      done >> myjobs.makefile
    done
  done
done
