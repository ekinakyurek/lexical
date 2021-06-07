#!/bin/bash
std(){
  awk '{sum+=$1; sumsq+=$1*$1}END{print "± " sqrt(sumsq/NR - (sum/NR)**2)}'
}

mean(){
  awk 'BEGIN { sum=0 } { sum+=$1 } END {print sum / NR}'
}
min(){
  sort -n | head -1
}
max(){
  sort -n | tail -1
}
stdmean(){
  echo -n "TEST:"
  mu=$(printf "$1" | mean | tr -d '\n')
  sigma=$(printf "$1" | std | tr -d '\n')
  maximum=$(printf "$1" | max | tr -d '\n')
  minimum=$(printf "$1" | min | tr -d '\n')
  length=$(printf "$1" | wc -l)
  echo -n "$mu ($sigma) (max: $maximum , min: $minimum ) count ($length)"
  echo
}

WARMUP=70
BATCHSIZE=5
LR=1.0
for prefix in "pmi" "goodman" "simple" "fast_forward" "fast_intersect"; do
  expname=${prefix}
  if [ -d "$expname" ]; then
    cd $expname
    numbers1=$(grep -oh 'test evaluation (greedy)/acc [0-9].*' *.out | awk '{print $4}' FS=" ")
    line1=$(echo "$numbers1" | wc -l)
    if [ "$line1" -lt 2 ]; then
      cd ..
      continue
    fi
    echo -n "${expname}"
    stdmean "$numbers1"
    cd ..
  else
    continue
  fi
done
