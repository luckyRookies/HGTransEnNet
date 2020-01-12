#!/bin/bash

domains=(attraction hotel restaurant taxi)

for d in ${domains[*]};
do
  nohup sh run_bahasa/bahasa_with_bert.sh ${d} > ${d}_log.txt 2>&1 &
done

