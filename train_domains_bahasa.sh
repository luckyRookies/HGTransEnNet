#!/bin/bash

domains1=(plane)
domains11=(hotel taxi)
first=(attraction restaurant hotel taxi)
second=(movie plane police wear hospital)
later_domains2=(movie plane police wear hospital) # not trained: nohup sh run_bahasa/bahasa_with_bert.sh ${d} > logs_slot_tagger_bert/${d}_log.txt 2>&1 &



for d in ${domains1[*]};
do
  nohup sh run_bahasa/bahasa_with_bert.sh ${d} > logs_slot_tagger_bert/${d}_log_$(date +%T).txt 2>&1 &
done

