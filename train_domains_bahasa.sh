#!/bin/bash

domains=(attraction hotel restaurant taxi)


for d in ${domains[*]};
do
  nohup sh run_bahasa/bahasa_with_fasttext.sh taxi > logs/taxi_log.txt 2>&1 &
done

