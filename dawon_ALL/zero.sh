#!/bin/bash

# 첫 번째 실험

python fenice_all.py \
  --input_file ../data/aggre_fact_xsum_sota_maverick.json \
  --result_file ../results/aggre_xsum_coref/fenice_wr0p3_wb0p7_wcc0_wc1_wm1_ww0_k2.json \
  --weight_rouge 0.3 \
  --weight_bertscore 0.7 \
  --weight_compare_cont 0.0 \
  --weight_cont 1.0 \
  --weight_min 1.0 \
  --weight_mean 0.0 \
  --num_of_top_k 2 \
  --cuda_device 0


python fenice_all.py \
  --input_file ../data/aggre_fact_xsum_sota_maverick.json \
  --result_file ../results/aggre_xsum_coref/fenice_wr0_wb1_wcc0_wc1_wm1_ww0_k2.json \
  --weight_rouge 0.0 \
  --weight_bertscore 1.0 \
  --weight_compare_cont 0.0 \
  --weight_cont 1.0 \
  --weight_min 1.0 \
  --weight_mean 0.0 \
  --num_of_top_k 2 \
  --cuda_device 0


