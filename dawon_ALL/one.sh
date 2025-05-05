#!/bin/bash

# 첫 번째 실험


python fenice_all.py \
  --input_file ../data/aggre_fact_cnndm_sota_maverick.json \
  --result_file ../results/aggre_cnndm_coref/fenice_wr0p3_wb0p7_wcc1_wc1_wm0_ww1_k2.json \
  --weight_rouge 0.3 \
  --weight_bertscore 0.7 \
  --weight_compare_cont 1.0 \
  --weight_cont 1.0 \
  --weight_min 0.0 \
  --weight_mean 1.0 \
  --num_of_top_k 2 \
  --cuda_device 1


python fenice_all.py \
  --input_file ../data/aggre_fact_xsum_sota_maverick.json \
  --result_file ../results/aggre_xsum_coref/fenice_wr0p3_wb0p7_wcc1_wc1_wm0_ww1_k2.json \
  --weight_rouge 0.3 \
  --weight_bertscore 0.7 \
  --weight_compare_cont 1.0 \
  --weight_cont 1.0 \
  --weight_min 0.0 \
  --weight_mean 1.0 \
  --num_of_top_k 2 \
  --cuda_device 1