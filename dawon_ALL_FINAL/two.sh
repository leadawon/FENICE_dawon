#!/bin/bash

# 첫 번째 실험


python fenice_all.py \
  --input_file ../data/aggregated_cnndm_final.json \
  --result_file ../results/aggre_cnndm_final/fenice_wr0_wb1_wcc1_wc1_wm0_ww1_k3.json \
  --weight_rouge 0.0 \
  --weight_bertscore 1.0 \
  --weight_compare_cont 1.0 \
  --weight_cont 1.0 \
  --weight_min 0.0 \
  --weight_mean 1.0 \
  --num_of_top_k 3 \
  --cuda_device 2





python fenice_all.py \
  --input_file ../data/aggregated_xsum_final.json \
  --result_file ../results/aggre_xsum_final/fenice_wr0_wb1_wcc1_wc1_wm0_ww1_k3.json \
  --weight_rouge 0.0 \
  --weight_bertscore 1.0 \
  --weight_compare_cont 1.0 \
  --weight_cont 1.0 \
  --weight_min 0.0 \
  --weight_mean 1.0 \
  --num_of_top_k 3 \
  --cuda_device 2


