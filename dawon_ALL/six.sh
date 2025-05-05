#!/bin/bash

# 첫 번째 실험

python fenice_all.py \
  --input_file ../data/aggre_fact_cnndm_sota_maverickv2.json \
  --result_file ../results/aggre_cnndm_corefv2/fenice_wr0p3_wb0p7_wcc0_wc1_wm0_ww1_k2.json \
  --weight_rouge 0.3 \
  --weight_bertscore 0.7 \
  --weight_compare_cont 0.0 \
  --weight_cont 1.0 \
  --weight_min 0.0 \
  --weight_mean 1.0 \
  --num_of_top_k 2 \
  --cuda_device 6



python fenice_all.py \
  --input_file ../data/aggre_fact_cnndm_sota_maverickv2.json \
  --result_file ../results/aggre_cnndm_corefv2/fenice_wr0p3_wb0p7_wcc0_wc1_wm1_ww0_k3.json \
  --weight_rouge 0.3 \
  --weight_bertscore 0.7 \
  --weight_compare_cont 0.0 \
  --weight_cont 1.0 \
  --weight_min 1.0 \
  --weight_mean 0.0 \
  --num_of_top_k 3 \
  --cuda_device 6