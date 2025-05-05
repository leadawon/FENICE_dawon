#!/bin/bash

# 첫 번째 실험

python fenice_all.py \
  --input_file ../data/aggre_fact_cnndm_sota.json \
  --result_file ../results/aggre_cnndm/fenice_wr0p5_wb0p5_wcc1_wc1_wm0_ww1_k2.json \
  --weight_rouge 0.5 \
  --weight_bertscore 0.5 \
  --weight_compare_cont 1.0 \
  --weight_cont 1.0 \
  --weight_min 0.0 \
  --weight_mean 1.0 \
  --num_of_top_k 2 \
  --cuda_device 5



python fenice_all.py \
  --input_file ../data/aggre_fact_cnndm_sota.json \
  --result_file ../results/aggre_cnndm/fenice_wr0p5_wb0p5_wcc1_wc1_wm0_ww1_k3.json \
  --weight_rouge 0.5 \
  --weight_bertscore 0.5 \
  --weight_compare_cont 1.0 \
  --weight_cont 1.0 \
  --weight_min 0.0 \
  --weight_mean 1.0 \
  --num_of_top_k 3 \
  --cuda_device 5

python fenice_all.py \
  --input_file ../data/aggre_fact_xsum_sota.json \
  --result_file ../results/aggre_xsum/fenice_wr0p5_wb0p5_wcc1_wc1_wm0_ww1_k2.json \
  --weight_rouge 0.5 \
  --weight_bertscore 0.5 \
  --weight_compare_cont 1.0 \
  --weight_cont 1.0 \
  --weight_min 0.0 \
  --weight_mean 1.0 \
  --num_of_top_k 2 \
  --cuda_device 5



python fenice_all.py \
  --input_file ../data/aggre_fact_xsum_sota.json \
  --result_file ../results/aggre_xsum/fenice_wr0p5_wb0p5_wcc1_wc1_wm0_ww1_k3.json \
  --weight_rouge 0.5 \
  --weight_bertscore 0.5 \
  --weight_compare_cont 1.0 \
  --weight_cont 1.0 \
  --weight_min 0.0 \
  --weight_mean 1.0 \
  --num_of_top_k 3 \
  --cuda_device 5


