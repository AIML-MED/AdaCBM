#!/usr/bin/env bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:$(pwd)/../..

seed_number=777

dataset='HAM10000' # HAM10000, BCCD, DTR
num_concept=70 # 7*50 HAM
num_layers=2

use_img_norm='false'
use_txt_norm='false'
use_residual='false'

func='our_main'
lr=0.0005
clip_model='ViT-L/14' 

cfn="our_selection" 
batch_size=256 # 256

concept_path="./gpt_concepts/${dataset}.json"
concept2type='false'
pearson_weight=0.9


python main.py --cfg cfg/${dataset}/${dataset}_allshot_fac.py \
--test \
--func ${func} \
--cfg-options "data_root='exp/${dataset}/${directory}'" \
"ckpt_path='saved_models/HAM10000_70concepts_seed_777_acc=82.39_last.ckpt'" \
"concept_select_fn=${cfn}" \
"concept_path='${concept_path}'" \
"concept_select_fn=${cfn}" \
"num_concept=${num_concept}" \
"lr=${lr}" \
"concept_path='${concept_path}'" \
"concept2type='${concept2type}'" \
"clip_model='${clip_model}'" \
"use_img_norm='${use_img_norm}'" \
"use_txt_norm='${use_txt_norm}'" \
"use_residual='${use_residual}'" \
"bs=${batch_size}" \
"num_layers=${num_layers}" \
"pearson_weight=${pearson_weight}" \
"freeze_backbone=true"