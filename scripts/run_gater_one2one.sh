#!/bin/bash
home_dir="/home/yjc/codes/kg_gater"
export PYTHONPATH=${home_dir}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=6

data_dir="data/kp20k_sorted"

ref_doc_path="${data_dir}/train_src.txt"
ref_kp_path="${data_dir}/train_trg.txt"
hash_path="${data_dir}/train_src-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz"

ref_kp=true
ref_doc=true

dense_retrieve=false
random_search=false

seed=9527
dropout=0.1
batch_size=64
copy_attention=true

model_type="rnn"
enc_layers=1
dec_layers=1
d_model=300
learning_rate=0.001
word_vec_size=100

use_multidoc_graph=true
use_multidoc_copy=true
n_ref_docs=3
n_head=5
atten_drop=0.3
n_topic_words=20
feat_drop=0.3
ffn_drop=0.3
n_iter=2


model_name="One2one"
data_args="Full"
main_args="Seed${seed}_Dropout${dropout}_LR${learning_rate}_BS${batch_size}_Embed${word_vec_size}_NEnc${enc_layers}_NDec${dec_layers}_Dim${d_model}"

if [ "$dense_retrieve" = true ] ; then
    data_args+="_Dense"
fi
if [ "$random_search" = true ] ; then
    data_args+="_Random"
fi
if [ "$ref_kp" = true ] ; then
    data_args+="_RefKP"
fi
if [ "$ref_doc" = true ] ; then
    data_args+="_RefDoc"
fi

if [ ${copy_attention} = true ] ; then
    model_name+="_Copy"
fi
if [ "$use_multidoc_graph" = true ] ; then
    model_name+="_RefGraph"
    data_args+="_RefGraph"
fi
if [ "$use_multidoc_copy" = true ] ; then
    model_name+="_CopyRef"
    data_args+="_CopyRef"
fi

save_data="${data_dir}/${data_args}"
mkdir -p ${save_data}

exp="${data_args}_${model_type}_${model_name}_${main_args}"

echo "============================= build_tfidf ================================="

cmd="python retrievers/build_tfidf.py \
-ref_doc_path ${ref_doc_path} \
-out_dir ${data_dir}
"

echo $cmd
eval $cmd

echo "============================= preprocess: ${save_data} ================================="

preprocess_out_dir="output/preprocess/${data_args}"
mkdir -p ${preprocess_out_dir}

cmd="python preprocess.py \
-data_dir=${data_dir} \
-save_data_dir=${save_data} \
-ref_doc_path=${ref_doc_path} \
-ref_kp_path=${ref_kp_path} \
-hash_path=${hash_path} \
-log_path=${preprocess_out_dir} \
-num_workers 3
"

if [ "$use_multidoc_graph" = true ] ; then
    cmd+=" -use_multidoc_graph"
fi
if [ "$use_multidoc_copy" = true ] ; then
    cmd+=" -use_multidoc_copy"
fi
if [ "$dense_retrieve" = true ] ; then
    cmd+=" -dense_retrieve"
fi
if [ "$random_search" = true ] ; then
    cmd+=" -random_search"
fi
if [ "$ref_kp" = true ] ; then
    cmd+=" -ref_kp"
fi
if [ "$ref_doc" = true ] ; then
    cmd+=" -ref_doc"
fi

echo $cmd
eval $cmd


echo "============================= train: ${exp} ================================="

train_out_dir="output/train/${exp}/"
mkdir -p ${train_out_dir}

cmd="python train.py \
-data ${save_data} \
-vocab ${save_data} \
-exp_path ${train_out_dir} \
-model_path=${train_out_dir} \
-learning_rate ${learning_rate} \
-batch_size ${batch_size} \
-seed ${seed} \
-dropout ${dropout} \
-model_type ${model_type} \
-enc_layers ${enc_layers} \
-dec_layers ${dec_layers} \
-d_model ${d_model} \
-word_vec_size ${word_vec_size}
"
if [ "$copy_attention" = true ] ; then
    cmd+=" -copy_attention"
fi
if [ "$use_multidoc_graph" = true ] ; then
    cmd+=" -use_multidoc_graph"
fi
if [ "$use_multidoc_copy" = true ] ; then
    cmd+=" -use_multidoc_copy"
fi

echo $cmd
eval $cmd

echo "============================= test: ${exp} ================================="

#for data in "kp20k"
for data in "inspec" "krapivin" "nus" "semeval" "kp20k"
do
  echo "============================= testing ${data} ================================="
  test_out_dir="output/test/${exp}/${data}"
  mkdir -p ${test_out_dir}

  src_file="data/testsets/${data}/test_src.txt"
  trg_file="data/testsets/${data}/test_trg.txt"

  cmd="python predict.py \
  -vocab ${save_data} \
  -src_file=${src_file} \
  -pred_path ${test_out_dir} \
  -exp_path ${test_out_dir} \
  -model ${train_out_dir}/best_model.pt \
  -max_length 6 \
  -n_best 200 \
  -beam_size 200 \
  -batch_size 1 \
  -replace_unk \
  -dropout ${dropout} \
  -model_type ${model_type} \
  -enc_layers ${enc_layers} \
  -dec_layers ${dec_layers} \
  -d_model ${d_model} \
  -word_vec_size ${word_vec_size} \
  -ref_doc_path=${ref_doc_path} \
  -ref_kp_path=${ref_kp_path} \
  -hash_path=${hash_path} \
  -num_workers 3
  "
  if [ "$copy_attention" = true ] ; then
      cmd+=" -copy_attention"
  fi
  if [ "$use_multidoc_graph" = true ] ; then
      cmd+=" -use_multidoc_graph"
  fi
  if [ "$use_multidoc_copy" = true ] ; then
      cmd+=" -use_multidoc_copy"
  fi
  if [ "$dense_retrieve" = true ] ; then
      cmd+=" -dense_retrieve"
  fi
  if [ "$random_search" = true ] ; then
      cmd+=" -random_search"
  fi
  if [ "$ref_kp" = true ] ; then
      cmd+=" -ref_kp"
  fi
  if [ "$ref_doc" = true ] ; then
      cmd+=" -ref_doc"
  fi
  echo $cmd
  eval $cmd

  cmd="python evaluate_prediction.py \
  -pred_file_path ${test_out_dir}/predictions.txt \
  -src_file_path ${src_file} \
  -trg_file_path ${trg_file} \
  -exp_path ${test_out_dir} \
  -export_filtered_pred \
  -filtered_pred_path ${test_out_dir} \
  -invalidate_unk \
  -all_ks 5 10 \
  -present_ks 5 10 \
  -absent_ks 5 10
  "
  if [ "$data" = 'kp20k'  ] ; then
      cmd+=" -disable_extra_one_word_filter"
  fi
  cmd+=";cat ${test_out_dir}/results_log_5_10_5_10_5_10.txt"


  echo $cmd
  eval $cmd

done

