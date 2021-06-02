# Fine-tuning BERT on SQuAD1.0

This example fine-tunes BERT on the SQuAD1.0 dataset. 

The following is the script to start training on 8 V100 GPUs and Bert Whole Word Masking uncased model:
  
```bash

export SQUAD_DIR=./downloads/SQUAD

python3 -m torch.distributed.launch --nproc_per_node=8 main.py \
  --model_type bert \
  --model_name_or_path bert-large-uncased-whole-word-masking \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/
```

For more information, you may refer to https://github.com/huggingface/transformers/tree/master/examples/pytorch/question-answering.
