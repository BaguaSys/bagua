# Fine-tuning BERT on SQuAD1.0

The following is an example to start training on 8 V100 GPUs and Bert Whole Word Masking uncased model to reach an F1 score above 93 on SQuAD1.1:
  
```bash

export SQUAD_DIR=/path/to/SQUAD

python3 -m bagua.distributed.launch --nproc_per_node=8 main.py \
  --model_type bert \
  --model_name_or_path bert-large-uncased-whole-word-masking \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/ \
  --per_gpu_eval_batch_size 6   \
  --per_gpu_train_batch_size=6  
```

Training with the previously defined hyper-parameters yields the following results:
```
f1 = 93.4 
exact_match = 87.3
```

For more information, you may refer to https://huggingface.co/transformers/v2.10.0/examples.html?highlight=squad#squad.
