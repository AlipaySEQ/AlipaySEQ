PRETRAINED_DIR="pretrained"
DATE_DIR="data"
OUTPUT_DIR="output"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port=453$(($RANDOM%90+10)) --nproc_per_node=4 src/run.py \
    --model_type ecopobert \
    --model_name_or_path $PRETRAINED_DIR \
    --output_dir $OUTPUT_DIR  \
    --do_train --do_eval --do_predict  \
    --data_dir $DATE_DIR \
    --train_file trainall.times2.pkl \
    --dev_file test.sighan15.pkl \
    --dev_label_file test.sighan15.lbl.tsv \
    --predict_file test.sighan15.pkl \
    --predict_label_file test.sighan15.lbl.tsv \
    --order_metric sent-detect-f1  \
    --metric_reverse  \
    --num_save_ckpts 5 \
    --remove_unused_ckpts  \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 50  \
    --learning_rate 5e-5 \
    --num_train_epochs 10  \
    --seed 17 \
    --warmup_steps 10000  \
    --eval_all_checkpoints \
    --overwrite_output_dir
