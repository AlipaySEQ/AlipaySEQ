PRETRAINED_DIR="chinese-roberta-wwm-ext"
DATE_DIR="alipayseq_processed"


seed=52




EPOCH_NUM=20


OUTPUT_DIR=srf_epoch_num_${EPOCH_NUM}_$seed
nvidia-smi



pip install -r requirements.txt



python src/run.py \
--model_type ecopobert \
--model_name_or_path $PRETRAINED_DIR \
--output_dir $OUTPUT_DIR  \
--do_train --do_eval --do_predict  \
--data_dir $DATE_DIR \
--train_file train_annotation_automatic.pkl \
--dev_file test.pkl \
--dev_label_file test.lbl.tsv \
--predict_file test.pkl \
--predict_label_file test.lbl.tsv \
--order_metric sent-detect-f1  \
--metric_reverse  \
--num_save_ckpts 5 \
--max_seq_length 32 \
--remove_unused_ckpts  \
--per_gpu_train_batch_size 64 \
--gradient_accumulation_steps 1 \
--per_gpu_eval_batch_size 50  \
--learning_rate 5e-5 \
--num_train_epochs $EPOCH_NUM  \
--seed $seed \
--warmup_steps 10000  \
--eval_all_checkpoints \
--overwrite_output_dir






for((i=0;i<=${EPOCH_NUM}-1;i++))
do
year=1


python src/test.py \
--device "cuda:0" \
--ckpt_dir $OUTPUT_DIR \
--data_dir $DATE_DIR \
--testset_year $year \
--ckpt_num $i \
--output_dir ${OUTPUT_DIR}_result

done