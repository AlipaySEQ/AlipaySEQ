pip install --editable . 
pip install -r requirements.txt




nvidia-smi


PRETRAINED_DIR="pretrained"
DATE_DIR="alipayseq_processed"



SEED=52



EPOCH_NUM=20


OUTPUT_DIR=srf_bs64_epoch_num_${EPOCH_NUM}_$SEED


export PYTHONPATH=./:${PYTHONPATH}




python src/run.py \
--model_type bert-pho2-res-arch3 \
--model_name_or_path $PRETRAINED_DIR \
--image_model_type 0 \
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
--per_gpu_eval_batch_size 50  \
--learning_rate 5e-5 \
--num_train_epochs $EPOCH_NUM \
--seed $SEED \
--warmup_steps 10000  \
--eval_all_checkpoints \
--overwrite_output_dir \
--resfonts font3_fanti




mkdir ${OUTPUT_DIR}_result


python ./src/metric_core.py \
-i ${OUTPUT_DIR}/labels.txt \
-t ${DATE_DIR}/test.lbl.tsv \