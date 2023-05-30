import pandas as pd
from transformers import BertTokenizer
import tqdm
import pickle

annotation_train = pd.read_csv('alipayseq/annotation_trainset.csv')
annotation_test = pd.read_csv('alipayseq/annotation_test.csv')
automatic_generation = pd.read_csv('alipayseq/automatic_generation.csv')

tokenizer = BertTokenizer.from_pretrained("chinese-roberta-wwm-ext")

total_train_data = []
index = 0

for item in tqdm.tqdm(annotation_train.iterrows(), total=annotation_train.shape[0]):
    data_point = dict()
    data_point['id'] = 'annotation-train-' + str(index)
    data_point['src'] = item[1][0]
    data_point['tgt'] = item[1][1]
    data_point['src_idx'] = tokenizer([item[1][0]])['input_ids'][0]
    data_point['tgt_idx'] = tokenizer([item[1][1]])['input_ids'][0]
    assert len(data_point['src_idx']) == len(data_point['tgt_idx'])
    data_point['lengths'] = len(data_point['src_idx']) - 2
    data_point['tokens_size'] = [1] * data_point['lengths']
    total_train_data.append(data_point)
    index += 1

index = 0
for item in tqdm.tqdm(automatic_generation.iterrows(), total=automatic_generation.shape[0]):
    data_point = dict()
    data_point['id'] = 'automatic-train-' + str(index)
    data_point['src'] = item[1][0]
    data_point['tgt'] = item[1][1]
    data_point['src_idx'] = tokenizer([item[1][0]])['input_ids'][0]
    data_point['tgt_idx'] = tokenizer([item[1][1]])['input_ids'][0]
    assert len(data_point['src_idx']) == len(data_point['tgt_idx'])
    data_point['lengths'] = len(data_point['src_idx']) - 2
    data_point['tokens_size'] = [1] * data_point['lengths']
    total_train_data.append(data_point)
    index += 1

with open('alipayseq_processed/train_annotation_automatic.pkl', 'wb') as f:
    pickle.dump(total_train_data, f)

test_data = []
test_error = []
index = 0

for item in tqdm.tqdm(annotation_test.iterrows(), total=annotation_test.shape[0]):
    data_point = dict()
    data_point['id'] = 'annotation-test-' + str(index)
    data_point['src'] = item[1][0]
    data_point['tgt'] = item[1][1]
    data_point['src_idx'] = tokenizer([item[1][0]])['input_ids'][0]
    data_point['tgt_idx'] = tokenizer([item[1][1]])['input_ids'][0]
    assert len(data_point['src_idx']) == len(data_point['tgt_idx'])
    data_point['lengths'] = len(data_point['src_idx']) - 2
    data_point['tokens_size'] = [1] * data_point['lengths']
    test_data.append(data_point)

    errors = []
    for char_idx in range(len(item[1][0])):
        if item[1][0][char_idx] != item[1][1][char_idx]:
            errors.append([char_idx + 1, item[1][1][char_idx]])

    tem_error = ['annotation-test-' + str(index)]
    if len(errors):
        for error in errors:
            tem_error.append(str(error[0]))
            tem_error.append(error[1])
    else:
        tem_error.append('0')
    test_error.append(', '.join(tem_error))

    index += 1

with open('alipayseq_processed/test.lbl.tsv', 'w') as f:
    f.write('\n'.join(test_error))

with open('alipayseq_processed/test.pkl', 'wb') as f:
    pickle.dump(test_data, f)
