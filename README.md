# AlipaySEQ

This repo is the office code for the manuscript "Towards Better Chinese Spelling Check for Search Engines: A New Dataset and Strong Baseline".

In this repo, we release AlipaySEQ, a new CSC dataset collected from the Alipay search engine, which is one of the most popular Chinese mobile search engines. To the best of our knowledge, AlipaySEQ is the first CSC dataset collected from a real-world mobile search engine. The dataset contains 15,522 human-annotated samples selected from 153,426 potentially misspelled queries.

Besides, we also introduce SRF, which is a model-agnostic framework to avoid make unconfident midifications.

<img src="model.pdf" width="65%">



### Data Processing

```
tar -xvf alipayseq.tar.gz
mkdir alipayseq_processed
python data_process.py
```

## Train

To reproduce the result of BERT+SRF, please:

```shell
cd bert
```

To reproduce the result of ReaLiSe+SRF, please:

```shell
cd ReaLiSe
```

