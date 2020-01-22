# Slot filling and intent detection tasks of spoken language understanding by HGTransEnNet
 * Basic models for slot filling and intent detection:
   * An implementation for "focus" part of the paper "Encoder-decoder with focus-mechanism for sequence labelling based spoken language understanding".
   * An implementation of BLSTM-CRF based on [jiesutd/NCRFpp](https://github.com/jiesutd/NCRFpp/blob/master/model/crf.py)
   * An implementation of joint training of slot filling and intent detection tasks [(Bing Liu and Ian Lane, 2016)](https://arxiv.org/abs/1609.01454).
 * Basic models + Hypergraph / [ELMo](https://arxiv.org/abs/1802.05365) / [BERT](https://github.com/google-research/bert) / [XLNET](https://github.com/zihangdai/xlnet)
 * Tutorials on [ATIS](https://github.com/yvchen/JointSLU), [SNIPS](https://github.com/snipsco/nlu-benchmark/tree/master/2017-06-custom-intent-engines) and [MIT_Restaurant_Movie_corpus](https://groups.csail.mit.edu/sls/downloads/)(w/o intent) datasets.
 
 <img src="./figs/data_annotation_ATIS.png" width="750" alt="data annotation"/>

## Setup
 * python 3.6.x
 * [pytorch](https://pytorch.org/) 1.1
 * pip install gpustat     [if gpu is used]
 * [embeddings](https://github.com/vzhong/embeddings): pip install embeddings
 * [ELMo in allennlp](https://github.com/allenai/allennlp): pip install allennlp
 * [BERT/XLNET in transformers](https://github.com/huggingface/transformers): pip install transformers
 
## About the evaluations of intent detection on ATIS and SNIPS datasets.

As we can know from the datasets, ATIS may have multiple intents for one utterance while SNIPS has only one intent for one utterance. For example, "show me all flights and fares from denver to san francisco <=> atis_flight && atis_airfare". Therefore, there is a public trick in the training and evaluation stages for intent detection of ATIS dataset.

![#f03c15](https://placehold.it/15/f03c15/000000?text=+)**NOTE!!!**![#f03c15](https://placehold.it/15/f03c15/000000?text=+): Impacted by the paper ["What is left to be understood in ATIS?"](https://ieeexplore.ieee.org/abstract/document/5700816), almost all works about ATIS choose the first intent as the label to train a "softmax" intent classifier. In the evaluation stage, it will be viewed as correct if the predicted intent is one of the multiple intents.

 
## Hypergraph Transfer Encoding Network (HGTransEnNet)
 1. Generating and extracting word embedding of English corpus using BERT

 2. Using HGTransEnNet to update word embedding of bahasa corpus by English corpus within the same domain
    
    Prepare X (vertex feature tensor) and H (incidence matrix) matrices then run
 ```sh
    python scripts/train_HGTransEnNet.py
    python data_processing/extract_HG_word_embedding.py
 ```

 3. Using word embedding of bahasa to do the slot filling and intent detection
 ```shell script
    sh run_bahasa/bahasa_with_word_mebedding_by_HG.py domain_name
 ```

