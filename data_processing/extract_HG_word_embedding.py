# -*- coding: utf-8 -*-
import os
import json
import re
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)#,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def format_data():
    # bahasa
    # dir_list = ['hotel', 'attraction', 'taxi', 'restaurant']
    # data_dir = '../../bert/HG_data'
    # HG_emb_file = 'ba_sen_emb_by_HG.npy'
    # sen_file = 'ba_sentences.txt'

    # english
    dir_list = ['hotel']
    data_dir = '../../bert/HG_data'
    HG_emb_file = 'ba_sen_emb_by_HG.npy'
    sen_file = 'ba_sentences.txt'

    for d in dir_list:

        sen2embs = np.load(os.path.join(data_dir, d, HG_emb_file))
        sen2embs = sen2embs.reshape(sen2embs.shape[0], -1, 768)
        word_embeddings = {}
        with open(os.path.join(data_dir, d, sen_file), 'r', encoding='utf8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                words = line.split()
                for i, w in enumerate(words):
                    if w not in word_embeddings:
                        word_embeddings[w] = []
                    w_emb = sen2embs[line_num, i]
                    word_embeddings[w].append(w_emb)

            for w, embs in word_embeddings.items():
                word_emb = np.average(np.array(embs), axis=0).tolist()
                word_embeddings[w] = word_emb
        voc_size, word_dim = len(word_embeddings), 768
        with open(os.path.join(data_dir, d, 'HG_word_embedding.txt'), 'w', encoding='utf8') as f:
            f.write(str(voc_size) + ' ' + str(word_dim) + '\n')
            for w, emb in word_embeddings.items():
                emb_str = ' '.join([ str(float(e)) for e in emb])
                f.write(w + ' ' + emb_str + '\n')
                
if __name__=='__main__':
    format_data()

