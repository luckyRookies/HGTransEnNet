# -*- coding: utf8 -*-

import numpy as np
import os
import json
from collections import Counter
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)#,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_H(en_ba_nums):
    '''

    :param en_ba_nums: [[num_of_vertex_within_edge1, num_of_vertex_within_edge2, ...], [num_of_other_vertex_within_edge1, num_of_other_vertex_within_edge2, ...]]
    :return: incidence matrix
    '''
    col = en_ba_nums.shape[1]
    en_matrix = []
    ba_matrix = []
    for c in range(col):
        en_num = en_ba_nums[0, c]
        ba_num = en_ba_nums[1, c]
        en_mx = [[1 if i == c else 0 for i in range(col)]] * en_num
        ba_mx = [[1 if i == c else 0 for i in range(col)]] * ba_num
        en_matrix += en_mx
        ba_matrix += ba_mx
    en_matrix += ba_matrix
    return np.array(en_matrix)

def generate_target(en_ba_dim):
    en_target = []
    ba_target = []

    for i in range(en_ba_dim.shape[1]):
        en_target += [i] * en_ba_dim[0, i]
        ba_target += [i] * en_ba_dim[1, i]
    return np.array(en_target + ba_target)


def extract_en_ba_sens_of_same_intent():
    lang_domain_intent_sens_both = json.load(open('../../multiwoz_bahasawoz_v2/lang_domain_intent_sens_both.json', 'r', encoding='utf8'))
    lang_domain_intent_both = json.load(open('../../multiwoz_bahasawoz_v2/lang_domain_intent_both.json', 'r', encoding='utf8'))
    dir_list = ['hotel', 'attraction', 'taxi', 'restaurant']
    output_dir = '../../multiwoz_bahasawoz_v2/HG_data'

    for d in dir_list:
        logger.debug('domain: %s' % d)
        en_domain_sens = []
        ba_domain_sens = []
        en_nums = []
        ba_nums = []
        # logger.debug('{}'.format(lang_domain_intent_both))
        for i, ba_en_intents in lang_domain_intent_both[d].items():
            logger.debug('i: {}, ba_en_intents: {}'.format(i, ba_en_intents))
            ba_intent, en_intent = ba_en_intents
            ba_domain_intent_sens = lang_domain_intent_sens_both['bahasawoz'][d][ba_intent]
            en_domain_intent_sens = lang_domain_intent_sens_both['multiwoz'][d][en_intent]
            logger.debug('{}'.format(en_domain_intent_sens[:3]))
            en_nums.append(len(en_domain_intent_sens))
            ba_nums.append(len(ba_domain_intent_sens))

            en_domain_sens += en_domain_intent_sens
            ba_domain_sens += ba_domain_intent_sens
        logger.debug('en nums: {}, ba nums: {}'.format(en_nums, ba_nums))

        # dump english sentences into file
        if not os.path.exists(os.path.join(output_dir, d)):
            os.makedirs(os.path.join(output_dir, d))
        # with open(os.path.join(output_dir, d, 'en_sentences.txt'), 'w', encoding='utf8') as fd:
        #     fd.write('\n'.join(en_domain_sens))
        with open(os.path.join(output_dir, d, 'ba_sentences.txt'), 'w', encoding='utf8') as fd:
            fd.write('\n'.join(ba_domain_sens))
        # np.save(os.path.join(output_dir, d, 'en_ba_sen_nums_metadata.npy'), np.array([en_nums, ba_nums]))

def bert_sen_feature_extract(file, sens_len, word_emb_d=768, max_len=76):
    domain_sens_emb = []
    with open(file, 'r', encoding='utf8') as fd:
        pbar = tqdm(total=sens_len)
        for line in fd:
            sen_bert = json.loads(line)
            sen_emb = []

            for f in sen_bert['features']:
                t, word_emb = f['token'], f['layers'][0]['values']
                if t in ['[CLS]', '[SEP]']:
                    continue
                sen_emb.append(word_emb)
            sen_emb = np.array(sen_emb)
            if len(sen_emb) < max_len:
                sen_emb = np.row_stack((sen_emb, np.zeros((max_len - len(sen_emb), word_emb_d))))
            else:
                sen_emb = sen_emb[:max_len]
            domain_sens_emb.append(sen_emb)
            pbar.update(1)
        pbar.close()
    return np.array(domain_sens_emb)

def main():
    domain_list = ['hotel'] #, 'attraction', 'taxi', 'restaurant']
    data_dir = './'
    # data_dir = '../../multiwoz_bahasawoz_v2/HG_data'
    for d in domain_list:
        print('domain:', d)

        print('generating H ...')
        en_ba_nums = np.load(os.path.join(data_dir, d, 'en_ba_sen_nums_metadata.npy'))
        H = generate_H(en_ba_nums)
        print('H.shape:', H.shape)
        np.save(os.path.join(data_dir, d, 'H.npy'), H)

        en_n, ba_n = np.sum(en_ba_nums, axis=1)

        print('generating X ...')
        en_file_path = os.path.join(data_dir, d, 'en_sentences_bert_embedding.jsonl')
        # en_file_path = os.path.join(data_dir, d, 'small.jsonl')
        en_domain_sen_emb = bert_sen_feature_extract(en_file_path, en_n)
        ba_sentences = open(os.path.join(data_dir, d, 'ba_sentences.txt'), 'r', encoding='utf8').read().strip().split('\n')
        # ba_sentences = open(os.path.join(data_dir, d, 'small_ba.txt'), 'r', encoding='utf8').read().strip().split('\n')
        ba_domain_sen_random_emb = random_word2vec(ba_sentences, ba_n)
        # print(np.array(en_domain_sen_emb).shape)

        X_2d = np.row_stack((en_domain_sen_emb, ba_domain_sen_random_emb))
        np.save(os.path.join(data_dir, d, 'X_2d.npy'), X_2d)
        print('X_2d.shape:', X_2d.shape)
        X_1d = X_2d.reshape((X_2d.shape[0], -1))
        np.save(os.path.join(data_dir, d, 'X_1d.npy'), X_1d)
        print('X_1d.shape:', X_1d.shape)

        print('en and ba sentences length:', np.sum(en_ba_nums))

def overall_distance(word_embeddings):
    word_embeddings = np.array(word_embeddings)
    ds = []
    for wemb in word_embeddings:
        d = np.sum(np.sqrt(np.sum(np.power(word_embeddings - wemb, 2), axis=1)))/(len(word_embeddings)-1)
        # print(d)
        ds.append(d)
    # print(ds)
    return np.average(ds)

def get_center(word_embeddings):
    word_embeddings = np.array(word_embeddings)
    c = np.average(word_embeddings, axis=0)
    return c

def main2_bert_word_embedding_distance():
    domain_list = ['taxi']  # , 'attraction', 'taxi', 'restaurant']
    data_dir = '../../bert/HG_data'
    # data_dir = '../../multiwoz_bahasawoz_v2/HG_data'
    for d in domain_list:
        print('domain:', d)

        en_ba_nums = np.load(os.path.join(data_dir, d, 'en_ba_sen_nums_metadata.npy'))
        # print('generating H ...')
        # H = generate_H(en_ba_nums)
        # print('H.shape:', H.shape)
        # np.save(os.path.join(data_dir, d, 'H.npy'), H)

        en_n, ba_n = np.sum(en_ba_nums, axis=1)

        # print('generating X ...')
        en_file_path = os.path.join(data_dir, d, 'en_sentences_bert_embedding.jsonl')
        # en_file_path = os.path.join(data_dir, d, 'small.jsonl')
        word_embeddings = {}
        with open(en_file_path, 'r', encoding='utf8') as fd:
            # pbar = tqdm(total=range(np.sum(en_n)))
            for line in fd:
                sen_bert = json.loads(line)
                sen_emb = []

                for f in sen_bert['features']:
                    t, word_emb = f['token'], f['layers'][0]['values']
                    if t in ['[CLS]', '[SEP]']:
                        continue
                    if t not in word_embeddings:
                        word_embeddings[t] = []
                    word_embeddings[t].append(word_emb)
                # pbar.update(1)
            # pbar.close()
        word_list = ['pick', 'take']
        for w in word_list:
            print('word: {}, overall distance: {}'.format(w, overall_distance(word_embeddings[w])))
            # print('word: {}, overall distance: {}'.format(w, get_center(word_embeddings[w]).shape))
        print(overall_distance([get_center(word_embeddings[word_list[0]]), get_center(word_embeddings[word_list[1]])]))


def random_word2vec(sentences, sens_len, word_emb_d=768, max_len=76):
    word_bank = Counter()
    # sentences = [s.strip() for s in sentences]
    pbar = tqdm(total=sens_len)
    for s in sentences:
        pbar.update(1)
        s = s.strip()
        word_bank.update([t.lower() for t in s.split()])
    pbar.close()
    word2idx = {w_f[0]: i for i, w_f in enumerate(word_bank.most_common())}
    idx2word = {i: w_f[0] for i, w_f in enumerate(word_bank.most_common())}
    voc_size = len(word2idx)
    voc_embedding = np.random.uniform(-1, 1, (voc_size, word_emb_d))
    assert (len(voc_embedding) == len(word2idx))
    print('vocabulary size:', len(word2idx))
    sens_emb = []
    pbar = tqdm(total=sens_len)
    for s in sentences:
        pbar.update(1)
        tokens = s.strip().split()
        sen_emb = np.array([voc_embedding[word2idx[t.lower()]] for t in tokens])
        if len(tokens) < max_len:
            sen_emb = np.row_stack((sen_emb, np.zeros((max_len - len(tokens), word_emb_d))))
        else:
            sen_emb = sen_emb[:max_len]
        # print(s)
        # print(sen_emb)
        sens_emb.append(sen_emb)
    pbar.close()
    return np.array(sens_emb)

def test():
    nums = np.array([[5,2,3], [4,4,2]])
    print(generate_H(nums))

    data_dir = '../../multiwoz_bahasawoz_v2/HG_data'
    d = 'hotel'
    en_n, ba_n = 10, 10

    en_file_path = os.path.join(data_dir, d, 'small.jsonl')
    en_domain_sen_emb = bert_sen_feature_extract(en_file_path, en_n)
    # ba_sentences = open(os.path.join(data_dir, d, 'ba_sentence.txt'), 'r', encoding='utf8').read().strip().split('\n')
    ba_sentences = open(os.path.join(data_dir, d, 'small_ba.txt'), 'r', encoding='utf8').read().strip().split('\n')
    ba_domain_sen_random_emb = random_word2vec(ba_sentences, ba_n)
    # print(np.array(en_domain_sen_emb).shape)

    X_2d = np.row_stack((en_domain_sen_emb, ba_domain_sen_random_emb))
    np.save(os.path.join(data_dir, d, 'X_2d.npy'), X_2d)
    print('X_2d.shape:', X_2d.shape)
    print('X_2d dtype:', X_2d.dtype)
    print(X_2d)
    X_1d = X_2d.reshape((X_2d.shape[0], -1))
    np.save(os.path.join(data_dir, d, 'X_1d.npy'), X_1d)
    print('X_1d.shape:', X_1d.shape)

    print('en and ba sentences length:', 20)

def small_test():
    nums = np.array([[5, 5], [ 4, 4], [3,3]])
    print(overall_distance(nums))

if __name__=='__main__':
    # main()
    # test()
    # small_test()
    main2_bert_word_embedding_distance()