# -*- coding: utf-8 -*-
import os
import json
import re
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)  # ,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_voacb(file_path):
    w_freq = {}
    with open(file_path, 'r', encoding='utf8') as f:
        for l in f:
            l = l.strip()
            if l not in w_freq:
                w_freq[l] = 0
    return w_freq

def format_data():
    dir_list = ['hotel', 'taxi']  # ['hotel', 'attraction', 'taxi', 'restaurant']
    data_dir = '../data/bahasa_both'
    data_file = 'data'
    slot_file = 'vocab.slot'
    intent_file = 'vocab.intent'

    for d in dir_list:
        print('domain:', d)
        slot_freq = get_voacb(os.path.join(data_dir, d, slot_file))
        intent_freq = get_voacb(os.path.join(data_dir, d, intent_file))

        with open(os.path.join(data_dir, d, data_file), 'r', encoding='utf8') as f:
            for l in f:
                l = l.strip()
                word_slot_str, intents = l.split(' <=> ')
                intent = intents.split(';')[0]
                if intent in intent_freq:
                    intent_freq[intent] += 1
                else:
                    intent_freq[intent] = 1
                for w_s in word_slot_str.split(' '):
                    ws_s = w_s.split(':')
                    if len(ws_s) == 2:
                        w, s = ws_s
                    if len(ws_s) > 2:
                        w, s = ':'.join(ws_s[:-1]), ws_s[-1]
                    if s.startswith('B'):
                        if s in slot_freq:
                            slot_freq[s] += 1
                        else:
                            slot_freq[s] = 1
            print('intent collection:', sorted(intent_freq.items(), key=lambda k:k[1], reverse=True))
            print('slot collection:', sorted(slot_freq.items(), key=lambda k:k[1], reverse=True))



if __name__ == '__main__':
    format_data()

