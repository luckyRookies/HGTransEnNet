
import os
import random
random.seed(2020)

if __name__ == '__main__':
    dir_list = ['attraction', 'hotel', 'taxi', 'restaurant']
    train_per, valid_per, test_per = 0.8, 0.9, 1
    for d in dir_list:
        train, valid, test = [], [], []
        data = os.path.join(d, 'data')
        with open(data, 'r', encoding='utf8') as fd:
            for l in fd:
                l = l.strip()
                if l == '':
                    continue
                rand = random.random()
                if rand <= train_per:
                    train.append(l)
                elif rand <= valid_per:
                    valid.append(l)
                else:
                    test.append(l)
        with open(os.path.join(d, 'train'), 'w', encoding='utf8') as fd:
            fd.writelines([s + '\n' for s in train])
        with open(os.path.join(d, 'valid'), 'w', encoding='utf8') as fd:
            fd.writelines([s + '\n' for s in valid])
        with open(os.path.join(d, 'test'), 'w', encoding='utf8') as fd:
            fd.writelines([s + '\n' for s in test])