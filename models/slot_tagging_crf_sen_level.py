"""Slot Tagger models."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

import models.crf as crf

class LSTMTagger_CRF_sen_level(nn.Module):
    def __init__(self, embedding_dim, max_sen_len, sen_size, hidden_dim, tagset_size, bidirectional=True, num_layers=1, dropout=0., device=None):
        """Initialize model."""
        super(LSTMTagger_CRF_sen_level, self).__init__()

        print('embedding: %s, max_sen_len: %s, sen_size: %s'% (embedding_dim, max_sen_len, sen_size))
        self.embedding_dim = embedding_dim
        self.sen_size = sen_size
        self.max_sen_len = max_sen_len
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        #self.pad_token_idxs = pad_token_idxs
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

        self.num_directions = 2 if self.bidirectional else 1
        self.dropout_layer = nn.Dropout(p=self.dropout)


        self.sen_embeddings = nn.Embedding(self.sen_size, self.embedding_dim * self.max_sen_len)

        # The LSTM takes word embeddings as inputs, and outputs hidden states


        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True, dropout=self.dropout)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(self.num_directions * self.hidden_dim, self.tagset_size + 2)

        self.crf_layer = crf.CRF(self.tagset_size, self.device)
        #self.init_weights()

    def init_weights(self, initrange=0.2):
        """Initialize weights."""


        for weight in self.lstm.parameters():
            weight.data.uniform_(-initrange, initrange)
        self.hidden2tag.weight.data.uniform_(-initrange, initrange)
        self.hidden2tag.bias.data.uniform_(-initrange, initrange)

    def _get_lstm_features(self, sentences, lengths, with_snt_classifier=False):
        # step 1: word embedding

        # sentence embedding
        sen_embeds = self.sen_embeddings(sentences)

        # reshape to max_len * embedding_dim
        embeds = sen_embeds.reshape((sen_embeds.shape[0], -1, self.embedding_dim))

        concat_input = embeds
        # print('concat shape: {}, length: {}'.format(concat_input.shape, len(lengths)))
        concat_input = self.dropout_layer(concat_input)

        # step 2: BLSTM encoder
        packed_embeds = rnn_utils.pack_padded_sequence(concat_input, lengths, batch_first=True)
        packed_lstm_out, packed_h_t_c_t = self.lstm(packed_embeds)  # bsize x seqlen x dim
        lstm_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_lstm_out, batch_first=True)

        lstm_out_reshape = lstm_out.contiguous().view(lstm_out.size(0)*lstm_out.size(1), lstm_out.size(2))
        tag_space = self.hidden2tag(self.dropout_layer(lstm_out_reshape))
        tag_space = tag_space.view(lstm_out.size(0), lstm_out.size(1), tag_space.size(1))

        if with_snt_classifier:
            return tag_space, (packed_h_t_c_t, lstm_out, lengths)
        else:
            return tag_space

    def neg_log_likelihood(self, feats, masks, tags):
        return self.crf_layer.neg_log_likelihood_loss(feats, masks, tags)

    def forward(self, feats, masks):
        path_score, best_path = self.crf_layer._viterbi_decode(feats, masks)
        return path_score, best_path

    def load_model(self, load_dir):
        if self.device.type == 'cuda':
            self.load_state_dict(torch.load(open(load_dir, 'rb')))
        else:
            self.load_state_dict(torch.load(open(load_dir, 'rb'), map_location=lambda storage, loc: storage))

    def save_model(self, save_dir):
        torch.save(self.state_dict(), open(save_dir, 'wb'))

