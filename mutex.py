import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from src import EncDec, Decoder, Vocab, batch_seqs, weight_top_p, RecordLoss, MultiIter
import random
from data import collate, eval_format
import collections
import math
import pdb
LossTrack = collections.namedtuple('LossTrack', 'nll mlogpyx pointkl')

class Mutex(nn.Module):
    def __init__(self,
                 vocab_x,
                 vocab_y,
                 emb,
                 dim,
                 copy=False,
                 temp=1.0,
                 max_len_x=8,
                 max_len_y=8,
                 n_layers=1,
                 self_att=False,
                 attention=True,
                 dropout=0.,
                 bidirectional=True,
                 rnntype=nn.LSTM,
                 kl_lamda=1.0,
                 recorder=RecordLoss(),
                 qxy=None,
                 ):

        super().__init__()

        self.vocab_x = vocab_x
        self.vocab_y = vocab_y
        self.rnntype = rnntype
        self.bidirectional = bidirectional
        self.dim = dim
        self.n_layers = n_layers
        self.temp = temp
        self.MAXLEN_X = max_len_x
        self.MAXLEN_Y = max_len_y
        self.pyx = EncDec(vocab_x,
                          vocab_y,
                          emb,
                          dim,
                          copy=copy,
                          n_layers=n_layers,
                          self_att=self_att,
                          source_att=attention,
                          dropout=dropout,
                          bidirectional=bidirectional,
                          rnntype=rnntype,
                          MAXLEN=self.MAXLEN_Y)
        if qxy:
            self.qxy = EncDec(vocab_y,
                              vocab_x,
                              emb,
                              dim,
                              copy=copy,
                              n_layers=n_layers,
                              self_att=self_att,
                              dropout=dropout,
                              bidirectional=bidirectional,
                              rnntype=rnntype,
                              source_att=attention,
                              MAXLEN=self.MAXLEN_X)
            # self.qxy = None
        self.recorder = recorder

    def forward(self, inp, out, lens=None, recorder=None):
        return self.pyx(inp, out, lens=lens)

    def print_tokens(self, vocab, tokens):
        return [" ".join(eval_format(vocab, tokens[i])) for i in range(len(tokens))]

    def sample(self, *args, **kwargs):
        return self.pyx.sample(*args, **kwargs)
