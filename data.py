import torch
from torch import nn, optim
import torch.utils.data as torch_data
import torch.nn.functional as F
import numpy as np
import random
import sys
from src import batch_seqs
import pdb
EPS = 1e-7

def encode(data, vocab_x, vocab_y):
    encoded = []
    for datum in data:
        encoded.append(encode_io(datum, vocab_x, vocab_y))
    return encoded

def encode_io(datum, vocab_x, vocab_y):
    inp, out = datum
    return ([vocab_x.sos()] + vocab_x.encode(inp) + [vocab_x.eos()], [vocab_y.sos()] + vocab_y.encode(out) + [vocab_y.eos()])

def eval_format(vocab, seq):
    if vocab.eos() in seq:
        seq = seq[:seq.index(vocab.eos())+1]
    seq = seq[1:-1]
    return vocab.decode(seq)

def collate(batch):
    batch = sorted(batch,
                   key=lambda x: len(x[0]),
                   reverse=True)
    inp, out = zip(*batch)
    lens = torch.LongTensor(list(map(len,inp)))
    inp = batch_seqs(inp)
    out = batch_seqs(out)
    return inp, out, lens

def get_fig2_exp(input_symbols, output_symbols):
    study = [(["dax"], ["RED"]),
             (["lug"],["BLUE"]),
             (["wif"],["GREEN"]),
             (["zup"],["YELLOW"]),
             (["lug","fep"],["BLUE","BLUE","BLUE"]),
             (["dax","fep"],["RED","RED","RED"]),
             (["lug","blicket","wif"],["BLUE","GREEN","BLUE"]),
             (["wif","blicket","dax"],["GREEN","RED","GREEN"]),
             (["lug","kiki","wif"],["GREEN","BLUE"]),
             (["dax","kiki","lug"],["BLUE","RED"]),
             (["lug","fep","kiki","wif"],["GREEN","BLUE","BLUE","BLUE"]),
             (["wif","kiki","dax","blicket","lug"],["RED","BLUE","RED","GREEN"]),
             (["lug","kiki","wif","fep"],["GREEN","GREEN","GREEN","BLUE"]),
             (["wif","blicket","dax","kiki","lug"],["BLUE","GREEN","RED","GREEN"])]

    test = [(["zup","fep"],["YELLOW","YELLOW","YELLOW"]),
            (["zup","blicket","lug"],["YELLOW","BLUE","YELLOW"]),
            (["dax","blicket","zup"],["RED","YELLOW","RED"]),
            (["zup","kiki","dax"],["RED","YELLOW"]),
            (["wif","kiki","zup"],["YELLOW","GREEN"]),
            (["zup","fep","kiki","lug"],["BLUE","YELLOW","YELLOW","YELLOW"]),
            (["wif","kiki","zup","fep"],["YELLOW","YELLOW","YELLOW","GREEN"]),
            (["lug","kiki","wif","blicket","zup"],["GREEN","YELLOW","GREEN","BLUE"]),
            (["zup","blicket","wif","kiki","dax","fep"],["RED","RED","RED","YELLOW","GREEN","YELLOW"]),
            (["zup","blicket","zup","kiki","zup","fep"],["YELLOW","YELLOW","YELLOW","YELLOW","YELLOW","YELLOW"])]
    return study, test
