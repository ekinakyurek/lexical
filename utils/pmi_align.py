import sys
from collections import Counter
import pdb
import pickle, json
import itertools
import numpy as np
import torch
import torch.nn.functional as F
EPS=1e-7

SPLIT_TOK=" ||| "

#SPLIT_TOK="\t"

def main(data_file, aligner_file):

    counts_xy, counts_x, counts_y  = {}, {}, {}

    with open(data_file, "r") as f:
        for line in f:
            inpstr, outstr = line.split(SPLIT_TOK)
            input, output = set(inpstr.strip().split(" ")), set(outstr.strip().split(" "))
            for x in input:
                counts_x[x] = counts_x.get(x,0) + 1.0
            for y in output:
                counts_y[y] = counts_y.get(y,0) + 1.0
            for (x,y) in itertools.product(input,output):
                counts_xy[(x,y)] = counts_xy.get((x,y),0) + 1.0


    matchings = {}
    word_adjacency = {}
    pmi = {}
    for x in counts_x.keys():
        max_y = -np.inf
        matching = None
        pmi[x] = {}
        for y in counts_y.keys():
            cxy = counts_xy[(x,y)] = np.log(counts_xy.get((x,y),EPS)) - np.log(counts_x[x] + EPS) - np.log(counts_y[y] + EPS)
            pmi[x][y] = cxy
            if cxy > max_y:
                matching=y
                max_y = cxy
        matchings[x] = matching
        
    for (k,vs) in pmi.items():
        
        normalized = F.softmax(torch.tensor(list(vs.values())), dim=0).numpy()
            
        for (i,(v,_)) in enumerate(vs.items()):
            vs[v] = normalized[i] * 100            
        
    with open(aligner_file + '.json', 'w') as handle:
        json.dump(pmi, handle)



if __name__ == "__main__":
    # execute only if run as a script
    assert len(sys.argv) > 2
    main(sys.argv[1], sys.argv[2])
